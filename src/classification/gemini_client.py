"""
Thin wrapper around google-generativeai for the Gemini FREE-TIER API key.

Adds:
  - Automatic MODEL FALLBACK: Google periodically retires model versions
    with little notice (this project has already had to migrate off
    gemini-2.5-flash, gemini-3.6-flash, and gemini-2.5-flash-lite during
    development). Rather than hardcoding one model name that can break
    again at any time, this tries a list of candidates in order and
    remembers whichever one actually works for the rest of the run.
  - A shared rate limiter so ALL worker threads cooperate against the same
    per-minute quota, instead of each thread independently retrying and
    colliding with the others.
  - Automatic retry with exponential backoff on rate-limit (429) errors.
  - A running counter of API calls made, so the Excel report can show
    "Gemini calls used" instead of a dollar cost (this pipeline is $0).
  - JSON-schema-constrained generation for reliable class_name/confidence
    parsing.
"""
import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

import google.generativeai as genai

from src.settings import get_settings

logger = logging.getLogger(__name__)


class GeminiUsageTracker:
    def __init__(self):
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.rate_limit_hits = 0

    def record(self, input_tokens: int = 0, output_tokens: int = 0):
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def record_rate_limit_hit(self):
        self.rate_limit_hits += 1

    def summary(self) -> Dict[str, Any]:
        return {
            "gemini_calls": self.total_calls,
            "gemini_input_tokens": self.total_input_tokens,
            "gemini_output_tokens": self.total_output_tokens,
            "gemini_rate_limit_retries": self.rate_limit_hits,
        }


usage_tracker = GeminiUsageTracker()

_configured = False


def _ensure_configured():
    global _configured
    if _configured:
        return
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
            "free API key from https://aistudio.google.com/apikey"
        )
    genai.configure(api_key=settings.gemini_api_key)
    _configured = True


class _SharedRateLimiter:
    """Ensures ALL worker threads share the same per-minute request budget."""
    def __init__(self, requests_per_minute: int):
        self.min_interval = 60.0 / max(1, requests_per_minute)
        self._lock = threading.Lock()
        self._last_call_time = 0.0

    def wait_for_slot(self):
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_call_time = time.time()


_rate_limiter: Optional[_SharedRateLimiter] = None


def _get_rate_limiter() -> _SharedRateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        rpm = int(os.getenv("GEMINI_RPM", "5"))
        _rate_limiter = _SharedRateLimiter(requests_per_minute=rpm)
        logger.info(f"Gemini rate limiter set to {rpm} requests/minute (GEMINI_RPM)")
    return _rate_limiter


# Ordered fallback list: the configured model (from .env / classifier_config.yaml)
# is tried first, then these known-generally-available models in order, in case
# the configured one has been retired. Update this list if all of these ever
# stop working too — check https://ai.google.dev/gemini-api/docs/models
MODEL_FALLBACK_CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]

# Remembers the first model that actually worked, so we don't re-try dead
# models on every single call once we've found a working one.
_working_model_name: Optional[str] = None
_model_lock = threading.Lock()


def _is_model_unavailable_error(error_str: str) -> bool:
    lowered = error_str.lower()
    return "404" in error_str and ("no longer available" in lowered or "not found" in lowered)


CLASSIFY_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "class_name": {"type": "STRING"},
        "confidence": {"type": "NUMBER"},
    },
    "required": ["class_name", "confidence"],
}


def _build_prompt(text: str, candidate_labels_with_descriptions: Dict[str, str]) -> str:
    classes_block = "\n".join(
        f"- {label}: {desc}" for label, desc in sorted(candidate_labels_with_descriptions.items())
    )
    return f"""
Analyze the provided document text and classify it into exactly ONE of the following categories.
Use 'UNKNOWN' if none clearly match.

Allowed Categories:
{classes_block}

Match the label spelling exactly as given above.
Return a JSON object with 'class_name' (must be one of the labels above or 'UNKNOWN')
and 'confidence' (a number from 0.0 to 1.0).
If you have even the slightest doubt, return UNKNOWN with confidence 1.0.

--- Document Content ---
{text[:10000]}
---
""".strip()


def classify_with_gemini(
    text: str,
    candidate_labels_with_descriptions: Dict[str, str],
    model_name: Optional[str] = None,
    temperature: float = 0,
    max_retries: int = 5,
) -> Dict[str, Any]:
    """
    Asks Gemini to pick the best class for the given document text.
    Automatically falls back across MODEL_FALLBACK_CANDIDATES if the
    configured/preferred model has been retired by Google.

    Returns: {"class_name": str, "confidence": float, "input_tokens": int,
              "output_tokens": int, "error": Optional[str]}
    """
    global _working_model_name

    _ensure_configured()
    settings = get_settings()
    rate_limiter = _get_rate_limiter()

    preferred_model = model_name or settings.gemini_model

    with _model_lock:
        if _working_model_name:
            models_to_try = [_working_model_name]
        else:
            models_to_try = [preferred_model] + [
                m for m in MODEL_FALLBACK_CANDIDATES if m != preferred_model
            ]

    generation_config = {
        "temperature": temperature,
        "response_mime_type": "application/json",
        "response_schema": CLASSIFY_RESPONSE_SCHEMA,
    }
    prompt = _build_prompt(text, candidate_labels_with_descriptions)

    last_error = None

    for candidate_model in models_to_try:
        model = genai.GenerativeModel(candidate_model)

        for attempt in range(max_retries):
            rate_limiter.wait_for_slot()

            try:
                response = model.generate_content(prompt, generation_config=generation_config)
                usage = getattr(response, "usage_metadata", None)
                input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
                output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
                usage_tracker.record(input_tokens, output_tokens)

                result = json.loads(response.text)
                class_name = result.get("class_name", "UNKNOWN")
                confidence = float(result.get("confidence", 0.0))

                if class_name not in candidate_labels_with_descriptions and class_name != "UNKNOWN":
                    class_name = "UNKNOWN"
                    confidence = 0.0

                # Remember this model worked, so future calls skip straight to it.
                if _working_model_name != candidate_model:
                    with _model_lock:
                        if not _working_model_name:
                            _working_model_name = candidate_model
                            logger.info(f"Gemini model confirmed working: {candidate_model}")

                return {
                    "class_name": class_name,
                    "confidence": confidence,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "error": None,
                }

            except Exception as e:
                error_str = str(e)
                last_error = error_str

                if _is_model_unavailable_error(error_str):
                    logger.warning(
                        f"Model '{candidate_model}' unavailable, trying next fallback candidate..."
                    )
                    break  # stop retrying THIS model, move to next candidate

                is_rate_limit = "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower()
                if is_rate_limit and attempt < max_retries - 1:
                    usage_tracker.record_rate_limit_hit()
                    wait_time = min(2 ** attempt * 3, 60)
                    logger.warning(
                        f"Gemini free-tier rate limit hit (attempt {attempt + 1}/{max_retries}). "
                        f"Waiting {wait_time}s before retry..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Gemini classification failed: {error_str}")
                    break

    return {
        "class_name": "UNKNOWN",
        "confidence": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "error": last_error,
    }