
"""
Central settings loader.

Combines:
  - .env (secrets: GEMINI_API_KEY, TESSERACT_CMD, etc.)
  - config/classifier_config.yaml (thresholds, layer1/layer2 structure)
  - config/document_classes.yaml (class labels + descriptions)

Everything else in the project reads settings from here rather than
re-parsing files, so there is exactly one source of truth.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")


def _resolve_document_classes_path() -> Path:
    """
    Looks for document_classes.yaml first; falls back to the .example.yaml
    template so the project still runs (with template classes) even before
    the user has dropped in their real file.
    """
    override = os.getenv("DOCUMENT_CLASSES_PATH")
    if override:
        return Path(override)

    real = PROJECT_ROOT / "config" / "document_classes.yaml"
    if real.exists():
        return real

    template = PROJECT_ROOT / "config" / "document_classes.example.yaml"
    return template


@dataclass
class Settings:
    # --- Secrets / environment ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    tesseract_cmd: str = "tesseract"
    tesseract_languages: str = "eng"

    # --- Paths ---
    project_root: Path = PROJECT_ROOT
    data_store: Path = PROJECT_ROOT / "store"
    cache_dir: Path = PROJECT_ROOT / "extraction_cache"
    results_dir: Path = PROJECT_ROOT / "classification_results"

    # --- Allowed input file types ---
    # NOTE: Excel (.xlsx/.xls) support was removed — a small number of
    # source files used inconsistent/non-standard export formats that
    # weren't worth the added extraction complexity. PDFs and images cover
    # the vast majority of documents in this pipeline.
    allowed_file_types: tuple = (".pdf", ".png", ".jpg", ".jpeg")

    # --- Classifier config (from classifier_config.yaml) ---
    classifier: Dict[str, Any] = field(default_factory=dict)

    # --- Document classes (from document_classes.yaml) ---
    document_classes: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def class_label_to_description(self) -> Dict[str, str]:
        return {
            dc["label"]: dc.get("description", "")
            for dc in self.document_classes
            if dc.get("label")
        }

    @property
    def valid_class_labels(self):
        return set(self.class_label_to_description.keys())


def load_yaml(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_settings: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    global _settings
    if _settings is not None and not reload:
        return _settings

    classifier_config_path = PROJECT_ROOT / "config" / "classifier_config.yaml"
    classifier_cfg = load_yaml(classifier_config_path) or {}

    doc_classes_path = _resolve_document_classes_path()
    document_classes = load_yaml(doc_classes_path) or []

    settings = Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", classifier_cfg.get("model_name", "gemini-2.5-flash")),
        tesseract_cmd=os.getenv("TESSERACT_CMD", "tesseract"),
        tesseract_languages=os.getenv("TESSERACT_LANGUAGES", "eng"),
        classifier=classifier_cfg,
        document_classes=document_classes,
    )

    settings.data_store.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)

    _settings = settings
    return settings