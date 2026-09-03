"""
Multi-Layered Multi-Class Classifier (MLMC) — mirrors the original project's
hierarchical classification strategy:

LAYER 1 (standalone classes + bundle parents):
    1. Extract document text (HE Extractor, cached by content hash).
    2. Run RAG (FAISS search over training embeddings) -> (rag_class, rag_confidence).
    3. If rag_confidence >= layer1_rag_confidence_threshold AND the class is
       NOT in force_llm_classes -> accept the RAG result, done.
    4. Otherwise, call Gemini with the full candidate list (standalone +
       bundle parent labels) -> (gemini_class, gemini_confidence). Gemini's
       result becomes the final Layer 1 prediction.

LAYER 2 (only if Layer 1 predicted a BUNDLE parent):
    1. Split the document into pages.
    2. Classify each page independently against ONLY that bundle's children
       (Gemini only, per configured layer2_strategy).
    3. Merge consecutive pages predicted as the same child class into a
       single (start_page, end_page) result.

Returns a ClassificationResult (or list of them, for bundles with multiple
children) with full metadata for reporting.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.settings import get_settings
from src.extraction.text_extractors import extract_text, extract_pdf_pages
from src.classification.rag_classifier import RagClassifier
from src.classification.gemini_client import classify_with_gemini
from src.classification.cache_manager import ExtractionCacheManager

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    class_name: str
    confidence: float
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None
    data: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class MLMCClassifier:
    def __init__(self, version: str, enable_cache: bool = True):
        settings = get_settings()
        self.settings = settings
        self.cfg = settings.classifier
        self.version = version

        self.layer1_categories: Dict[str, Any] = self.cfg.get("layer1_categories", {}) or {}
        self.layer1_standalone: List[str] = self.cfg.get("layer1_standalone", []) or []
        self.force_llm_classes: List[str] = self.cfg.get("force_llm_classes", []) or []

        self.layer1_rag_threshold = float(self.cfg.get("layer1_rag_confidence_threshold", 0.67))
        self.layer1_top_k = int(self.cfg.get("layer1_top_k", 3))
        self.max_pages = int(self.cfg.get("max_pages", 20) or 0)

        self.gemini_model = settings.gemini_model
        self.temperature = float((self.cfg.get("llm_generation_config") or {}).get("temperature", 0))

        # All Layer-1 candidate labels: standalone classes + bundle parent names.
        self.layer1_labels = list(self.layer1_standalone) + list(self.layer1_categories.keys())
        self.label_descriptions = settings.class_label_to_description

        self.rag = RagClassifier(
            version=version,
            embedding_model=self.cfg.get("embedding_model", "all-MiniLM-L6-v2"),
            top_k=self.layer1_top_k,
        )

        self.cache = ExtractionCacheManager(str(settings.cache_dir)) if enable_cache else None

    # ------------------------------------------------------------------
    def _extract(self, file_path: str) -> tuple:
        if self.cache:
            cached = self.cache.get(file_path)
            if cached is not None:
                return cached
        text, meta = extract_text(file_path, max_pages=self.max_pages)
        if self.cache:
            self.cache.set(file_path, text, meta)
        return text, meta

    def _layer1_candidate_descriptions(self) -> Dict[str, str]:
        """Descriptions for Layer 1 candidates: standalone classes + bundle parents."""
        descriptions = {}
        for label in self.layer1_labels:
            descriptions[label] = self.label_descriptions.get(
                label, f"Document class '{label}'."
            )
        if "UNKNOWN" not in descriptions:
            descriptions["UNKNOWN"] = "Default catch-all when nothing else matches."
        return descriptions

    # ------------------------------------------------------------------
    def classify(self, file_path: str) -> List[ClassificationResult]:
        """
        Classifies a single document. Returns a list because bundle documents
        produce multiple results (one per resolved child + page range).
        """
        text, ext_meta = self._extract(file_path)

        if not text or not text.strip():
            return [ClassificationResult(
                class_name="UNKNOWN",
                confidence=0.0,
                error="No text extracted from document",
                metadata={"layer": 1, "method": "empty_extraction", "extraction_metadata": ext_meta},
            )]

        # ---------------- LAYER 1 ----------------
        rag_class, rag_confidence, rag_hits = self.rag.classify(text, top_k=self.layer1_top_k)

        force_gemini = rag_class in self.force_llm_classes
        use_rag_result = (rag_confidence >= self.layer1_rag_threshold) and not force_gemini

        gemini_result = None
        if not use_rag_result:
            candidates = self._layer1_candidate_descriptions()
            gemini_result = classify_with_gemini(
                text=text,
                candidate_labels_with_descriptions=candidates,
                model_name=self.gemini_model,
                temperature=self.temperature,
            )
            final_class = gemini_result["class_name"]
            final_confidence = gemini_result["confidence"]
            method = "gemini" if not force_gemini else "gemini_forced"
        else:
            final_class = rag_class
            final_confidence = rag_confidence
            method = "rag_only"

        layer1_meta = {
            "layer": 1,
            "method": method,
            "rag_class": rag_class,
            "rag_confidence": rag_confidence,
            "rag_hints": rag_hits,
            "gemini_class": gemini_result["class_name"] if gemini_result else None,
            "gemini_confidence": gemini_result["confidence"] if gemini_result else None,
            "extraction_metadata": ext_meta,
        }

        # ---------------- LAYER 2 (bundle resolution) ----------------
        if final_class in self.layer1_categories:
            return self._resolve_bundle(
                file_path=file_path,
                bundle_name=final_class,
                layer1_meta=layer1_meta,
                gemini_result=gemini_result,
            )

        input_tok = gemini_result["input_tokens"] if gemini_result else 0
        output_tok = gemini_result["output_tokens"] if gemini_result else 0
        error = gemini_result["error"] if gemini_result else None

        return [ClassificationResult(
            class_name=final_class,
            confidence=final_confidence,
            input_tokens=input_tok,
            output_tokens=output_tok,
            error=error,
            data=text,
            metadata=layer1_meta,
        )]

    # ------------------------------------------------------------------
    def _resolve_bundle(
        self,
        file_path: str,
        bundle_name: str,
        layer1_meta: Dict[str, Any],
        gemini_result: Optional[Dict[str, Any]],
    ) -> List[ClassificationResult]:
        """
        Layer 2: split the bundle document page-wise and classify each page
        into one of the bundle's configured children (Gemini only).
        """
        children_cfg = self.layer1_categories[bundle_name].get("children", [])
        child_labels = [c["class"] for c in children_cfg]
        child_descriptions = {
            label: self.label_descriptions.get(label, f"Bundle child class '{label}' of {bundle_name}.")
            for label in child_labels
        }

        ext = Path(file_path).suffix.lower()
        if ext != ".pdf":
            # Non-PDF bundles: can't page-split, classify the whole thing as one child.
            text, _ = self._extract(file_path)
            result = classify_with_gemini(
                text=text,
                candidate_labels_with_descriptions=child_descriptions,
                model_name=self.gemini_model,
                temperature=self.temperature,
            )
            meta = dict(layer1_meta)
            meta.update({"layer": 2, "method": "gemini_only", "parent_category": bundle_name})
            return [ClassificationResult(
                class_name=result["class_name"],
                confidence=result["confidence"],
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                error=result["error"],
                data=text,
                metadata=meta,
            )]

        page_texts = extract_pdf_pages(file_path, max_pages=self.max_pages)

        page_predictions = []
        total_in_tok = gemini_result["input_tokens"] if gemini_result else 0
        total_out_tok = gemini_result["output_tokens"] if gemini_result else 0

        for page_idx, page_text in enumerate(page_texts):
            if not page_text.strip():
                page_predictions.append(("UNKNOWN", 0.0))
                continue
            result = classify_with_gemini(
                text=page_text,
                candidate_labels_with_descriptions=child_descriptions,
                model_name=self.gemini_model,
                temperature=self.temperature,
            )
            page_predictions.append((result["class_name"], result["confidence"]))
            total_in_tok += result["input_tokens"]
            total_out_tok += result["output_tokens"]

        # Merge consecutive pages of the same predicted class into ranges.
        results: List[ClassificationResult] = []
        if not page_predictions:
            return [ClassificationResult(
                class_name="UNKNOWN",
                confidence=0.0,
                error="No pages found to resolve bundle",
                metadata={**layer1_meta, "layer": 2, "parent_category": bundle_name},
            )]

        start_idx = 0
        current_class, current_conf = page_predictions[0]
        confs = [current_conf]

        def _flush(start, end, cls, conf_list):
            meta = dict(layer1_meta)
            meta.update({
                "layer": 2,
                "method": "gemini_only",
                "parent_category": bundle_name,
            })
            results.append(ClassificationResult(
                class_name=cls,
                confidence=sum(conf_list) / len(conf_list) if conf_list else 0.0,
                start_page=start + 1,  # 1-indexed for readability
                end_page=end + 1,
                input_tokens=0,
                output_tokens=0,
                metadata=meta,
            ))

        for idx in range(1, len(page_predictions)):
            cls, conf = page_predictions[idx]
            if cls == current_class:
                confs.append(conf)
                continue
            _flush(start_idx, idx - 1, current_class, confs)
            start_idx = idx
            current_class = cls
            confs = [conf]

        _flush(start_idx, len(page_predictions) - 1, current_class, confs)

        # Attribute total token usage to the first child result (avoid duplicate counting).
        if results:
            results[0].input_tokens = total_in_tok
            results[0].output_tokens = total_out_tok

        return results
