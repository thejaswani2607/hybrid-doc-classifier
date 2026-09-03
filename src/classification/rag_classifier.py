"""
RAG (Retrieval-Augmented) classifier — searches the FAISS index built from
training documents to find the most similar training examples and predict
a class by (weighted) majority vote among the top_k nearest neighbours.

Uses COSINE SIMILARITY (matching how the index was built): the index stores
L2-normalized vectors and uses inner product search, so a raw search score
IS already a cosine similarity in [-1, 1] — no distance-to-similarity
conversion needed, unlike the old L2-distance approach.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.settings import get_settings

logger = logging.getLogger(__name__)

_model_cache = {}


def _get_embedding_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return vectors / norms


class RagClassifier:
    def __init__(self, version: str, embedding_model: str = None, top_k: int = 3):
        settings = get_settings()
        self.settings = settings
        self.version = version
        self.embedding_model_name = embedding_model or settings.classifier.get(
            "embedding_model", "all-MiniLM-L6-v2"
        )
        self.top_k = top_k

        index_path = settings.data_store / version / "index.faiss"
        metadata_path = settings.data_store / version / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"FAISS index or metadata not found for version '{version}'. "
                f"Run 'build-index --version {version}' first."
            )

        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.documents = meta["documents"]
        self.metric = meta.get("metric", "l2")  # "cosine" for new indexes, "l2" for old ones
        self.model = _get_embedding_model(self.embedding_model_name)

    def search(self, text: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Returns top_k nearest training docs: [{class_name, confidence, file_path}, ...]"""
        top_k = top_k or self.top_k
        if not text or not text.strip():
            return []

        query_vec = self.model.encode([text]).astype("float32")

        if self.metric == "cosine":
            query_vec = _normalize(query_vec)

        scores, indices = self.index.search(query_vec, min(top_k, len(self.documents)))

        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue

            if self.metric == "cosine":
                # Inner product on normalized vectors IS cosine similarity,
                # already in a sensible [-1, 1] range (practically [0, 1]
                # for text embeddings). Clip defensively in case of float noise.
                similarity = float(max(0.0, min(1.0, score)))
            else:
                # Legacy L2-distance index (backward compatibility for
                # indexes built before this change)
                similarity = 1.0 / (1.0 + float(score))

            doc = self.documents[idx]
            hits.append({
                "class_name": doc["class_label"],
                "confidence": similarity,
                "file_path": doc["file_path"],
            })
        return hits

    def classify(self, text: str, top_k: int = None) -> Tuple[str, float, List[Dict[str, Any]]]:
        """
        Returns (predicted_class, confidence, rag_hints).

        Confidence logic: take the top hit's similarity as the base score,
        but boost slightly if multiple of the top_k neighbours agree on the
        same class (more agreement = more trustworthy RAG prediction).
        """
        hits = self.search(text, top_k=top_k)
        if not hits:
            return "UNKNOWN", 0.0, []

        top_class = hits[0]["class_name"]
        top_conf = hits[0]["confidence"]

        agreeing = [h for h in hits if h["class_name"] == top_class]
        agreement_ratio = len(agreeing) / len(hits)
        confidence = min(1.0, top_conf * (0.85 + 0.15 * agreement_ratio))

        return top_class, confidence, hits