"""
Builds a FAISS index from a data_he_extractor.json file, using COSINE
SIMILARITY (normalized vectors + inner product) rather than raw L2
distance — sentence-transformers models are trained/tuned for cosine
similarity, so this better reflects what the embedding model actually
considers "similar," typically giving more meaningful confidence scores.

Output (matching original project's structure):
    store/<version>/index.faiss     - the vector index
    store/<version>/metadata.json   - {"documents": [...], "embedding_model", "dimension", "metric": "cosine"}
"""
import json
import logging
from pathlib import Path
from typing import Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.settings import get_settings

logger = logging.getLogger(__name__)

_model_cache = {}


def _get_embedding_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        logger.info(f"Loading embedding model: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row so inner product = cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10  # avoid division by zero for empty-text docs
    return vectors / norms


def build_index(version: str, embedding_model: str = None) -> Tuple[str, str]:
    settings = get_settings()
    embedding_model = embedding_model or settings.classifier.get("embedding_model", "all-MiniLM-L6-v2")

    data_path = settings.data_store / version / "data_he_extractor.json"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}. Run 'extract-data' for this version first."
        )

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError(f"No documents found in {data_path}")

    logger.info(f"Loaded {len(data)} documents for indexing")

    model = _get_embedding_model(embedding_model)

    texts = [item["data"] or "" for item in data]
    logger.info("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings).astype("float32")
    embeddings = _normalize(embeddings)  # <-- normalize for cosine similarity

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # <-- inner product on normalized vectors = cosine similarity
    index.add(embeddings)

    output_dir = settings.data_store / version
    index_path = output_dir / "index.faiss"
    metadata_path = output_dir / "metadata.json"

    faiss.write_index(index, str(index_path))

    metadata = {
        "documents": [
            {"file_path": item["file_path"], "class_label": item["class_label"]}
            for item in data
        ],
        "embedding_model": embedding_model,
        "dimension": dimension,
        "metric": "cosine",
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Index saved: {index_path} ({index.ntotal} vectors, dim={dimension}, metric=cosine)")
    return str(index_path), str(metadata_path)