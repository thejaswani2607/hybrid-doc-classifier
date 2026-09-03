"""
Caches extracted text by file content hash (MD5), so re-running evaluation
on the same test set doesn't re-run OCR/extraction every time.

Cache key = MD5 of file bytes -> if the file content changes, the hash
changes, and it's treated as a new document automatically.
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ExtractionCacheManager:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _file_hash(self, file_path: str) -> str:
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _cache_path(self, file_hash: str) -> Path:
        return self.cache_dir / f"{file_hash}.json"

    def get(self, file_path: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        try:
            file_hash = self._file_hash(file_path)
            cache_file = self._cache_path(file_hash)
            if cache_file.exists():
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                self.hits += 1
                return cached["text"], cached["metadata"]
        except Exception as e:
            logger.warning(f"Cache lookup failed for {file_path}: {e}")
        self.misses += 1
        return None

    def set(self, file_path: str, text: str, metadata: Dict[str, Any]):
        try:
            file_hash = self._file_hash(file_path)
            cache_file = self._cache_path(file_hash)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"text": text, "metadata": metadata}, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Cache write failed for {file_path}: {e}")

    def print_stats(self):
        total = self.hits + self.misses
        rate = (self.hits / total * 100) if total else 0
        print(f"\nExtraction Cache: {self.hits} hits, {self.misses} misses ({rate:.1f}% hit rate)")
