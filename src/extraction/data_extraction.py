"""
Walks a train/test folder structured as:

    input_dir/
    ├── CLASS_NAME_1/
    │   ├── doc1.pdf
    │   └── doc2.png
    ├── CLASS_NAME_2/
    │   └── ...

Extracts text from every supported file (parallel, threaded) and writes
data_he_extractor.json in the SAME SCHEMA as the original project's output:

    {
        "file_path": "...",
        "file_name": "...",
        "class_label": "...",
        "data": "<extracted text>",
        "version": "...",
        "extraction_metadata": {
            "total_pages": int,
            "pages_processed": int,
            "free_text_pages": int,
            "ocr_pages": int
        }
    }

Also writes extraction_warnings.txt alongside it, listing any documents
that came back with empty/near-empty text — these are worth checking
manually (corrupted file, unreadable scan, wrong file type, etc.).
"""
import json
import logging
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional

from tqdm import tqdm

from src.settings import get_settings
from src.extraction.text_extractors import extract_text

logger = logging.getLogger(__name__)

# A document with fewer than this many extracted characters is flagged as
# "suspiciously empty" in the warnings report (still saved either way).
MIN_CHARS_FOR_HEALTHY_EXTRACTION = 15


def _process_one(file_path: Path, class_label: str, version: str, max_pages: int) -> Optional[Dict[str, Any]]:
    try:
        text, ext_meta = extract_text(str(file_path), max_pages=max_pages)
        return {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "class_label": class_label,
            "data": text,
            "version": version,
            "extraction_metadata": ext_meta,
        }
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return None


def extract_and_save_data(
    input_dir: Path,
    version: str,
    max_pages: int = 0,
    max_workers: int = 8,
) -> str:
    """
    Extracts text from every document under input_dir/<CLASS>/* and writes
    store/<version>/data_he_extractor.json

    Returns the path to the written JSON file.
    """
    settings = get_settings()
    output_dir = settings.data_store / version
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "data_he_extractor.json"
    warnings_file = output_dir / "extraction_warnings.txt"

    files_to_process = []
    for class_dir in sorted(input_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_label = class_dir.name
        for file_path in class_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in settings.allowed_file_types:
                continue
            files_to_process.append((file_path, class_label))

    logger.info(f"Found {len(files_to_process)} files to extract under {input_dir}")

    documents = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_one, fp, cls, version, max_pages)
            for fp, cls in files_to_process
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Extracting text"):
            result = future.result()
            if result:
                documents.append(result)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    # --- Build a plain-English warnings report for suspiciously empty docs ---
    problem_docs = [
        d for d in documents
        if len((d.get("data") or "").strip()) < MIN_CHARS_FOR_HEALTHY_EXTRACTION
    ]
    with open(warnings_file, "w", encoding="utf-8") as f:
        f.write(f"Extraction warnings for version: {version}\n")
        f.write(f"Total documents processed: {len(documents)}\n")
        f.write(f"Documents with little/no extracted text: {len(problem_docs)}\n")
        f.write("=" * 70 + "\n\n")
        if not problem_docs:
            f.write("None — every document produced readable text.\n")
        else:
            for d in problem_docs:
                meta = d.get("extraction_metadata", {})
                f.write(f"File:  {d['file_path']}\n")
                f.write(f"Class: {d['class_label']}\n")
                f.write(f"Pages: {meta.get('total_pages')} total, "
                        f"{meta.get('free_text_pages')} free-text, "
                        f"{meta.get('ocr_pages')} OCR'd\n")
                f.write(f"Extracted chars: {len((d.get('data') or '').strip())}\n")
                f.write("-" * 70 + "\n")

    logger.info(f"Saved extracted data for {len(documents)} documents to {output_file}")
    if problem_docs:
        logger.warning(
            f"{len(problem_docs)} document(s) had little/no extracted text. "
            f"See {warnings_file} for the full list."
        )
    return str(output_file)