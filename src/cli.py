"""
Command-line interface for the document classification pipeline.

Commands:
    extract-data   - extract text from a train/test folder -> data_he_extractor.json
    build-index    - build FAISS index from extracted data
    evaluate       - run MLMC classification over a LABELLED test folder,
                     produce an Excel accuracy report (train/test workflow)
    classify       - classify a SINGLE unlabeled document (production-style
                     inference; no ground truth, just a prediction)
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.settings import get_settings
from src.extraction.data_extraction import extract_and_save_data
from src.embeddings.index_builder import build_index as _build_index
from src.classification.mlmc_classifier import MLMCClassifier
from src.classification.gemini_client import usage_tracker
from src.reporting.excel_report import build_report, _compute_document_level_dependency

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(help="Zero-budget document classification pipeline (RAG + Gemini free-tier).")
console = Console()


# ======================================================================
# extract-data
# ======================================================================
@app.command("extract-data")
def extract_data(
    input_dir: Path = typer.Argument(..., help="Folder with one subfolder per class (e.g. store/train)"),
    version: str = typer.Option(..., "--version", "-v", help="Version name for this dataset snapshot, e.g. cl_v1"),
    num_pages: int = typer.Option(0, "--num-pages", "-n", help="Max pages per document (0 = all)"),
    workers: int = typer.Option(8, "--workers", "-w", help="Parallel extraction workers"),
):
    """Extract text from all documents under input_dir/<CLASS>/* into data_he_extractor.json"""
    if not input_dir.exists():
        console.print(f"[red]Input directory not found: {input_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Extracting text from {input_dir} (version={version})...[/cyan]")
    output_path = extract_and_save_data(
        input_dir=input_dir, version=version, max_pages=num_pages, max_workers=workers
    )
    console.print(f"[green]✨ Extraction complete![/green] Saved to: {output_path}")


# ======================================================================
# build-index
# ======================================================================
@app.command("build-index")
def build_index_cmd(
    version: str = typer.Option(..., "--version", "-v", help="Version name matching a prior extract-data run"),
    embedding_model: Optional[str] = typer.Option(None, "--embedding-model", "-e"),
):
    """Build a FAISS index from a previously extracted data_he_extractor.json"""
    console.print(f"[cyan]Building FAISS index for version={version}...[/cyan]")
    try:
        index_path, metadata_path = _build_index(version=version, embedding_model=embedding_model)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    console.print("[green]✨ Index build complete![/green]")
    console.print(f"  Index:    {index_path}")
    console.print(f"  Metadata: {metadata_path}")


# ======================================================================
# evaluate
# ======================================================================
def _normalize_label(label: str) -> str:
    return label.strip().upper().replace(" ", "_")


@app.command("evaluate")
def evaluate(
    test_dir: Path = typer.Argument(..., help="Labelled test folder (one subfolder per class), e.g. store/test"),
    version: str = typer.Option(..., "--version", "-v", help="Index version to classify against (from build-index)"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Where to save the Excel report"),
    workers: int = typer.Option(5, "--workers", "-w", help="Parallel classification workers"),
    enable_cache: bool = typer.Option(True, "--cache/--no-cache", help="Cache extraction results by file hash"),
):
    """
    Classify every document in test_dir, compare against ground-truth folder
    names, and produce a full Excel accuracy report.

    IMPORTANT — how bundle classes are scored:
    A bundle document (e.g. one whose folder is MOA_AOA_COI_BUNDLE) gets
    split by Layer 2 into child predictions (MOA / AOA / COI), which is
    exactly what should happen — a bundle test folder holds combined
    multi-document PDFs, and we WANT the final prediction to be the child
    label, not the bundle label. Since there's no page-level ground truth
    to check individual child accuracy against, "correct" for a bundle
    document means "Layer 1 correctly recognized this as belonging to the
    right bundle family and successfully entered Layer 2" — i.e. we check
    the document's ground-truth folder against Parent_Category, not against
    the child-level Predicted_Class. The child-level prediction still shows
    in the Results sheet exactly as before; only the correctness scoring
    changed.
    """
    settings = get_settings()
    output_dir = output_dir or settings.results_dir

    if not test_dir.exists():
        console.print(f"[red]Test directory not found: {test_dir}[/red]")
        raise typer.Exit(1)

    console.print("=" * 70)
    console.print("[bold]DOCUMENT CLASSIFICATION — EVALUATION RUN[/bold]")
    console.print("=" * 70)
    console.print(f"Test folder: {test_dir}")
    console.print(f"Index version: {version}")

    # Scan documents
    documents = []
    class_counts = {}
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        actual_class = _normalize_label(class_dir.name)
        for file_path in class_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in settings.allowed_file_types:
                continue
            documents.append({"file_path": str(file_path), "file_name": file_path.name, "actual_class": actual_class})
            class_counts[actual_class] = class_counts.get(actual_class, 0) + 1

    console.print(f"\nFound {len(documents)} documents across {len(class_counts)} classes:")
    for cls, count in sorted(class_counts.items()):
        console.print(f"  {cls}: {count}")

    if not documents:
        console.print("[yellow]No documents found. Nothing to evaluate.[/yellow]")
        raise typer.Exit(0)

    console.print("\n[cyan]Initializing classifier...[/cyan]")
    classifier = MLMCClassifier(version=version, enable_cache=enable_cache)

    def _classify_one(doc):
        try:
            start = time.time()
            results = classifier.classify(doc["file_path"])
            elapsed = round(time.time() - start, 2)
            rows = []
            for idx, r in enumerate(results):
                meta = r.metadata or {}
                ext_meta = meta.get("extraction_metadata", {}) or {}
                is_layer2 = meta.get("layer") == 2
                parent_category = meta.get("parent_category")

                if is_layer2:
                    # Bundle document: "correct" means Layer 1 routed it into
                    # the right bundle family. We can't check the individual
                    # child prediction against ground truth since there's no
                    # page-level label — only the document-level bundle folder.
                    is_correct = (doc["actual_class"] == parent_category)
                    bundle_recognized = is_correct
                else:
                    is_correct = (doc["actual_class"] == r.class_name)
                    bundle_recognized = "N/A"

                row = {
                    "Actual_Class": doc["actual_class"],
                    "Document_Name": doc["file_name"],
                    "Predicted_Class": r.class_name,
                    "Is_Correct": is_correct,
                    "Bundle_Recognized": bundle_recognized,
                    "Final_Confidence": round(r.confidence, 4) if r.confidence is not None else "N/A",
                    "RAG_Class": meta.get("rag_class") or "N/A",
                    "RAG_Confidence": round(meta.get("rag_confidence"), 4) if meta.get("rag_confidence") is not None else "N/A",
                    "Gemini_Class": meta.get("gemini_class") or "N/A",
                    "Gemini_Confidence": round(meta.get("gemini_confidence"), 4) if meta.get("gemini_confidence") is not None else "N/A",
                    "Layer": meta.get("layer", "N/A"),
                    "Method": meta.get("method", "N/A"),
                    "Parent_Category": parent_category or "N/A",
                    "Start_Page": r.start_page if r.start_page is not None else "N/A",
                    "End_Page": r.end_page if r.end_page is not None else "N/A",
                    "Input_Tokens": r.input_tokens,
                    "Output_Tokens": r.output_tokens,
                    "he_text_pages_free": ext_meta.get("free_text_pages", 0),
                    "he_ocr_pages_paid": ext_meta.get("ocr_pages", 0),
                    "he_total_pages": ext_meta.get("total_pages", 0),
                    "Elapsed_Sec": elapsed if idx == 0 else "N/A",
                    "Error": r.error or "",
                    "Document_Path": doc["file_path"],
                }
                row["RAG_Gemini_Agree"] = (
                    row["RAG_Class"] == row["Gemini_Class"]
                    if row["RAG_Class"] != "N/A" and row["Gemini_Class"] != "N/A"
                    else "N/A"
                )
                rows.append(row)
            return rows
        except Exception as e:
            logger.error(f"Classification failed for {doc['file_path']}: {e}", exc_info=True)
            return [{
                "Actual_Class": doc["actual_class"],
                "Document_Name": doc["file_name"],
                "Predicted_Class": "ERROR",
                "Is_Correct": False,
                "Bundle_Recognized": "N/A",
                "Final_Confidence": "N/A",
                "RAG_Class": "N/A", "RAG_Confidence": "N/A",
                "Gemini_Class": "N/A", "Gemini_Confidence": "N/A",
                "RAG_Gemini_Agree": "N/A",
                "Layer": "N/A", "Method": "N/A", "Parent_Category": "N/A",
                "Start_Page": "N/A", "End_Page": "N/A",
                "Input_Tokens": 0, "Output_Tokens": 0,
                "he_text_pages_free": 0, "he_ocr_pages_paid": 0, "he_total_pages": 0,
                "Elapsed_Sec": "N/A",
                "Error": str(e),
                "Document_Path": doc["file_path"],
            }]

    all_rows = []
    console.print(f"\n[cyan]Classifying {len(documents)} documents ({workers} workers)...[/cyan]")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_classify_one, d) for d in documents]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Classifying"):
            all_rows.extend(future.result())

    run_config_summary = {
        "Test Folder": str(test_dir),
        "Index Version": version,
        "Embedding Model": settings.classifier.get("embedding_model"),
        "Gemini Model": settings.gemini_model,
        "Layer1 RAG Threshold": settings.classifier.get("layer1_rag_confidence_threshold"),
        "Force LLM Classes": ", ".join(settings.classifier.get("force_llm_classes") or []) or "(none)",
        "Workers": workers,
        "Cache Enabled": enable_cache,
    }

    console.print("\n[cyan]Building Excel report...[/cyan]")
    excel_file = build_report(
        rows=all_rows,
        output_dir=str(output_dir),
        run_config_summary=run_config_summary,
        gemini_usage_summary=usage_tracker.summary(),
        dataset_folder=str(test_dir),
    )

    # Console summary
    total = len(all_rows)
    correct = sum(1 for r in all_rows if r["Is_Correct"])
    accuracy = round(correct / total * 100, 1) if total else 0.0

    console.print("\n" + "=" * 70)
    console.print("[bold]SUMMARY[/bold]")
    console.print("=" * 70)
    console.print(f"Total Documents: {total}")
    console.print(f"Correct Predictions: {correct}")
    console.print(f"Overall Accuracy: {accuracy}%")
    console.print(f"Gemini calls used: {usage_tracker.total_calls}")
    console.print(f"\nExcel Report: {excel_file}")

    # -------------------------------------------------------------
    # Document-level RAG vs Gemini dependency (terminal only).
    # A bundle document (e.g. a 44-page MOA/AOA/COI file) counts as ONE
    # document here, not one row per child section — unlike the Results
    # sheet in Excel, which intentionally keeps bundle children split out.
    # The exact same numbers are also written to the Excel report's
    # "RAG vs Gemini Dependency" sheet for reference; nothing else in the
    # Excel report is changed.
    # -------------------------------------------------------------
    dep = _compute_document_level_dependency(all_rows)
    console.print("\n" + "=" * 70)
    console.print("[bold]RAG vs GEMINI DEPENDENCY (per physical document, bundles counted once)[/bold]")
    console.print("=" * 70)
    console.print(f"Total Documents: {dep['total_documents']}")
    console.print(f"Resolved by RAG alone: {dep['rag_only_documents']}  ({dep['rag_only_pct']}%)")
    console.print(f"Needed Gemini (incl. bundle documents): {dep['gemini_dependent_documents']}  ({dep['gemini_dependent_pct']}%)")

    if classifier.cache:
        classifier.cache.print_stats()

    # Per-class accuracy table
    table = Table(title="Per-Class Accuracy")
    table.add_column("Class")
    table.add_column("Correct/Total")
    table.add_column("Accuracy %")
    class_stats = {}
    for r in all_rows:
        cls = r["Actual_Class"]
        class_stats.setdefault(cls, [0, 0])
        class_stats[cls][1] += 1
        if r["Is_Correct"]:
            class_stats[cls][0] += 1
    for cls, (correct_n, total_n) in sorted(class_stats.items()):
        pct = round(correct_n / total_n * 100, 1) if total_n else 0.0
        table.add_row(cls, f"{correct_n}/{total_n}", f"{pct}%")
    console.print(table)


# ======================================================================
# classify (single unlabeled document — production-style inference)
# ======================================================================
@app.command("classify")
def classify_single(
    file_path: Path = typer.Argument(..., help="Path to a single document to classify"),
    version: str = typer.Option(..., "--version", "-v", help="Index version to classify against"),
):
    """Classify ONE document with no known ground truth — just prints the prediction."""
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)

    classifier = MLMCClassifier(version=version, enable_cache=True)
    results = classifier.classify(str(file_path))

    table = Table(title=f"Prediction: {file_path.name}")
    table.add_column("Class")
    table.add_column("Confidence")
    table.add_column("Pages")
    table.add_column("Method")
    for r in results:
        pages = f"{r.start_page}-{r.end_page}" if r.start_page else "N/A"
        table.add_row(r.class_name, f"{r.confidence:.3f}", pages, r.metadata.get("method", "N/A"))
    console.print(table)


if __name__ == "__main__":
    app()