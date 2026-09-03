# Document Classification Pipeline (Zero-Budget)

A hierarchical document classifier: **RAG (FAISS vector search) + Gemini LLM fallback**,
with a two-layer structure that can resolve "bundle" documents (a single PDF
that stitches together multiple sub-documents, e.g. Certificate of
Incorporation + MOA + AOA) into their individual child classes.

Runs entirely on your own laptop, **$0 cost**:

| Original component        | This project uses instead      | Cost |
|----------------------------|--------------------------------|------|
| HE Extractor (PyMuPDF + paid Cloud OCR) | PyMuPDF + **Tesseract OCR** (local, free) | $0 |
| Vertex AI (service account, billing)     | **Gemini free-tier API key** (AI Studio) | $0 |
| Embeddings (all-MiniLM-L6-v2)            | Same — `sentence-transformers`, local     | $0 |
| FAISS vector index                       | Same — `faiss-cpu`, local                 | $0 |
| Excel reporting                          | Same — `pandas` + `openpyxl`              | $0 |

---

## 1. Prerequisites

- **Python 3.10+**
- **Tesseract OCR** installed on your machine (this is a separate binary,
  not just a pip package):
  - Windows: [download installer](https://github.com/UB-Mannheim/tesseract/wiki), default install path is usually `C:\Program Files\Tesseract-OCR\tesseract.exe`
  - Mac: `brew install tesseract`
  - Linux: `sudo apt install tesseract-ocr`
- A **free Gemini API key** from [Google AI Studio](https://aistudio.google.com/apikey)
  (this is separate from Vertex AI — no billing account, no credit card needed)

## 2. Setup

```bash
# 1. Extract the zip and cd into it
cd document-classifier

# 2. Create a virtual environment
python -m venv venv

# Activate it
#   Windows:
venv\Scripts\activate
#   Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your environment variables
cp .env.example .env       # (Windows: copy .env.example .env)
# Now open .env and paste in your Gemini API key + your Tesseract path
```

## 3. Add your data

Your training and test datasets go here, one subfolder per class:

```
store/
├── train/
│   ├── PAN_CARD/
│   │   ├── pan_001.pdf
│   │   └── ...
│   ├── AADHAR_CARD/
│   ├── BANK_STATEMENT/
│   ├── MOA_AOA_COI_BUNDLE/        <- bundle classes: same structure,
│   │   ├── bundle_doc_01.pdf         just multi-page files that combine
│   │   └── ...                       COI + MOA + AOA in one PDF
│   └── ...
└── test/
    ├── PAN_CARD/
    ├── AADHAR_CARD/
    ├── MOA_AOA_COI_BUNDLE/
    └── ...   (same class folders, different documents — for evaluation)
```

**Important:** the folder name IS the ground-truth label used for accuracy
scoring, and it must exactly match a `label` in your `document_classes.yaml`
and an entry in `layer1_standalone` / `layer1_categories` in
`config/classifier_config.yaml`.

## 4. Add your document classes

Copy your real `document_classes.yaml` (the one with your full class
descriptions) into:

```
config/document_classes.yaml
```

A template with the exact expected schema is at
`config/document_classes.example.yaml` — read the comments there.
If you don't add a real file, the pipeline will run using the small example
template (5 classes) so you can smoke-test the setup end-to-end first.

## 5. Tune thresholds (optional)

Open `config/classifier_config.yaml` to adjust:
- `layer1_rag_confidence_threshold` — how confident RAG must be before Gemini isn't consulted
- `force_llm_classes` — classes that always get a Gemini opinion regardless of RAG confidence (set to `[]` to disable and see if accuracy holds up without it — good experiment for comparing approaches)
- `layer1_categories` / `layer1_standalone` — your bundle and standalone classes
- `max_pages` — cap on how many pages get extracted per document (0 = no cap)

## 6. Run the pipeline

```bash
# Step 1: Extract text from your training documents
python run.py extract-data store/train --version cl_v1

# Step 2: Build the FAISS vector index from the extracted training data
python run.py build-index --version cl_v1

# Step 3: Evaluate against your labelled test set — produces an Excel report
python run.py evaluate store/test --version cl_v1

# (Optional) Classify a single new, unlabeled document — just prints a prediction
python run.py classify path/to/some_document.pdf --version cl_v1
```

The Excel report lands in `classification_results/` with 5 sheets:
- **Results** — every prediction, row by row
- **Error Analysis** — only the wrong ones, for quick debugging
- **Per-Class Accuracy** — sorted worst-to-best, so you know where to focus
- **RAG vs Gemini** — agreement rate + every case they disagreed
- **Configuration** — full run settings + Gemini free-tier call count used

## 7. Re-running / iterating

- Re-running `evaluate` reuses the extraction cache (`extraction_cache/`) by
  file content hash — changed files re-extract automatically, unchanged ones
  don't cost you another OCR pass or Gemini call.
- To try a new threshold or class list, just edit `config/classifier_config.yaml`
  and re-run `evaluate` — no need to re-extract or re-build the index unless
  your *training* data changed.
- To add more training documents, drop them into `store/train/<CLASS>/`,
  re-run `extract-data` with the same `--version` (or a new one like `cl_v2`),
  then `build-index` again with that version.

## 8. Project structure

```
document-classifier/
├── run.py                          # entry point
├── requirements.txt
├── .env.example                    # copy to .env, fill in your Gemini key
├── config/
│   ├── classifier_config.yaml      # thresholds, bundle structure (cleaned from activities.yaml)
│   └── document_classes.example.yaml
├── src/
│   ├── settings.py                 # loads .env + yaml configs
│   ├── extraction/
│   │   ├── text_extractors.py      # PyMuPDF + Tesseract OCR + Excel extraction
│   │   └── data_extraction.py      # walks train/test folders, parallel extraction
│   ├── embeddings/
│   │   └── index_builder.py        # builds FAISS index + metadata.json
│   ├── classification/
│   │   ├── rag_classifier.py       # FAISS similarity search + confidence scoring
│   │   ├── gemini_client.py        # Gemini free-tier API wrapper, retry/backoff
│   │   ├── mlmc_classifier.py      # Layer 1 (RAG->Gemini) + Layer 2 (bundle split)
│   │   └── cache_manager.py        # MD5-based extraction cache
│   ├── reporting/
│   │   └── excel_report.py         # multi-sheet Excel report generator
│   └── cli.py                      # Typer CLI (extract-data, build-index, evaluate, classify)
├── store/
│   ├── train/                      # <- put your training data here
│   └── test/                       # <- put your test data here
├── extraction_cache/                # auto-generated, gitignored
└── classification_results/          # Excel reports land here
```

## How the classification logic works

**Layer 1** (every document):
1. Extract text (PyMuPDF direct extraction; Tesseract OCR fallback per page
   for scanned content; direct cell reading for Excel).
2. Search the FAISS index of training embeddings for the `top_k` nearest
   neighbours → RAG prediction + confidence.
3. If RAG confidence ≥ `layer1_rag_confidence_threshold` **and** the class
   isn't in `force_llm_classes` → accept the RAG result.
4. Otherwise, ask Gemini to classify from the full candidate list
   (standalone classes + bundle parent names) → Gemini's answer becomes the
   final Layer 1 result.

**Layer 2** (only if Layer 1 predicted a *bundle* class):
1. Split the PDF into individual pages.
2. Classify each page independently against only that bundle's configured
   children (Gemini only).
3. Merge consecutive pages predicted as the same child class into a single
   result with a page range (e.g. pages 1–3 = COI, pages 4–9 = MOA, pages
   10–15 = AOA).

## Notes on the Gemini free tier

The free tier has real rate limits (requests per minute / per day). This
project automatically retries with exponential backoff on rate-limit errors,
and the Excel report's Configuration sheet shows exactly how many Gemini
calls were used per run — useful for knowing when you're getting close to
the daily quota. If you hit persistent quota errors, either wait for the
quota window to reset, or raise `layer1_rag_confidence_threshold` so more
documents get accepted from RAG alone without needing Gemini.








## How to Run This Project — Full Step-by-Step Guide

### Part 1 — One-Time Setup (do this once, when you first clone/download the project)

**Step 1: Open the project folder in VS Code**
- File → Open Folder → select `document-classifier`

**Step 2: Open a terminal inside VS Code**
- Terminal → New Terminal

**Step 3: Create a virtual environment**
```bash
python -m venv venv
```
*Creates: a new `venv/` folder containing an isolated Python environment for this project.*

**Step 4: Activate the virtual environment**
```bash
venv\Scripts\activate       
```
You'll know it worked when you see `(venv)` at the start of your terminal prompt. **You must do this every time you open a new terminal to work on this project** — it doesn't stay activated permanently.

**Step 5: Install all dependencies**
```bash
pip install -r requirements.txt
```
📁 *Installs all required packages into `venv/Lib/site-packages/` (nothing outside the project folder is touched).*

**Step 6: Create your environment file**
```bash
copy .env.example .env        # Windows
cp .env.example .env          # Mac/Linux
```
📁 *Creates: `.env` — open this file and fill in your-
Gemini API key, 
Tesseract path, 
and OCR language list.*

**Step 7: Add your document classes file**
- Place your real `document_classes.yaml` at:


**Step 8: Add your training and test data**
store/train/<CLASS_NAME>/ → your training documents, one folder per class
store/test/<CLASS_NAME>/ → your test documents, same folder structure


Setup is now complete — steps 1-8 are one-time only.

---

### Part 2 — Running the Pipeline (do these 3 steps in order, every time you have new/changed training data)

**Step 1: Extract text from training documents**
```bash
python run.py extract-data store/train --version cl_v1
```
📁 *Creates:*
- `store/cl_v1/data_he_extractor.json` — extracted text for every training document
- `store/cl_v1/extraction_warnings.txt` — a list of any documents that extracted with little/no text (worth checking manually)

**Step 2: Build the search index**
```bash
python run.py build-index --version cl_v1
```
📁 *Creates:*
- `store/cl_v1/index.faiss` — the FAISS vector search index
- `store/cl_v1/metadata.json` — maps each vector back to its file path and class label

**Step 3: Evaluate against your test set**
```bash
python run.py evaluate store/test --version cl_v1 --workers 3    (#chanhe the workers according to your requirement)
```
📁 *Creates:*
- `classification_results/classification_report_<timestamp>.xlsx` — the full accuracy report (Results, Error Analysis, Per-Class Accuracy, RAG vs Gemini, RAG vs Gemini Dependency, Configuration sheets)
- Also silently updates `extraction_cache/` with cached extraction results (see Part 3 below)

At the end, the terminal prints:
- Overall accuracy and per-class accuracy table
- Total Gemini calls used
- RAG vs Gemini dependency (per physical document, bundles counted once)

# NOT REQUIRED FOR NORMALLY **Optional — classify a single new document with no known answer:**
```bash
python run.py classify path\to\some_document.pdf --version cl_v1
```
This just prints a prediction table to the terminal — no files created, no ground truth needed.

---

### Part 3 — Re-running / Iterating (use this whenever you change something and want to test again)

**What triggers what — this matters, since not every change needs every step redone:**

| If you changed... | You need to re-run... |
|---|---|
| Training documents (`store/train/`) | `extract-data` → `build-index` → `evaluate` |
| `config/classifier_config.yaml` (thresholds, class lists) | `evaluate` only |
| `config/document_classes.yaml` (descriptions) | `evaluate` only |
| Extraction logic itself (if a code file in `src/extraction/` was edited) | Clear cache first (see below), then `extract-data` → `build-index` → `evaluate` |
| Test documents (`store/test/`) | `evaluate` only |

**Clearing the extraction cache (only needed when extraction logic changes, not for normal re-runs):**
```bash
# Windows PowerShell
Remove-Item extraction_cache\* -Force

```
The cache is keyed by file content hash, so it automatically detects if a *document* changed — you only need to manually clear it when the *extraction code itself* changes (e.g. new OCR preprocessing), since old cached results won't reflect the new logic.

**Every evaluation run creates a NEW timestamped Excel file** — old reports are never overwritten, so you can keep a history of runs to compare against each other.

---

### Quick Reference — All Commands in Order (fresh setup)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# (now edit .env, add document_classes.yaml, add train/test data)
python run.py extract-data store/train --version cl_v1
python run.py build-index --version cl_v1
python run.py evaluate store/test --version cl_v1 --workers 3
```