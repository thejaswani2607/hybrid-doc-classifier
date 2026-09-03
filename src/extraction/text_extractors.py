"""
HE (Hybrid Extractor) — zero-budget text extraction.

Mirrors the original HE Extractor's behaviour and output shape:
  - PDFs: try direct text extraction per page first (PyMuPDF, free).
          ALSO extracts any embedded raster images at their native
          resolution and OCRs those separately — this catches cases where
          a page has a small amount of real text (e.g. a scanner header)
          alongside a scanned ID card image; relying on direct text alone
          would otherwise skip OCR and miss the card content entirely.
          Whichever produces more content (direct text, embedded-image OCR,
          or full-page-render OCR) is used per page.
  - Images (png/jpg/jpeg): OCR'd directly with Tesseract.

  IMAGE PREPROCESSING: every image handed to Tesseract first goes through:
    1. Automatic rotation correction via Tesseract OSD.
    2. Enhancement — grayscale, auto-contrast, sharpening, upscaling.

  NOTE: Excel (.xls/.xlsx) extraction was removed.

Returns a (text, metadata) tuple where metadata mirrors the original schema:
    {
        "total_pages": int,
        "pages_processed": int,
        "free_text_pages": int,   # pages extracted without OCR
        "ocr_pages": int,         # pages that needed OCR
        "method": "he_extractor",
    }
"""
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageOps, ImageFilter
import io

from src.settings import get_settings

logger = logging.getLogger(__name__)

# Minimum characters on a page before we trust the direct text extraction
# and skip OCR for that page.
MIN_CHARS_FOR_FREE_EXTRACTION = 20

# Even if direct text clears the threshold above, if the page ALSO contains
# embedded raster images of at least this size, we still attempt image OCR
# and compare — a short scanner-header text shouldn't hide a real ID card
# image sitting on the same page.
MIN_EMBEDDED_IMAGE_PIXELS = 150 * 150

KNOWN_SCAN_WATERMARKS = [
    "scanned by camscanner",
    "scanned with camscanner",
    "cam scanner",
    "adobe scan",
    "scanned by",
]

MIN_DIMENSION_BEFORE_UPSCALE = 1200
UPSCALE_FACTOR = 2.0

FULL_PAGE_RENDER_DPI = 300


def _looks_like_scan_watermark_only(text: str) -> bool:
    stripped = text.strip().lower()
    if not stripped:
        return False
    if len(stripped) <= 60:
        for watermark in KNOWN_SCAN_WATERMARKS:
            if watermark in stripped:
                return True
    return False


def _configure_tesseract():
    settings = get_settings()
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


def _correct_rotation(img: Image.Image) -> Image.Image:
    try:
        osd = pytesseract.image_to_osd(img)
        match = re.search(r"Rotate:\s*(\d+)", osd)
        if match:
            rotation_needed = int(match.group(1))
            if rotation_needed != 0:
                img = img.rotate(-rotation_needed, expand=True)
    except Exception as e:
        logger.debug(f"Rotation detection skipped (OSD failed): {e}")
    return img


def _enhance_for_ocr(img: Image.Image) -> Image.Image:
    if img.mode != "L":
        img = img.convert("L")
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.SHARPEN)
    width, height = img.size
    if min(width, height) < MIN_DIMENSION_BEFORE_UPSCALE:
        new_size = (int(width * UPSCALE_FACTOR), int(height * UPSCALE_FACTOR))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    img = _correct_rotation(img)
    img = _enhance_for_ocr(img)
    return img


def _ocr_image(img: Image.Image, settings) -> str:
    img = _preprocess_for_ocr(img)
    # --psm 11 ("sparse text, no particular order") often reads ID-card-style
    # layouts (PAN/Aadhaar/Voter ID — scattered fields, not paragraphs) more
    # reliably than Tesseract's default paragraph-oriented mode.
    custom_config = "--psm 11"
    text = pytesseract.image_to_string(
        img, lang=settings.tesseract_languages or "eng", config=custom_config
    )
    # Sparse mode occasionally under-reads dense paragraph documents (e.g.
    # bank statements, deeds) — if it returns very little, retry with the
    # default paragraph mode as a fallback.
    if len(text.strip()) < 20:
        text = pytesseract.image_to_string(img, lang=settings.tesseract_languages or "eng")
    return text


def _extract_embedded_images_text(page, doc, settings) -> str:
    """
    Extracts every embedded raster image on a page AT ITS NATIVE RESOLUTION
    and OCRs each one separately. This is the key fix for scanned ID cards
    (Aadhaar/PAN) embedded inside a PDF page: a flattened whole-page render
    at a fixed DPI can downsample the actual card image below what OCR can
    read, especially if the card only occupies part of the page.
    """
    texts = []
    try:
        image_list = page.get_images(full=True)
    except Exception as e:
        logger.debug(f"Could not list embedded images: {e}")
        return ""

    for img_info in image_list:
        xref = img_info[0]
        try:
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_img = Image.open(io.BytesIO(image_bytes))

            if pil_img.width * pil_img.height < MIN_EMBEDDED_IMAGE_PIXELS:
                continue  # too small to be a meaningful scanned document/card

            if pil_img.mode not in ("RGB", "L"):
                pil_img = pil_img.convert("RGB")

            text = _ocr_image(pil_img, settings)
            if text.strip():
                texts.append(text.strip())
        except Exception as e:
            logger.debug(f"Embedded image OCR failed for xref {xref}: {e}")
            continue

    return "\n".join(texts)


def _render_full_page_and_ocr(page, settings, dpi: int = FULL_PAGE_RENDER_DPI) -> str:
    """Fallback: render the whole page as one image and OCR it."""
    try:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return _ocr_image(img, settings).strip()
    except Exception as e:
        logger.warning(f"Full-page OCR render failed: {e}")
        return ""


def _get_best_page_text(page, doc, settings, direct_text: str) -> Tuple[str, bool]:
    """
    Tries all available extraction paths for a page and returns whichever
    produced the most content, plus whether OCR was actually used.

    Order tried:
      1. Direct text (free, already have it)
      2. Embedded-image OCR at native resolution (best for scanned ID cards)
      3. Full-page-render OCR (fallback, catches anything not embedded as
         a discrete image, e.g. a page that's itself one big scanned image)
    """
    candidates = [("direct", direct_text)]

    embedded_text = _extract_embedded_images_text(page, doc, settings)
    if embedded_text:
        candidates.append(("embedded_ocr", embedded_text))

    # Only bother with a full-page render if the other two are weak —
    # this is the slowest option and usually redundant if embedded-image
    # OCR already found real content.
    best_so_far = max(candidates, key=lambda c: len(c[1]))
    if len(best_so_far[1]) < MIN_CHARS_FOR_FREE_EXTRACTION:
        full_page_text = _render_full_page_and_ocr(page, settings)
        if full_page_text:
            candidates.append(("full_page_ocr", full_page_text))

    best_method, best_text = max(candidates, key=lambda c: len(c[1]))
    used_ocr = best_method in ("embedded_ocr", "full_page_ocr")
    return best_text, used_ocr


def extract_pdf(file_path: str, max_pages: int = 0) -> Tuple[str, Dict[str, Any]]:
    _configure_tesseract()
    settings = get_settings()
    doc = fitz.open(file_path)
    total_pages = len(doc)

    if total_pages == 0:
        logger.warning(f"PDF has 0 pages (possibly corrupted or empty file): {file_path}")
        doc.close()
        return "", {
            "total_pages": 0, "pages_processed": 0,
            "free_text_pages": 0, "ocr_pages": 0,
            "method": "he_extractor_empty_or_corrupt",
        }

    pages_to_process = total_pages if max_pages <= 0 else min(max_pages, total_pages)

    texts = []
    free_text_pages = 0
    ocr_pages = 0

    for page_index in range(pages_to_process):
        page = doc[page_index]
        direct_text = page.get_text("text").strip()

        is_watermark_only = _looks_like_scan_watermark_only(direct_text)
        has_embedded_images = len(page.get_images(full=True)) > 0

        # Skip the extra work ONLY if we have solid direct text AND there
        # are no embedded images worth checking (i.e. genuinely a
        # text-native page, not a scanned card sitting next to some text).
        if (
            len(direct_text) >= MIN_CHARS_FOR_FREE_EXTRACTION
            and not is_watermark_only
            and not has_embedded_images
        ):
            texts.append(direct_text)
            free_text_pages += 1
            continue

        best_text, used_ocr = _get_best_page_text(page, doc, settings, direct_text)
        texts.append(best_text)
        if used_ocr:
            ocr_pages += 1
        else:
            free_text_pages += 1

    doc.close()

    full_text = "\n".join(t for t in texts if t)
    metadata = {
        "total_pages": total_pages,
        "pages_processed": pages_to_process,
        "free_text_pages": free_text_pages,
        "ocr_pages": ocr_pages,
        "method": "he_extractor",
    }
    return full_text, metadata


def extract_pdf_pages(file_path: str, max_pages: int = 0) -> list:
    """Same as extract_pdf but returns per-page text list (for Layer 2 bundle splitting)."""
    _configure_tesseract()
    settings = get_settings()
    doc = fitz.open(file_path)
    total_pages = len(doc)

    if total_pages == 0:
        doc.close()
        return []

    pages_to_process = total_pages if max_pages <= 0 else min(max_pages, total_pages)

    page_texts = []
    for page_index in range(pages_to_process):
        page = doc[page_index]
        direct_text = page.get_text("text").strip()

        is_watermark_only = _looks_like_scan_watermark_only(direct_text)
        has_embedded_images = len(page.get_images(full=True)) > 0

        if (
            len(direct_text) >= MIN_CHARS_FOR_FREE_EXTRACTION
            and not is_watermark_only
            and not has_embedded_images
        ):
            page_texts.append(direct_text)
            continue

        best_text, _ = _get_best_page_text(page, doc, settings, direct_text)
        page_texts.append(best_text)

    doc.close()
    return page_texts


def extract_image(file_path: str) -> Tuple[str, Dict[str, Any]]:
    _configure_tesseract()
    settings = get_settings()
    try:
        img = Image.open(file_path)
        text = _ocr_image(img, settings).strip()
        metadata = {
            "total_pages": 1, "pages_processed": 1,
            "free_text_pages": 0, "ocr_pages": 1,
            "method": "he_extractor",
        }
        return text, metadata
    except Exception as e:
        logger.error(f"Image OCR failed for {file_path}: {e}")
        return "", {
            "total_pages": 1, "pages_processed": 0,
            "free_text_pages": 0, "ocr_pages": 0,
            "method": "he_extractor_error",
        }


def extract_text(file_path: str, max_pages: int = 0) -> Tuple[str, Dict[str, Any]]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_pdf(file_path, max_pages=max_pages)
    elif ext in (".png", ".jpg", ".jpeg"):
        return extract_image(file_path)
    else:
        logger.warning(f"Unsupported file type for extraction (Excel support removed): {file_path}")
        return "", {
            "total_pages": 0, "pages_processed": 0,
            "free_text_pages": 0, "ocr_pages": 0,
            "method": "unsupported",
        }