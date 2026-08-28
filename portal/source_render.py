"""
Render annual-report pages with the exact source region highlighted.

Turns a Tier 3 citation (chunk -> page + normalised bbox) into an image of that
page with the sourced block outlined, so a portal user can see precisely where a
number came from rather than being dropped on a dense page and left to hunt.

Boxes come from portal/boxes/{file_name}__{type}.json, produced either by the
live pipeline (new extractions) or portal/backfill_boxes.py (existing ones).
"""

import os
import json
import hashlib

import fitz  # PyMuPDF

PORTAL_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(PORTAL_DIR)
BOXES_DIR = os.path.join(PORTAL_DIR, "boxes")
RENDER_CACHE = os.path.join(PORTAL_DIR, ".render_cache")

# Deployed context ships only portal/pdfs/ (the subset built by
# build_pdf_subset.py, covering the companies with live tokens). Locally the
# full corpus is still available, so fall back to it for anything not yet
# copied into the subset (e.g. a company minted after the last subset build).
_SUBSET_PDF_DIR = os.path.join(PORTAL_DIR, "pdfs")
_FULL_PDF_DIR = os.path.join(BASE, "data", "metadata", "pdfs")


def _resolve_pdf_path(file_name: str) -> str | None:
    subset_path = os.path.join(_SUBSET_PDF_DIR, f"{file_name}.pdf")
    if os.path.exists(subset_path):
        return subset_path
    full_path = os.path.join(_FULL_PDF_DIR, f"{file_name}.pdf")
    if os.path.exists(full_path):
        return full_path
    return None

HIGHLIGHT_RGB = (0.77, 0.35, 0.07)   # matches the portal accent
PAD = 0.012                          # padding around the box, in page fractions


def load_boxes(file_name: str, extraction_type: str = "ltip") -> dict:
    """{chunk_index(int): {'page':int, 'box':{left,top,right,bottom}}}"""
    path = os.path.join(BOXES_DIR, f"{file_name}__{extraction_type}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    out = {}
    for k, v in raw.items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out


def has_box(file_name: str, chunk_id, extraction_type: str = "ltip") -> bool:
    if chunk_id is None:
        return False
    try:
        chunk_id = int(chunk_id)
    except (TypeError, ValueError):
        return False
    entry = load_boxes(file_name, extraction_type).get(chunk_id)
    return bool(entry and entry.get("box"))


def render_page(file_name: str, page_no: int, box: dict = None,
                zoom: float = 2.0, crop: bool = False) -> str | None:
    """Render a page to PNG (cached on disk). Returns the file path or None.

    box   — normalised 0-1 {left,top,right,bottom}; outlined when supplied
    crop  — if True, tightly crop around the box (with padding) for a close-up
    """
    pdf_path = _resolve_pdf_path(file_name)
    if not pdf_path:
        return None

    key = hashlib.md5(
        f"{file_name}|{page_no}|{json.dumps(box, sort_keys=True)}|{zoom}|{crop}".encode()
    ).hexdigest()[:16]
    os.makedirs(RENDER_CACHE, exist_ok=True)
    out_path = os.path.join(RENDER_CACHE, f"{key}.png")
    if os.path.exists(out_path):
        return out_path

    try:
        doc = fitz.open(pdf_path)
        if page_no - 1 < 0 or page_no - 1 >= len(doc):
            doc.close()
            return None
        page = doc.load_page(page_no - 1)
        pw, ph = page.rect.width, page.rect.height

        clip = None
        if box:
            rect = fitz.Rect(box["left"] * pw, box["top"] * ph,
                             box["right"] * pw, box["bottom"] * ph)
            # Outline the sourced region
            page.draw_rect(rect, color=HIGHLIGHT_RGB, width=2.0)
            if crop:
                clip = fitz.Rect(
                    max(0, (box["left"] - PAD) * pw),
                    max(0, (box["top"] - PAD) * ph),
                    min(pw, (box["right"] + PAD) * pw),
                    min(ph, (box["bottom"] + PAD) * ph),
                )

        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
        pix.save(out_path)
        doc.close()
        return out_path
    except Exception:
        return None


def render_citation(file_name: str, chunk_id, extraction_type: str = "ltip",
                    crop: bool = False, zoom: float = 2.0):
    """Convenience: resolve a chunk citation straight to a rendered PNG path.

    Returns (path, page_no) or (None, None).
    """
    if chunk_id is None:
        return None, None
    try:
        chunk_id = int(chunk_id)
    except (TypeError, ValueError):
        return None, None
    entry = load_boxes(file_name, extraction_type).get(chunk_id)
    if not entry:
        return None, None
    page_no, box = entry.get("page"), entry.get("box")
    if not page_no:
        return None, None
    return render_page(file_name, page_no, box, zoom=zoom, crop=crop), page_no
