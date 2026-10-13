from __future__ import annotations

import io
import os
from typing import Dict, List

import fitz
from PIL import Image


def extract_pdf_pages(pdf_path: str, zoom: float = 2.0) -> List[Dict]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages: List[Dict] = []
    doc = fitz.open(pdf_path)
    try:
        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            text = page.get_text("text").strip()
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            pages.append(
                {
                    "page_num": page_idx + 1,
                    "text": text,
                    "image": img,
                }
            )
    finally:
        doc.close()
    return pages
