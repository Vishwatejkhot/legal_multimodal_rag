"""
Routes uploaded files to the correct handler:
  - Images (JPG, PNG, WEBP) -> GPT-4o Vision
  - PDFs -> PyMuPDF (with per-page image fallback for scanned PDFs)
  - DOCX -> python-docx
Returns a flat list of text chunks.
"""
from __future__ import annotations
import io
from pathlib import Path
import fitz  # PyMuPDF
import docx
from image_pipeline.gpt4o_vision import describe_image
from image_pipeline.image_to_text import vision_output_to_chunks
from ingestion.chunker import chunk_text
from reliability.logger import log

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}
PDF_EXTENSION = ".pdf"
DOCX_EXTENSION = ".docx"

def handle_file(file_path: str | Path, question: str = "") -> list[dict]:
    path = Path(file_path)
    ext = path.suffix.lower()
    log.info("handling_file", file=str(path), ext=ext)

    if ext in IMAGE_EXTENSIONS:
        return _handle_image(path, question)
    elif ext == PDF_EXTENSION:
        return _handle_pdf(path, question)
    elif ext == DOCX_EXTENSION:
        return _handle_docx(path)
    else:
        log.warning("unsupported_file_type", ext=ext)
        return []

def _handle_image(path: Path, question: str) -> list[dict]:
    vision_text = describe_image(str(path), question)
    return vision_output_to_chunks(vision_text, source=str(path))

def _handle_pdf(path: Path, question: str) -> list[dict]:
    doc = fitz.open(str(path))
    all_chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if len(text) > 100:
            # Native text PDF page
            all_chunks.extend(chunk_text(text, source=f"{path.name} p{page_num+1}"))
        else:
            # Scanned page - render and send to Vision
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            tmp = path.parent / f"_tmp_page_{page_num}.png"
            tmp.write_bytes(img_bytes)
            vision_text = describe_image(str(tmp), question)
            all_chunks.extend(vision_output_to_chunks(vision_text, source=f"{path.name} p{page_num+1} [scanned]"))
            tmp.unlink(missing_ok=True)
    doc.close()
    return all_chunks

def _handle_docx(path: Path) -> list[dict]:
    document = docx.Document(str(path))
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    return chunk_text(text, source=str(path))

def handle_uploaded_bytes(file_bytes: bytes, filename: str, question: str = "") -> list[dict]:
    """Handle file uploaded as bytes (from Streamlit)."""
    import tempfile, os
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return handle_file(tmp_path, question)
    finally:
        os.unlink(tmp_path)
