from __future__ import annotations
from pathlib import Path
import fitz  # PyMuPDF
from ingestion.chunker import chunk_text
from ingestion.deduplicator import is_duplicate
from reliability.logger import log

def parse_pdf(pdf_path: str | Path) -> list[dict]:
    path = Path(pdf_path)
    log.info("parsing_pdf", file=str(path))
    doc = fitz.open(str(path))
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    if is_duplicate(full_text):
        log.info("skipping_duplicate", file=str(path))
        return []

    chunks = chunk_text(full_text, source=str(path))
    log.info("pdf_parsed", file=str(path), chunks=len(chunks))
    return chunks

def parse_pdf_folder(folder: str | Path) -> list[dict]:
    folder = Path(folder)
    all_chunks = []
    for pdf in folder.glob("**/*.pdf"):
        all_chunks.extend(parse_pdf(pdf))
    return all_chunks
