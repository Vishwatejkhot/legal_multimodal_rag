"""
Converts GPT-4o Vision output into structured text chunks ready for embedding.
"""
from __future__ import annotations
from ingestion.chunker import chunk_text

def vision_output_to_chunks(vision_text: str, source: str = "uploaded_image") -> list[dict]:
    """Wrap vision output as text chunks with metadata."""
    if not vision_text.strip():
        return []
    chunks = chunk_text(vision_text, source=source)
    for chunk in chunks:
        chunk["from_image"] = True
    return chunks
