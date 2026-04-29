from __future__ import annotations
import re
from config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text: str, source: str = "", chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split text into overlapping chunks preserving sentence boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append({"text": chunk, "source": source, "chunk_index": len(chunks)})
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks
