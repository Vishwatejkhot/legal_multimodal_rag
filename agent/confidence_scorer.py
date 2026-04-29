from __future__ import annotations
from config import CONFIDENCE_THRESHOLD

def score_chunk(chunk: dict, query: str) -> float:
    rrf = chunk.get("rrf_score", 0.0)
    faiss = chunk.get("faiss_score", 0.0)
    bm25_raw = chunk.get("bm25_score", 0.0)

    semantic_score = min(faiss, 1.0) if faiss else min(rrf * 100, 1.0)
    bm25_normalised = min(bm25_raw / 20.0, 1.0)

    query_words = set(query.lower().split())
    chunk_words = set(chunk["text"].lower().split())
    overlap = len(query_words & chunk_words) / max(len(query_words), 1)

    raw = 0.5 * semantic_score + 0.3 * bm25_normalised + 0.2 * overlap
    return round(raw * 100, 1)

def score_chunks(chunks: list[dict], query: str) -> list[dict]:
    for chunk in chunks:
        chunk["confidence"] = score_chunk(chunk, query)
    return sorted(chunks, key=lambda c: c["confidence"], reverse=True)

def is_low_confidence(chunk: dict) -> bool:
    return chunk.get("confidence", 0) < CONFIDENCE_THRESHOLD
