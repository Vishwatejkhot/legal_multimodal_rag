"""
Reciprocal Rank Fusion: combines FAISS and BM25 ranked lists.
Score: sum of 1 / (k + rank) across all rankers, where k=60.
"""
from __future__ import annotations
from retrieval.faiss_index import search_faiss
from retrieval.bm25_index import search_bm25
from config import TOP_K

_K = 60

def _text_key(chunk: dict) -> str:
    return chunk["text"][:200]

def rrf_search(query: str, top_k: int = TOP_K) -> list[dict]:
    faiss_results = search_faiss(query, top_k=top_k * 2)
    bm25_results = search_bm25(query, top_k=top_k * 2)

    scores: dict[str, float] = {}
    chunks_by_key: dict[str, dict] = {}

    for rank, chunk in enumerate(faiss_results):
        key = _text_key(chunk)
        scores[key] = scores.get(key, 0.0) + 1.0 / (_K + rank + 1)
        chunks_by_key[key] = chunk

    for rank, chunk in enumerate(bm25_results):
        key = _text_key(chunk)
        scores[key] = scores.get(key, 0.0) + 1.0 / (_K + rank + 1)
        if key not in chunks_by_key:
            chunks_by_key[key] = chunk

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]
    results = []
    for key in sorted_keys:
        chunk = dict(chunks_by_key[key])
        chunk["rrf_score"] = scores[key]
        results.append(chunk)
    return results
