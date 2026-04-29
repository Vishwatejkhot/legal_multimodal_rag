from __future__ import annotations
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from config import BM25_INDEX_PATH, TOP_K
from reliability.logger import log

def build_bm25_index(chunks: list[dict]) -> None:
    log.info("building_bm25", total_chunks=len(chunks))
    corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(corpus)
    Path(BM25_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    log.info("bm25_built", docs=len(corpus))

def search_bm25(query: str, top_k: int = TOP_K) -> list[dict]:
    if not Path(BM25_INDEX_PATH).exists():
        log.warning("bm25_index_missing")
        return []
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    bm25: BM25Okapi = data["bm25"]
    chunks: list[dict] = data["chunks"]
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        chunk = dict(chunks[idx])
        chunk["bm25_score"] = float(scores[idx])
        results.append(chunk)
    return results
