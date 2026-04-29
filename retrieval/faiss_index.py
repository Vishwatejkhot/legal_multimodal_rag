from __future__ import annotations
import json
import numpy as np
import faiss
from pathlib import Path
from config import FAISS_INDEX_PATH, FAISS_META_PATH, TOP_K
from retrieval.embedder import embed_text, embed_batch
from reliability.logger import log

def build_faiss_index(chunks: list[dict]) -> None:
    log.info("building_faiss", total_chunks=len(chunks))
    texts = [c["text"] for c in chunks]
    embeddings = embed_batch(texts)
    matrix = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "w") as f:
        json.dump(chunks, f)
    log.info("faiss_built", vectors=matrix.shape[0])

def search_faiss(query: str, top_k: int = TOP_K) -> list[dict]:
    if not Path(FAISS_INDEX_PATH).exists():
        log.warning("faiss_index_missing")
        return []
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_META_PATH) as f:
        meta = json.load(f)
    qvec = np.array([embed_text(query)], dtype="float32")
    faiss.normalize_L2(qvec)
    scores, indices = index.search(qvec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = dict(meta[idx])
        chunk["faiss_score"] = float(score)
        results.append(chunk)
    return results
