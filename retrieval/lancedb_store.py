from __future__ import annotations
import lancedb
import pyarrow as pa
from config import LANCEDB_PATH, TOP_K
from retrieval.embedder import embed_text, embed_batch
from reliability.logger import log

_EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension

_SCHEMA = pa.schema([
    pa.field("text", pa.string()),
    pa.field("source", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("from_image", pa.bool_()),
    pa.field("vector", pa.list_(pa.float32(), _EMBED_DIM)),
])

def _get_table():
    db = lancedb.connect(LANCEDB_PATH)
    if "legal_chunks" in db.table_names():
        return db.open_table("legal_chunks")
    return db.create_table("legal_chunks", schema=_SCHEMA)

def upsert_chunks(chunks: list[dict]) -> None:
    if not chunks:
        return
    log.info("lancedb_upserting", count=len(chunks))
    texts = [c["text"] for c in chunks]
    embeddings = embed_batch(texts)
    rows = []
    for chunk, emb in zip(chunks, embeddings):
        rows.append({
            "text": chunk["text"],
            "source": chunk.get("source", ""),
            "chunk_index": int(chunk.get("chunk_index", 0)),
            "from_image": bool(chunk.get("from_image", False)),
            "vector": emb,
        })
    table = _get_table()
    table.add(rows)
    log.info("lancedb_upserted", count=len(rows))

def search_lancedb(query: str, top_k: int = TOP_K) -> list[dict]:
    try:
        table = _get_table()
        qvec = embed_text(query)
        results = table.search(qvec).limit(top_k).to_list()
        out = []
        for r in results:
            out.append({
                "text": r["text"],
                "source": r["source"],
                "chunk_index": r["chunk_index"],
                "from_image": r.get("from_image", False),
                "lancedb_score": float(r.get("_distance", 0)),
            })
        return out
    except Exception as e:
        log.warning("lancedb_search_failed", error=str(e))
        return []
