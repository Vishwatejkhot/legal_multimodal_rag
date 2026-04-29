"""
Reads all fetched legal text and builds FAISS, BM25, and LanceDB indexes.
Run after fetch_data.py.

Usage:
    python scripts/build_index.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LEGISLATION_DIR, CASE_LAW_DIR, SENTENCING_DIR, HMCTS_DIR
from ingestion.chunker import chunk_text
from retrieval.faiss_index import build_faiss_index
from retrieval.bm25_index import build_bm25_index
from retrieval.lancedb_store import upsert_chunks
from scorer.xgboost_trainer import train as train_xgboost
from reliability.logger import log


def load_text_files(folder: Path, source_prefix: str) -> list[dict]:
    chunks = []
    for txt in folder.glob("**/*.txt"):
        text = txt.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue
        chunks.extend(chunk_text(text, source=f"{source_prefix}: {txt.stem}"))
    return chunks


def main():
    log.info("=== JusticeAI Index Builder ===")

    all_chunks = []

    log.info("loading_legislation")
    all_chunks.extend(load_text_files(LEGISLATION_DIR, "UK Legislation"))

    log.info("loading_case_law")
    all_chunks.extend(load_text_files(CASE_LAW_DIR, "UK Case Law"))

    log.info("loading_sentencing")
    all_chunks.extend(load_text_files(SENTENCING_DIR, "Sentencing Council"))

    log.info("loading_hmcts")
    all_chunks.extend(load_text_files(HMCTS_DIR, "HMCTS CPR"))

    log.info("total_chunks_to_index", count=len(all_chunks))

    if not all_chunks:
        print("No data found. Run:  python scripts/fetch_data.py  first.")
        return

    log.info("building_faiss_index")
    build_faiss_index(all_chunks)

    log.info("building_bm25_index")
    build_bm25_index(all_chunks)

    log.info("building_lancedb_index")
    upsert_chunks(all_chunks)

    log.info("training_xgboost")
    train_xgboost()

    print(f"\nIndex build complete: {len(all_chunks)} chunks indexed.")
    print("Run:  streamlit run app.py")


if __name__ == "__main__":
    main()
