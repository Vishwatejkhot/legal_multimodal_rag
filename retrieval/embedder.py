from __future__ import annotations
from sentence_transformers import SentenceTransformer
from reliability.logger import log
from config import EMBED_MODEL

_model: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("loading_embed_model", model=EMBED_MODEL)
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def embed_text(text: str) -> list[float]:
    vec = _get_model().encode(text[:2048], normalize_embeddings=True)
    return vec.tolist()

def embed_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    model = _get_model()
    all_vecs: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = [t[:2048] for t in texts[i : i + batch_size]]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.extend(vecs.tolist())
    return all_vecs
