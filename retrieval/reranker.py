from __future__ import annotations
from reliability.logger import log

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None
_model_failed = False


def _get_model():
    global _model, _model_failed
    if _model_failed:
        return None
    if _model is not None:
        return _model
    try:
        from sentence_transformers import CrossEncoder
        log.info("loading_cross_encoder", model=_MODEL_NAME)
        _model = CrossEncoder(_MODEL_NAME)
        log.info("cross_encoder_loaded")
    except Exception as e:
        log.warning("cross_encoder_unavailable", error=str(e))
        _model_failed = True
    return _model


def rerank(query: str, chunks: list[dict], top_k: int = 10) -> list[dict]:
    if not chunks:
        return chunks
    model = _get_model()
    if model is None:
        # Fall back to returning chunks as-is if model unavailable
        return chunks[:top_k]
    try:
        pairs = [(query, c["text"][:512]) for c in chunks]
        scores = model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)
        reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
        log.info("reranked", input=len(chunks), output=len(reranked))
        return reranked
    except Exception as e:
        log.warning("rerank_failed", error=str(e))
        return chunks[:top_k]
