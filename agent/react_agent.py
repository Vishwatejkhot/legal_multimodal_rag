from __future__ import annotations
from collections.abc import Iterator
from retrieval.rrf_fusion import rrf_search
from retrieval.reranker import rerank
from agent.confidence_scorer import score_chunks, is_low_confidence
from agent.conflict_detector import detect_conflicts
from agent.synthesiser import synthesise, synthesise_stream
from reliability.logger import log
from config import TOP_K


def _retrieve_and_score(query: str, extra_chunks: list[dict] | None) -> tuple[list[dict], list[str]]:
    retrieved = rrf_search(query, top_k=TOP_K * 2)

    if extra_chunks:
        retrieved = extra_chunks + retrieved

    # Cross-encoder re-rank before confidence scoring
    retrieved = rerank(query, retrieved, top_k=TOP_K)
    scored = score_chunks(retrieved, query)
    conflicts = detect_conflicts(scored[:8], query)

    if conflicts:
        log.info("re_searching_due_to_conflicts")
        refined = query + " " + " ".join(c.replace("CONFLICT:", "").strip() for c in conflicts[:2])
        extra_hits = score_chunks(rerank(refined, rrf_search(refined, top_k=10), top_k=5), query)
        merged = {c["text"][:200]: c for c in scored + extra_hits}
        scored = sorted(merged.values(), key=lambda c: c["confidence"], reverse=True)

    return scored, conflicts


def _build_result(scored: list[dict], conflicts: list[str], answer: str) -> dict:
    low_conf = [c for c in scored if is_low_confidence(c)]
    return {
        "answer": answer,
        "chunks": scored[:8],
        "conflicts": conflicts,
        "low_confidence_count": len(low_conf),
        "sources": list({c.get("source", "unknown") for c in scored[:8]}),
    }


def run(query: str, extra_chunks: list[dict] | None = None) -> dict:
    log.info("agent_start", query=query[:80])
    scored, conflicts = _retrieve_and_score(query, extra_chunks)
    answer = synthesise(query, scored[:6], conflicts)
    result = _build_result(scored, conflicts, answer)
    log.info("agent_complete", sources=len(result["sources"]), conflicts=len(conflicts))
    return result


def run_stream(query: str, extra_chunks: list[dict] | None = None) -> tuple[dict, Iterator[str]]:
    """Returns (partial_result_without_answer, answer_stream).
    Caller should collect the stream and store the final answer via result['answer'] = collected."""
    log.info("agent_stream_start", query=query[:80])
    scored, conflicts = _retrieve_and_score(query, extra_chunks)
    stream = synthesise_stream(query, scored[:6], conflicts)
    partial = _build_result(scored, conflicts, answer="")
    return partial, stream
