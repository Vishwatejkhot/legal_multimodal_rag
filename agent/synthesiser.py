from __future__ import annotations
from collections.abc import Iterator
import anthropic
from config import ANTHROPIC_API_KEY, REASON_MODEL
from reliability.rate_limiter import throttle, anthropic_retry
from reliability.logger import log

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_SYSTEM = (
    "You are JusticeAI, a legal research assistant for UK caseworkers. "
    "Answer questions about UK law using ONLY the provided evidence passages. "
    "Always cite sources by name in brackets. "
    "If evidence is weak or contradictory, say so explicitly. "
    "Never make up law. If you don't know, say 'Insufficient evidence to determine this.'"
)

_USER_TEMPLATE = """QUERY: {query}

EVIDENCE PASSAGES (ranked by relevance):
{passages}

{conflict_block}
Provide a structured answer:
1. DIRECT ANSWER to the query
2. RELEVANT LAW (cite specific Acts, Sections, case names)
3. HOW THE EVIDENCE SUPPORTS OR WEAKENS THE CLAIM
4. UNCERTAINTY FLAGS (what is missing or contradictory)""".strip()


def _build_prompt(query: str, chunks: list[dict], conflicts: list[str]) -> str:
    passages = "\n\n".join(
        f"[Confidence: {c.get('confidence', 0):.0f}%] [{c.get('source', 'unknown')}]\n{c['text'][:500]}"
        for c in chunks[:6]
    )
    conflict_block = ""
    if conflicts:
        conflict_block = "⚠️ CONFLICTS DETECTED:\n" + "\n".join(conflicts) + "\n\n"
    return _USER_TEMPLATE.format(query=query, passages=passages, conflict_block=conflict_block)


@anthropic_retry
def synthesise(query: str, chunks: list[dict], conflicts: list[str]) -> str:
    throttle()
    log.info("synthesising_answer", query=query[:60])
    response = _client.messages.create(
        model=REASON_MODEL,
        max_tokens=1500,
        system=_SYSTEM,
        messages=[{"role": "user", "content": _build_prompt(query, chunks, conflicts)}],
    )
    answer = response.content[0].text if response.content else ""
    log.info("answer_synthesised", chars=len(answer))
    return answer


def synthesise_stream(query: str, chunks: list[dict], conflicts: list[str]) -> Iterator[str]:
    throttle()
    log.info("synthesising_stream", query=query[:60])
    with _client.messages.stream(
        model=REASON_MODEL,
        max_tokens=1500,
        system=_SYSTEM,
        messages=[{"role": "user", "content": _build_prompt(query, chunks, conflicts)}],
    ) as stream:
        for token in stream.text_stream:
            yield token
