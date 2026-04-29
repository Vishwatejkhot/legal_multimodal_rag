from __future__ import annotations
import anthropic
from config import ANTHROPIC_API_KEY, REASON_MODEL
from reliability.rate_limiter import throttle, anthropic_retry
from reliability.logger import log

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_PROMPT = """You are a legal evidence analyst.
Below are {n} evidence passages related to the query: "{query}"

Passages:
{passages}

Identify any DIRECT CONTRADICTIONS between passages.
For each conflict found, output exactly:
CONFLICT: [Passage A claim] vs [Passage B claim]

If no conflicts are found, output: NO CONFLICTS DETECTED"""


@anthropic_retry
def detect_conflicts(chunks: list[dict], query: str) -> list[str]:
    if len(chunks) < 2:
        return []

    passages = "\n\n".join(
        f"[{i+1}] (Source: {c.get('source','unknown')}) {c['text'][:400]}"
        for i, c in enumerate(chunks[:8])
    )
    prompt = _PROMPT.format(n=min(len(chunks), 8), query=query, passages=passages)

    throttle()
    log.info("conflict_detection", chunks=len(chunks))

    response = _client.messages.create(
        model=REASON_MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text if response.content else ""
    if "NO CONFLICTS DETECTED" in text:
        return []
    conflicts = [line.strip() for line in text.splitlines() if line.strip().startswith("CONFLICT:")]
    log.info("conflicts_found", count=len(conflicts))
    return conflicts
