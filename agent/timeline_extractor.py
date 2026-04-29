from __future__ import annotations
import json
import re
from datetime import datetime
import anthropic
from config import ANTHROPIC_API_KEY, REASON_MODEL
from reliability.rate_limiter import throttle, anthropic_retry
from reliability.logger import log

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

_PROMPT = """Extract every date mentioned in the legal evidence below.
Return a JSON array where each item has:
  "date"       - YYYY-MM-DD format (use YYYY-01-01 if only year is known)
  "event"      - one sentence describing what happened
  "legal_note" - legal significance if any, else empty string

Evidence:
{text}

Return ONLY valid JSON array, no other text."""

_DEADLINE_RULES = [
    ("section 21", 60, "Section 21 requires at least 2 months notice (Housing Act 1988 s.21)"),
    ("deposit", 30, "Deposit must be protected within 30 days of receipt (Deregulation Act 2015)"),
    ("section 8", 14, "Section 8 requires at least 14 days notice for rent arrears grounds"),
    ("notice to quit", 28, "Periodic tenancy requires at least 28 days notice to quit"),
    ("prescribed information", 30, "Prescribed information must be served within 30 days of deposit receipt"),
]


@anthropic_retry
def extract_timeline(chunks: list[dict]) -> list[dict]:
    if not chunks:
        return []

    combined = "\n\n".join(c["text"][:600] for c in chunks[:8])
    throttle()
    log.info("extracting_timeline", chars=len(combined))

    response = _client.messages.create(
        model=REASON_MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": _PROMPT.format(text=combined)}],
    )
    raw = response.content[0].text if response.content else "[]"

    # Extract JSON array from response
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not match:
        return []
    try:
        events = json.loads(match.group())
    except Exception:
        return []

    valid = []
    for e in events:
        if not isinstance(e, dict) or not e.get("date"):
            continue
        try:
            dt = datetime.strptime(str(e["date"])[:10], "%Y-%m-%d")
            valid.append({
                "date": dt,
                "date_str": dt.strftime("%d %b %Y"),
                "event": e.get("event", ""),
                "legal_note": e.get("legal_note", ""),
            })
        except ValueError:
            pass

    valid.sort(key=lambda x: x["date"])
    log.info("timeline_extracted", events=len(valid))
    return valid


def flag_deadline_issues(events: list[dict], full_text: str) -> list[dict]:
    issues = []
    text_lower = full_text.lower()
    for keyword, days_required, rule in _DEADLINE_RULES:
        if keyword not in text_lower:
            continue
        keyword_events = [e for e in events if keyword in e["event"].lower()]
        if len(keyword_events) < 2:
            continue
        start = keyword_events[0]["date"]
        end   = keyword_events[-1]["date"]
        gap   = (end - start).days
        if gap < days_required:
            issues.append({
                "rule": rule,
                "start": keyword_events[0]["date_str"],
                "end":   keyword_events[-1]["date_str"],
                "gap_days": gap,
                "required_days": days_required,
                "breach": True,
            })
    return issues
