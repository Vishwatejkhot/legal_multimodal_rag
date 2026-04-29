from __future__ import annotations
import re
import time
import requests
from lxml import etree
from config import LEGISLATION_DIR, LEGISLATION_TARGETS
from ingestion.chunker import chunk_text
from ingestion.deduplicator import is_duplicate
from reliability.logger import log

LEGISLATION_DIR.mkdir(parents=True, exist_ok=True)

_BASE = "https://www.legislation.gov.uk"
_NAMESPACES = {
    "leg": "http://www.legislation.gov.uk/namespaces/legislation",
    "ukl": "http://www.legislation.gov.uk/namespaces/legislation",
}

def _fetch_xml(url: str) -> bytes | None:
    try:
        r = requests.get(url, timeout=30, headers={"Accept": "application/xml"})
        r.raise_for_status()
        return r.content
    except Exception as e:
        log.warning("fetch_failed", url=url, error=str(e))
        return None

def _extract_text_from_xml(xml_bytes: bytes) -> str:
    try:
        root = etree.fromstring(xml_bytes)
        texts = root.itertext()
        raw = " ".join(t for t in texts if t.strip())
        return re.sub(r"\s+", " ", raw).strip()
    except Exception as e:
        log.warning("xml_parse_failed", error=str(e))
        return ""

def fetch_legislation(target: dict) -> list[dict]:
    typ = target["type"]
    year = target["year"]
    number = target["number"]
    title = target["title"]

    out_file = LEGISLATION_DIR / f"{typ}_{year}_{number}.txt"
    url = f"{_BASE}/{typ}/{year}/{number}/data.xml"

    log.info("fetching_legislation", title=title, url=url)

    if out_file.exists():
        log.info("using_cached_legislation", title=title)
        text = out_file.read_text(encoding="utf-8")
    else:
        xml = _fetch_xml(url)
        if not xml:
            return []
        text = _extract_text_from_xml(xml)
        if not text:
            return []
        if is_duplicate(text):
            return []
        out_file.write_text(text, encoding="utf-8")
        time.sleep(1)

    chunks = chunk_text(text, source=f"UK Legislation: {title}")
    log.info("legislation_fetched", title=title, chunks=len(chunks))
    return chunks

def fetch_all_legislation() -> list[dict]:
    all_chunks = []
    for target in LEGISLATION_TARGETS:
        all_chunks.extend(fetch_legislation(target))
    return all_chunks
