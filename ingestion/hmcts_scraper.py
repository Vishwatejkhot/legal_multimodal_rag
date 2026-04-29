from __future__ import annotations
import re
import time
import requests
from bs4 import BeautifulSoup
from config import HMCTS_DIR
from ingestion.chunker import chunk_text
from reliability.logger import log

HMCTS_DIR.mkdir(parents=True, exist_ok=True)

_HEADERS = {"User-Agent": "JusticeAI-Research/1.0 (academic; contact khotvishwatej@gmail.com)"}

_CPR_PARTS = [
    ("CPR_Part_01_Overriding_Objective", "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part01"),
    ("CPR_Part_03_Case_Management", "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part03"),
    ("CPR_Part_06_Service", "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part06"),
    ("CPR_Part_07_Starting_Proceedings", "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part07"),
    ("CPR_Part_08_Alternative_Procedure", "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part08"),
    ("CPR_Part_12_Default_Judgment", "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part12"),
    ("CPR_Part_24_Summary_Judgment", "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part24"),
    ("CPR_Part_55_Possession_Claims", "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part55"),
    ("CPR_Part_65_Proceedings_Relating_To_Anti_Social_Behaviour", "https://www.justice.gov.uk/courts/procedure-rules/civil/rules/part65"),
]


def _fetch_html_page(name: str, url: str) -> list[dict]:
    out_file = HMCTS_DIR / f"{name}.txt"

    # Already on disk — read directly, no dedup check
    if out_file.exists():
        text = out_file.read_text(encoding="utf-8")
        return chunk_text(text, source=f"HMCTS CPR: {name}")

    try:
        log.info("fetching_hmcts_page", name=name)
        r = requests.get(url, headers=_HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        main = soup.find("main") or soup.find("div", class_=re.compile("content|main|article"))
        text = re.sub(r"\s+", " ", (main or soup).get_text(separator=" ")).strip()
        if not text:
            return []
        out_file.write_text(text, encoding="utf-8")
        time.sleep(1)
        chunks = chunk_text(text, source=f"HMCTS CPR: {name}")
        log.info("hmcts_page_done", name=name, chunks=len(chunks))
        return chunks
    except Exception as e:
        log.warning("hmcts_page_failed", name=name, error=str(e))
        return []


def scrape_hmcts() -> list[dict]:
    all_chunks = []
    for name, url in _CPR_PARTS:
        all_chunks.extend(_fetch_html_page(name, url))
    return all_chunks
