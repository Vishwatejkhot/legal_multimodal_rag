from __future__ import annotations
import re
import time
import requests
from bs4 import BeautifulSoup
from config import SENTENCING_DIR
from ingestion.chunker import chunk_text
from reliability.logger import log

SENTENCING_DIR.mkdir(parents=True, exist_ok=True)

_BASE = "https://www.sentencingcouncil.org.uk"
_CROWN_LISTING = f"{_BASE}/guidelines/crown-court/"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120"}

_KNOWN_OFFENCES = [
    ("Assault_ABH", f"{_BASE}/offences/crown-court/item/assault-occasioning-actual-bodily-harm-racially-religiously-aggravated-abh/"),
    ("Harassment", f"{_BASE}/offences/crown-court/item/harassment-putting-people-in-fear-of-violence/"),
    ("Burglary_Domestic", f"{_BASE}/offences/crown-court/item/domestic-burglary/"),
    ("Theft_General", f"{_BASE}/offences/crown-court/item/theft-general/"),
    ("Drug_Supply", f"{_BASE}/offences/crown-court/item/drug-offences-supply-of-a-controlled-drug/"),
    ("Sexual_Assault", f"{_BASE}/offences/crown-court/item/sexual-assault/"),
    ("Fraud_General", f"{_BASE}/offences/crown-court/item/fraud-general/"),
    ("Robbery", f"{_BASE}/offences/crown-court/item/robbery-street-and-less-sophisticated-commercial/"),
    ("Stalking", f"{_BASE}/offences/crown-court/item/stalking-involving-fear-of-violence-or-serious-alarm-or-distress/"),
    ("Breach_Restraining_Order", f"{_BASE}/offences/magistrates-court/item/breach-of-a-protective-order-restraining-and-non-molestation-orders/"),
]


def _discover_offence_links(max_links: int = 15) -> list[tuple[str, str]]:
    try:
        r = requests.get(_CROWN_LISTING, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        seen, links = set(), []
        for a in soup.find_all("a", href=True):
            href = str(a["href"])
            if "/offences/" in href and "/item/" in href and href not in seen:
                seen.add(href)
                full = href if href.startswith("http") else _BASE + href
                name = re.sub(r"[^\w]", "_", a.get_text(strip=True))[:60] or re.sub(r"[^\w]", "_", href)[:60]
                links.append((name, full))
                if len(links) >= max_links:
                    break
        return links if links else _KNOWN_OFFENCES
    except Exception as e:
        log.warning("sentencing_listing_failed", error=str(e))
        return _KNOWN_OFFENCES


def _scrape_offence_page(name: str, url: str) -> list[dict]:
    out_file = SENTENCING_DIR / f"{name}.txt"
    if out_file.exists():
        text = out_file.read_text(encoding="utf-8")
        return chunk_text(text, source=f"Sentencing Council: {name}")

    try:
        log.info("fetching_sentencing_offence", name=name)
        r = requests.get(url, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        main = soup.find("main") or soup.find("div", id=re.compile("content|main"))
        text = re.sub(r"\s+", " ", (main or soup).get_text(separator=" ")).strip()

        if len(text) < 300:
            log.warning("sentencing_page_too_short", name=name, chars=len(text))
            return []

        out_file.write_text(text, encoding="utf-8")
        time.sleep(1)
        chunks = chunk_text(text, source=f"Sentencing Council: {name}")
        log.info("sentencing_offence_done", name=name, chunks=len(chunks))
        return chunks
    except Exception as e:
        log.warning("sentencing_offence_failed", name=name, error=str(e))
        return []


def scrape_sentencing() -> list[dict]:
    links = _discover_offence_links(max_links=15)
    log.info("sentencing_offences_found", count=len(links))
    all_chunks = []
    for name, url in links:
        all_chunks.extend(_scrape_offence_page(name, url))
    return all_chunks
