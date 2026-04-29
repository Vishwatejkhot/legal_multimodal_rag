import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.legislation_scraper import fetch_all_legislation
from ingestion.bailii_scraper import scrape_bailii
from ingestion.sentencing_scraper import scrape_sentencing
from ingestion.hmcts_scraper import scrape_hmcts
from reliability.logger import log


def main():
    log.info("=== JusticeAI Data Fetcher ===")

    print("\n[1/4] Fetching UK Legislation (legislation.gov.uk)...")
    leg_chunks = fetch_all_legislation()
    print(f"  Done: {len(leg_chunks)} chunks")

    print("\n[2/4] Fetching BAILII Case Law...")
    case_chunks = scrape_bailii()
    print(f"  Done: {len(case_chunks)} chunks")

    print("\n[3/4] Fetching Sentencing Council Guidelines (PDFs)...")
    sentencing_chunks = scrape_sentencing()
    print(f"  Done: {len(sentencing_chunks)} chunks")

    print("\n[4/4] Fetching HMCTS Court Procedure Rules...")
    hmcts_chunks = scrape_hmcts()
    print(f"  Done: {len(hmcts_chunks)} chunks")

    total = len(leg_chunks) + len(case_chunks) + len(sentencing_chunks) + len(hmcts_chunks)
    print(f"\nData fetch complete: {total} total chunks")
    print("Next:  python scripts/fetch_training.py")
    print("Then:  python scripts/build_index.py")


if __name__ == "__main__":
    main()
