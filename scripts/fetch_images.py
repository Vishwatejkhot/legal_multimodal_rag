"""
Downloads sample test images from free datasets:
  - FUNSD (handwritten forms) from HuggingFace  -> data/images/funsd/
  - DocVQA (document QA) from HuggingFace       -> data/images/docvqa/

Usage:
    python scripts/fetch_images.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from PIL import Image
from reliability.logger import log

IMAGES_DIR = Path(__file__).parent.parent / "data" / "images"


def fetch_funsd(n: int = 20):
    out = IMAGES_DIR / "funsd"
    out.mkdir(parents=True, exist_ok=True)
    if len(list(out.glob("*.png"))) >= n:
        log.info("funsd_already_exists")
        return
    log.info("fetching_funsd", n=n)
    ds = load_dataset("nielsr/funsd", split="test", streaming=True, trust_remote_code=True)
    saved = 0
    for i, sample in enumerate(ds):
        if saved >= n:
            break
        img: Image.Image = sample["image"]
        img.save(out / f"form_{i:04d}.png")
        saved += 1
    log.info("funsd_done", saved=saved)


def fetch_docvqa(n: int = 20):
    out = IMAGES_DIR / "docvqa"
    out.mkdir(parents=True, exist_ok=True)
    if len(list(out.glob("*.png"))) >= n:
        log.info("docvqa_already_exists")
        return
    log.info("fetching_docvqa", n=n)
    ds = load_dataset("nielsr/docvqa_1200_examples", split="test", streaming=True, trust_remote_code=True)
    saved = 0
    for i, sample in enumerate(ds):
        if saved >= n:
            break
        img: Image.Image = sample["image"]
        img.save(out / f"docvqa_{i:04d}.png")
        saved += 1
    log.info("docvqa_done", saved=saved)


def main():
    log.info("=== JusticeAI Image Fetcher ===")

    print("\n[1/2] Downloading FUNSD handwritten forms (20 samples)...")
    try:
        fetch_funsd(n=20)
    except Exception as e:
        print(f"  FUNSD failed: {e}")

    print("\n[2/2] Downloading DocVQA document images (20 samples)...")
    try:
        fetch_docvqa(n=20)
    except Exception as e:
        print(f"  DocVQA failed: {e}")

    total = sum(len(list(p.glob("*.png"))) for p in [IMAGES_DIR / "funsd", IMAGES_DIR / "docvqa"] if p.exists())
    print(f"\nDone. {total} images saved to data/images/")
    print("Upload them via the Streamlit UI to test the image pipeline.")


if __name__ == "__main__":
    main()
