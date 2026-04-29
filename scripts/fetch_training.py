import sys
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from reliability.logger import log

TRAINING_DIR = Path(__file__).parent.parent / "data" / "training"
XGBOOST_DIR = TRAINING_DIR / "xgboost_cases"
SYNTHETIC_DIR = TRAINING_DIR / "synthetic"
XGBOOST_DIR.mkdir(parents=True, exist_ok=True)
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}


def fetch_lexglue(n: int = 300):
    log.info("fetching_lexglue", n=n)
    out = XGBOOST_DIR / "lexglue_ecthr.json"
    if out.exists():
        log.info("lexglue_already_exists")
        return

    ds = load_dataset(
        "coastalcph/lex_glue",
        "ecthr_a",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    records = []
    for i, sample in enumerate(ds):
        if len(records) >= n:
            break
        text = " ".join(sample.get("text", []))[:2000]
        labels = sample.get("labels", [])
        if len(labels) == 0:
            strength = 0
        elif len(labels) <= 2:
            strength = 1
        else:
            strength = 2
        records.append({"text": text, "label": strength, "label_name": LABEL_MAP[strength]})

    out.write_text(json.dumps(records, indent=2))
    log.info("lexglue_saved", records=len(records), path=str(out))
    print(f"  Saved {len(records)} LexGLUE records -> {out}")


_TEMPLATES = {
    "High": [
        "Landlord entered the property without notice on three occasions and changed the locks while the tenant was present. Witness statements from two neighbours confirm this. The tenant has a valid tenancy agreement dated {year} and rent receipts showing full payment.",
        "Section 21 notice served with less than 2 months notice. The tenancy is an assured shorthold tenancy under the Housing Act 1988. Notice was not served using Form 6A. Rent was paid in full throughout.",
        "Council confirmed the property has category 1 hazards under the Housing Health and Safety Rating System. Landlord was notified in writing {months} months ago and took no action. Environmental Health report attached.",
        "Written demand for illegal premium of £{amount} documented in email exchange. This violates Section 19 of the Landlord and Tenant Act 1985. Bank transfer records confirm payment was made.",
    ],
    "Medium": [
        "Tenant claims landlord refused to carry out repairs but has only one written complaint on record. Landlord disputes the severity of the disrepair. No independent inspection has been carried out.",
        "Notice to quit served but date of service is unclear. Witness states it was hand-delivered but no postal record exists. Tenancy agreement does not specify service method.",
        "Rent arrears of £{amount} are disputed. Tenant claims partial payments were made in cash but has no receipts. Landlord's statement shows {months} months of arrears.",
        "Tenant alleges harassment but incidents are undocumented. One text message from landlord referenced in statement. Tone is aggressive but no explicit threats.",
    ],
    "Low": [
        "Verbal complaint with no documentary evidence. Tenant cannot recall exact dates of alleged incidents.",
        "Single incident described in a statement written {months} months after the event. No witnesses and no contemporaneous notes.",
        "Claim based on tenant's understanding of the lease which they have not retained a copy of. Landlord denies the alleged verbal agreement.",
        "Allegation is contradicted by a signed document in the landlord's possession. No explanation for the discrepancy has been provided.",
    ],
}

def _generate_synthetic(n: int = 500):
    log.info("generating_synthetic", n=n)
    out = SYNTHETIC_DIR / "synthetic_cases.json"
    if out.exists():
        log.info("synthetic_already_exists")
        return

    rng = random.Random(42)
    records = []
    label_names = list(_TEMPLATES.keys())
    per_label = n // len(label_names)

    for label_name in label_names:
        templates = _TEMPLATES[label_name]
        label_idx = list(LABEL_MAP.values()).index(label_name)
        for _ in range(per_label):
            tpl = rng.choice(templates)
            text = tpl.format(
                year=rng.randint(2015, 2024),
                months=rng.randint(1, 18),
                amount=rng.randint(500, 5000),
            )
            records.append({"text": text, "label": label_idx, "label_name": label_name})

    rng.shuffle(records)
    out.write_text(json.dumps(records, indent=2))
    log.info("synthetic_saved", records=len(records), path=str(out))
    print(f"  Saved {len(records)} synthetic cases -> {out}")


def main():
    log.info("=== JusticeAI Training Data Fetcher ===")

    print("\n[1/2] Downloading LexGLUE (European Court decisions, 300 samples)...")
    try:
        fetch_lexglue(n=300)
    except Exception as e:
        print(f"  LexGLUE failed: {e}")

    print("\n[2/2] Generating synthetic UK housing case scenarios (500 samples)...")
    _generate_synthetic(n=500)

    total = sum(
        len(json.loads(f.read_text()))
        for f in list(XGBOOST_DIR.glob("*.json")) + list(SYNTHETIC_DIR.glob("*.json"))
        if f.exists()
    )
    print(f"\nDone. {total} training records in data/training/")
    print("Run:  python scripts/build_index.py  to train XGBoost on this data.")


if __name__ == "__main__":
    main()
