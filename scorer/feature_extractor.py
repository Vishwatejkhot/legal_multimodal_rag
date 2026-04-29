"""
Extracts numerical features from a query result for XGBoost evidence scoring.
"""
from __future__ import annotations
import numpy as np

FEATURE_NAMES = [
    "num_chunks",
    "avg_confidence",
    "max_confidence",
    "min_confidence",
    "num_conflicts",
    "num_sources",
    "num_image_chunks",
    "low_conf_ratio",
    "has_legislation_source",
    "has_case_law_source",
]

def extract_features(result: dict) -> np.ndarray:
    chunks = result.get("chunks", [])
    conflicts = result.get("conflicts", [])
    sources = result.get("sources", [])

    confidences = [c.get("confidence", 0) for c in chunks] or [0]

    has_legislation = int(any("Legislation" in s or "ukpga" in s or "Act" in s for s in sources))
    has_case_law = int(any("BAILII" in s or "Case Law" in s for s in sources))
    num_image_chunks = sum(1 for c in chunks if c.get("from_image", False))
    low_conf = sum(1 for c in chunks if c.get("confidence", 0) < 40)
    low_conf_ratio = low_conf / max(len(chunks), 1)

    features = [
        len(chunks),
        float(np.mean(confidences)),
        float(np.max(confidences)),
        float(np.min(confidences)),
        len(conflicts),
        len(sources),
        num_image_chunks,
        low_conf_ratio,
        has_legislation,
        has_case_law,
    ]
    return np.array(features, dtype="float32").reshape(1, -1)
