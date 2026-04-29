from __future__ import annotations
import json
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path
from config import XGBOOST_MODEL_PATH, TRAINING_DIR
from reliability.logger import log

_TRAINING_FILES = [
    Path(TRAINING_DIR) / "synthetic" / "synthetic_cases.json",
    Path(TRAINING_DIR) / "xgboost_cases" / "lexglue_ecthr.json",
]

def _generate_synthetic_data(n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X, y = [], []
    for _ in range(n):
        label = rng.integers(0, 3)
        if label == 2:
            row = [rng.integers(5, 10), rng.uniform(70, 95), rng.uniform(80, 100),
                   rng.uniform(50, 80), 0, rng.integers(3, 7), rng.integers(0, 3),
                   rng.uniform(0, 0.1), 1, 1]
        elif label == 1:
            row = [rng.integers(3, 7), rng.uniform(40, 70), rng.uniform(60, 85),
                   rng.uniform(20, 50), rng.integers(0, 2), rng.integers(1, 4),
                   rng.integers(0, 2), rng.uniform(0.1, 0.3), rng.integers(0, 2),
                   rng.integers(0, 2)]
        else:
            row = [rng.integers(0, 4), rng.uniform(5, 40), rng.uniform(20, 60),
                   rng.uniform(0, 20), rng.integers(1, 4), rng.integers(0, 2),
                   0, rng.uniform(0.4, 1.0), 0, 0]
        X.append(row)
        y.append(label)
    return np.array(X, dtype="float32"), np.array(y, dtype="int")

def train(use_synthetic: bool = True) -> None:
    log.info("xgboost_training_start")

    available = [f for f in _TRAINING_FILES if f.exists()]
    if use_synthetic or not available:
        X, y = _generate_synthetic_data(n=2000)
        log.info("using_synthetic_data", samples=len(y))
    else:
        records = []
        for f in available:
            records.extend(json.loads(f.read_text()))
        if records and "features" in records[0]:
            X = np.array([d["features"] for d in records], dtype="float32")
            y = np.array([d["label"] for d in records], dtype="int")
        else:
            X, y = _generate_synthetic_data(n=2000)
        log.info("using_saved_data", files=[str(f) for f in available], samples=len(y))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_val)
    report = classification_report(y_val, preds, target_names=["Low", "Medium", "High"])
    log.info("xgboost_eval", report=report)
    print(report)

    Path(XGBOOST_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(XGBOOST_MODEL_PATH)
    log.info("xgboost_saved", path=XGBOOST_MODEL_PATH)
