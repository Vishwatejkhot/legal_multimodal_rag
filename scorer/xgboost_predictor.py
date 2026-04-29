from __future__ import annotations
import numpy as np
import xgboost as xgb
from pathlib import Path
from config import XGBOOST_MODEL_PATH
from scorer.feature_extractor import extract_features
from scorer.xgboost_trainer import train
from reliability.logger import log

_LABELS = ["Low", "Medium", "High"]
_model: xgb.XGBClassifier | None = None

def _load_model() -> xgb.XGBClassifier:
    global _model
    if _model is not None:
        return _model
    if not Path(XGBOOST_MODEL_PATH).exists():
        log.info("xgboost_model_not_found_training")
        train()
    _model = xgb.XGBClassifier()
    _model.load_model(XGBOOST_MODEL_PATH)
    return _model

def predict_evidence_strength(result: dict) -> dict:
    model = _load_model()
    features = extract_features(result)
    proba = model.predict_proba(features)[0]
    label_idx = int(np.argmax(proba))
    label = _LABELS[label_idx]
    confidence_pct = round(float(proba[label_idx]) * 100, 1)
    log.info("evidence_strength", label=label, confidence=confidence_pct)
    return {
        "label": label,
        "confidence": confidence_pct,
        "probabilities": {_LABELS[i]: round(float(p) * 100, 1) for i, p in enumerate(proba)},
    }
