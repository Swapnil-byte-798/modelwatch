"""The monitored model.

Trained once on the reference cell, frozen, pinned by SHA-256, never retrained.
The classifier is deliberately boring: the monitor is the subject of this
project, not the model. Choosing HistGradientBoosting over a neural network is
itself a design decision -- it trains in seconds, so every experiment can be
re-run from scratch.

All features are fed as numeric codes rather than declared categoricals. ACS
OCCP and POBP have several hundred levels, above HistGradientBoosting's 255-bin
categorical limit. This matches the standard folktables baseline and is recorded
in the model card as a known limitation.
"""
from __future__ import annotations

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from . import config as C
from .data import load_window

MODEL_PATH = C.ARTIFACT_DIR / "model.joblib"
CALIB_PATH = C.ARTIFACT_DIR / "calibrator.joblib"
META_PATH = C.ARTIFACT_DIR / "model_meta.json"
SPLIT_PATH = C.ARTIFACT_DIR / "reference_splits.parquet"


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def split_reference(df: pd.DataFrame, seed: int = C.SEED):
    """60/20/20 train / calibration / reference-holdout."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_tr = int(C.TRAIN_FRACTION * len(df))
    n_ca = int(C.CALIB_FRACTION * len(df))
    return (df.iloc[idx[:n_tr]].reset_index(drop=True),
            df.iloc[idx[n_tr:n_tr + n_ca]].reset_index(drop=True),
            df.iloc[idx[n_tr + n_ca:]].reset_index(drop=True))


def train() -> dict:
    ref = load_window(C.REF_STATE, C.REF_YEAR)
    train_df, calib_df, holdout_df = split_reference(ref)

    clf = HistGradientBoostingClassifier(random_state=C.SEED)
    clf.fit(train_df[C.FEATURES].to_numpy(float), train_df["y"].to_numpy())

    # Isotonic calibration on a split the model never saw. CBPE needs calibrated
    # probabilities: its whole premise is that the expected confusion matrix can
    # be read off the score distribution, which is only true under calibration.
    calibrator = CalibratedClassifierCV(clf, method="isotonic", cv="prefit")
    calibrator.fit(calib_df[C.FEATURES].to_numpy(float), calib_df["y"].to_numpy())

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(calibrator, CALIB_PATH)

    holdout_scores = predict(clf, holdout_df)
    holdout_auc = float(roc_auc_score(holdout_df["y"], holdout_scores))

    splits = pd.concat([
        train_df.assign(split="train"),
        calib_df.assign(split="calib"),
        holdout_df.assign(split="holdout"),
    ], ignore_index=True)
    splits.to_parquet(SPLIT_PATH, index=False)

    meta = {
        "task": C.TASK,
        "reference_cell": f"{C.REF_STATE}_{C.REF_YEAR}",
        "n_train": len(train_df), "n_calib": len(calib_df), "n_holdout": len(holdout_df),
        "features": C.FEATURES,
        "model": "HistGradientBoostingClassifier",
        "params": {"random_state": C.SEED},
        "reference_holdout_auc": holdout_auc,
        "model_sha256": _sha256(MODEL_PATH),
        "seed": C.SEED,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    return meta


def predict(clf, df: pd.DataFrame) -> np.ndarray:
    return clf.predict_proba(df[C.FEATURES].to_numpy(float))[:, 1]


def load():
    clf = joblib.load(MODEL_PATH)
    calibrator = joblib.load(CALIB_PATH)
    meta = json.loads(META_PATH.read_text())
    splits = pd.read_parquet(SPLIT_PATH)
    return clf, calibrator, meta, splits


if __name__ == "__main__":
    m = train()
    print(json.dumps(m, indent=2))
