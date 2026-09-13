"""Grade every detector as a binary classifier against realised degradation.

This is the part no cloned drift project has, because it is the only part that
requires a negative class the author did not manufacture.

Reported with PR-AUC rather than ROC-AUC: harmful windows are a minority class,
and at a low base rate ROC-AUC flatters a detector that a pager would not
tolerate.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import average_precision_score

from . import config as C
from .detectors import bh_reject

# A detector needs enough finite windows, and both classes present, before
# PR-AUC means anything. Below this the leaderboard is reported as unavailable
# rather than computed on a handful of points.
MIN_WINDOWS_FOR_LEADERBOARD = 10

CORPUS_PATH = C.ARTIFACT_DIR / "corpus.parquet"
RESULTS_PATH = C.ARTIFACT_DIR / "benchmark.json"
LEADERBOARD_PATH = C.REPORT_DIR / "leaderboard.csv"


def detector_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Map every detector to a score where HIGHER means 'more likely harmful'."""
    eps = 1e-300
    return {
        "psi_max": df["psi_max"].to_numpy(),
        "psi_max_numeric": df["psi_max_numeric"].to_numpy(),
        "psi_max_categorical": df["psi_max_categorical"].to_numpy(),
        "psi_mean": df["psi_mean"].to_numpy(),
        "ks_max_d": df["ks_max_d"].to_numpy(),
        "neglog_min_p": -np.log(np.clip(df["min_p_any"].to_numpy(), eps, None)),
        "score_psi": df["score_psi"].to_numpy(),
        "domain_auc": df["domain_auc"].to_numpy(),
        "cbpe_predicted_drop": -df["cbpe_delta_auc"].to_numpy(),
    }


def policy_alerts(df: pd.DataFrame, calibrated_psi: float | None = None) -> dict[str, np.ndarray]:
    """Binary alerting policies, including the folklore control arm."""
    psi_cols = [c for c in df.columns if c.startswith("psi__")]
    p_cols = [c for c in df.columns if c.startswith("ksp__") or c.startswith("chip__")]

    naive = ((df[psi_cols] > C.FOLKLORE_PSI_THRESHOLD).any(axis=1)
             | (df[p_cols] < C.FOLKLORE_ALPHA).any(axis=1)).to_numpy()

    bh = np.array([bh_reject(list(row)) for row in df[p_cols].to_numpy()])

    out = {"v0_naive": naive, "bh_corrected": bh}
    if calibrated_psi is not None:
        out["calibrated_psi"] = (df[psi_cols] > calibrated_psi).any(axis=1).to_numpy()
    return out


def score_binary(alert: np.ndarray, harmful: np.ndarray) -> dict:
    tp = int(np.sum(alert & harmful))
    fp = int(np.sum(alert & ~harmful))
    fn = int(np.sum(~alert & harmful))
    tn = int(np.sum(~alert & ~harmful))
    n = alert.size
    return {
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "alert_rate": (tp + fp) / n,
        "alarm_no_harm_rate": fp / n,      # fired, nothing was wrong
        "silent_failure_rate": fn / n,     # stayed quiet, model was degraded
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def run(calibrated_psi: float | None = None) -> dict:
    df = pd.read_parquet(CORPUS_PATH)
    harmful = df["harmful"].to_numpy().astype(bool)
    base_rate = float(harmful.mean())

    # Leaderboard: each detector as a ranker
    rows = []
    for name, s in detector_scores(df).items():
        ok = np.isfinite(s)
        # both classes must be present: PR-AUC is undefined with no positives,
        # and uninformative with no negatives
        if (ok.sum() < MIN_WINDOWS_FOR_LEADERBOARD
                or harmful[ok].sum() == 0
                or (~harmful[ok]).sum() == 0):
            continue
        rho, pval = sps.spearmanr(s[ok], df["delta_auc"].to_numpy()[ok])
        rows.append({
            "detector": name,
            "pr_auc": float(average_precision_score(harmful[ok], s[ok])),
            "spearman_vs_delta_auc": float(rho),
            "spearman_p": float(pval),
            "n": int(ok.sum()),
        })
    cols = ["detector", "pr_auc", "spearman_vs_delta_auc", "spearman_p", "n"]
    leaderboard = pd.DataFrame(rows, columns=cols)
    if leaderboard.empty:
        print(f"WARNING: leaderboard unavailable — need >= "
              f"{MIN_WINDOWS_FOR_LEADERBOARD} windows with both classes present; "
              f"corpus has {len(df)} windows, {int(harmful.sum())} harmful.")
    else:
        leaderboard = leaderboard.sort_values("pr_auc", ascending=False)

    # Policies: each as a pager
    policies = {}
    for name, alert in policy_alerts(df, calibrated_psi).items():
        policies[name] = score_binary(alert, harmful)

    # Pre-registered robustness check: the -0.02 materiality threshold is a
    # judgement fixed in advance, so its influence on the base rate and on every
    # policy score is reported rather than hidden.
    sensitivity = {}
    for thr in (-0.01, -0.02, -0.03):
        h = ((df["delta_auc"] <= thr) & (df["delta_ci_hi"] < 0)).to_numpy()
        entry = {"harm_threshold": thr, "base_rate": float(h.mean()),
                 "n_harmful": int(h.sum())}
        for name, alert in policy_alerts(df, calibrated_psi).items():
            entry[name] = score_binary(alert, h)
        sensitivity[f"{thr}"] = entry

    result = {
        "n_windows": int(len(df)),
        "base_rate_harmful": base_rate,
        "base_rate_ci": _wilson(int(harmful.sum()), len(df)),
        "reference_holdout_auc": float(json.loads(
            (C.ARTIFACT_DIR / "model_meta.json").read_text())["reference_holdout_auc"]),
        "mean_delta_auc": float(df["delta_auc"].mean()),
        "harm_threshold_sensitivity": sensitivity,
        "leaderboard": leaderboard.to_dict("records"),
        "policies": policies,
        "by_shift_kind": {
            k: {"n": int(len(g)), "harmful_rate": float(g["harmful"].mean())}
            for k, g in df.groupby("shift_kind")
        },
        "cbpe": {
            "mae_vs_realised": float(np.nanmean(np.abs(df["cbpe_delta_auc"] - df["delta_auc"]))),
            "mae_harmful": float(np.nanmean(np.abs(
                (df["cbpe_delta_auc"] - df["delta_auc"])[harmful]))) if harmful.any() else None,
            "mae_benign": float(np.nanmean(np.abs(
                (df["cbpe_delta_auc"] - df["delta_auc"])[~harmful]))) if (~harmful).any() else None,
            "mean_residual": float(np.nanmean(df["cbpe_residual"])),
        },
    }
    RESULTS_PATH.write_text(json.dumps(result, indent=2, default=float))
    leaderboard.to_csv(LEADERBOARD_PATH, index=False)
    return result


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return [float((c - h) / d), float((c + h) / d)]


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "leaderboard"}, indent=2))
    print("\nLEADERBOARD")
    print(pd.DataFrame(r["leaderboard"]).to_string(index=False))
