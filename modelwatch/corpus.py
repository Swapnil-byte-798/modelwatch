"""Build the deployment corpus: ground truth + label-blind detector statistics.

For every (state, year) window this records two independent things:

  * ground truth  -- realised delta AUC against the reference holdout, with a
    bootstrap CI, computed WITH labels. No detector ever sees this.
  * detector statistics -- the label-blind battery.

Keeping them in one table but computing them in strict isolation is what makes
the benchmark honest.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from . import config as C
from . import cbpe, detectors
from .data import available_cells, load_window
from .model import load as load_model, predict

CORPUS_PATH = C.ARTIFACT_DIR / "corpus.parquet"


def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based ROC-AUC with tie handling. ~50x faster than sklearn in a loop."""
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = np.asarray(s)[order]
    ranks = np.empty(s.size, dtype=float)
    i = 0
    while i < s_sorted.size:
        j = i
        while j + 1 < s_sorted.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0   # average rank, 1-based
        i = j + 1
    r = np.empty(s.size, dtype=float)
    r[order] = ranks
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_auc_dist(y: np.ndarray, s: np.ndarray, n_boot: int,
                       rng: np.random.Generator) -> np.ndarray:
    n = y.size
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        out[b] = fast_auc(y[idx], s[idx])
    return out


def build(limit: int | None = None, with_domain: bool = True,
          n_boot: int = C.BOOTSTRAP_N) -> pd.DataFrame:
    clf, calibrator, meta, splits = load_model()
    holdout = splits[splits["split"] == "holdout"].reset_index(drop=True)
    holdout_y = holdout["y"].to_numpy()
    holdout_scores = predict(clf, holdout)
    holdout_auc = fast_auc(holdout_y, holdout_scores)

    rng = np.random.default_rng(C.SEED)
    # Resampled once and reused: delta_b = auc_window_b - auc_holdout_b with the
    # two resampled independently, so the holdout draws do not need redoing per
    # window. Saves ~250x the work with identical semantics.
    holdout_boot = bootstrap_auc_dist(holdout_y, holdout_scores, n_boot, rng)

    cells = [(s, y) for (s, y) in available_cells()
             if not (s == C.REF_STATE and y == C.REF_YEAR)]
    if limit:
        cells = cells[:limit]

    lo_q = (1 - C.BOOTSTRAP_CI) / 2
    hi_q = 1 - lo_q
    rows = []
    t0 = time.time()
    for k, (state, year) in enumerate(cells, 1):
        win = load_window(state, year)
        y = win["y"].to_numpy()
        scores = predict(clf, win)

        realised_auc = fast_auc(y, scores)
        win_boot = bootstrap_auc_dist(y, scores, n_boot, rng)
        delta_boot = win_boot - holdout_boot
        ci_lo, ci_hi = np.quantile(delta_boot, [lo_q, hi_q])
        delta_auc = realised_auc - holdout_auc
        harmful = bool(delta_auc <= C.HARM_DELTA_AUC and ci_hi < 0)

        rec = {
            "state": state, "year": year,
            "shift_kind": ("temporal" if state == C.REF_STATE else
                           ("spatial" if year == C.REF_YEAR else "both")),
            "n": len(win),
            "realised_auc": realised_auc,
            "delta_auc": delta_auc,
            "delta_ci_lo": float(ci_lo), "delta_ci_hi": float(ci_hi),
            "harmful": harmful,
            "positive_rate": float(y.mean()),
        }
        rec.update(detectors.run_battery(holdout, win, holdout_scores, scores,
                                         with_domain=with_domain))
        est = cbpe.estimate_window(calibrator, win[C.FEATURES].to_numpy(float), scores)
        rec["cbpe_auc"] = est["auc"]
        rec["cbpe_delta_auc"] = est["auc"] - holdout_auc
        rec["cbpe_accuracy"] = est["accuracy"]
        # Positive residual => realised is WORSE than CBPE predicted => the part
        # of the degradation that covariate shift cannot explain.
        rec["cbpe_residual"] = rec["cbpe_delta_auc"] - delta_auc
        rows.append(rec)

        if k % 10 == 0 or k == len(cells):
            el = time.time() - t0
            print(f"  {k}/{len(cells)} cells  {el:.0f}s elapsed "
                  f"({el/k:.1f}s/cell)", flush=True)

    df = pd.DataFrame(rows)
    df.attrs["holdout_auc"] = holdout_auc
    df.to_parquet(CORPUS_PATH, index=False)
    print(f"\nreference holdout AUC: {holdout_auc:.4f}")
    print(f"windows: {len(df)}   harmful: {int(df['harmful'].sum())} "
          f"({100*df['harmful'].mean():.1f}%)")
    print(f"-> {CORPUS_PATH}")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-domain", action="store_true",
                    help="skip the domain classifier (fast pass)")
    ap.add_argument("--boot", type=int, default=C.BOOTSTRAP_N)
    a = ap.parse_args(argv)
    build(limit=a.limit, with_domain=not a.no_domain, n_boot=a.boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
