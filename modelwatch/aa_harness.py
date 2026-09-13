"""A/A harness: measure the false-alarm rate on data where nothing happened.

Every detector threshold in circulation -- PSI > 0.2, p < 0.05 -- is folklore
until someone measures what it does on a sample where the null is true by
construction. That is what this does: draw two disjoint equal-sized chunks from
*within a single (state, year) cell*, so there is no drift by construction, and
run the full battery.

Two outputs:
  1. the measured false-alarm rate of each folklore threshold at the deployed n
  2. the empirical 99th-percentile null, used to replace folklore thresholds
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import config as C
from . import detectors
from .data import load_window
from .model import load as load_model, predict

AA_PATH = C.ARTIFACT_DIR / "aa_null.parquet"
AA_SUMMARY = C.ARTIFACT_DIR / "aa_summary.json"


def run(cell: tuple[str, int] | None = None, pairs: int = C.AA_PAIRS,
        n: int = C.WINDOW_N, with_domain: bool = False) -> pd.DataFrame:
    """Draw `pairs` disjoint A/A splits and run the battery on each.

    The domain classifier is off by default here: it costs a 5-fold model fit
    per pair, and 2,000 pairs of that is hours. Its null is estimated from a
    smaller pair count separately.
    """
    state, year = cell or (C.REF_STATE, C.REF_YEAR)
    df = load_window(state, year)
    clf, _, _, _ = load_model()
    scores_all = predict(clf, df)

    if len(df) < 2 * n:
        raise SystemExit(
            f"cell {state}_{year} has {len(df)} rows; need {2*n} for disjoint A/A "
            f"splits at n={n}. Use the reference cell or lower --n.")

    rng = np.random.default_rng(C.SEED + 1)
    rows = []
    for i in range(pairs):
        idx = rng.permutation(len(df))
        a_idx, b_idx = idx[:n], idx[n:2 * n]
        a, b = df.iloc[a_idx], df.iloc[b_idx]
        rec = detectors.run_battery(a, b, scores_all[a_idx], scores_all[b_idx],
                                    with_domain=with_domain)
        rows.append(rec)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{pairs} A/A pairs", flush=True)

    aa = pd.DataFrame(rows)
    aa.to_parquet(AA_PATH, index=False)

    psi_cols = [c for c in aa.columns if c.startswith("psi__")]
    p_cols = [c for c in aa.columns if c.startswith("ksp__") or c.startswith("chip__")]

    naive_fire = ((aa[psi_cols] > C.FOLKLORE_PSI_THRESHOLD).any(axis=1)
                  | (aa[p_cols] < C.FOLKLORE_ALPHA).any(axis=1)).mean()
    bh_fire = np.mean([detectors.bh_reject(list(r)) for r in aa[p_cols].to_numpy()])

    summary = {
        "cell": f"{state}_{year}", "pairs": pairs, "n_per_side": n,
        "false_alarm_rate": {
            "v0_naive": float(naive_fire),
            "bh_corrected": float(bh_fire),
            "psi_max_over_folklore": float((aa["psi_max"] > C.FOLKLORE_PSI_THRESHOLD).mean()),
            "any_p_below_alpha": float((aa[p_cols] < C.FOLKLORE_ALPHA).any(axis=1).mean()),
        },
        "calibrated_thresholds": {
            "psi_max_q99": float(np.quantile(aa["psi_max"], C.AA_CALIBRATION_QUANTILE)),
            "psi_per_feature_q99": float(np.quantile(aa[psi_cols].to_numpy().ravel(),
                                                     C.AA_CALIBRATION_QUANTILE)),
            "ks_max_d_q99": float(np.quantile(aa["ks_max_d"], C.AA_CALIBRATION_QUANTILE)),
            "score_psi_q99": float(np.quantile(aa["score_psi"], C.AA_CALIBRATION_QUANTILE)),
        },
        "null_summary": {
            "psi_max_median": float(aa["psi_max"].median()),
            "psi_max_max": float(aa["psi_max"].max()),
        },
    }
    AA_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return aa


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=C.AA_PAIRS)
    ap.add_argument("--n", type=int, default=C.WINDOW_N)
    ap.add_argument("--domain", action="store_true")
    a = ap.parse_args(argv)
    run(pairs=a.pairs, n=a.n, with_domain=a.domain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
