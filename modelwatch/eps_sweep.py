"""Epsilon sweep: how much of the no-drift PSI floor is the smoothing constant?

This module exists because the first version of this project got its headline
wrong. It reported that `PSI > 0.2` fires on 100% of no-drift samples, and
attributed that to sample size and bin count. Most of it was actually the
smoothing floor: at eps=1e-6, categories absent from one side by pure sampling
contribute about (1/n)*ln(1/(n*eps)) each, and ACS OCCP has hundreds of levels.

Rather than delete the wrong number, the project measures it. This sweep runs
the A/A null (no drift by construction) across a range of epsilon values and
reports where the conclusion flips.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from . import detectors
from .data import load_window

OUT_JSON = C.ARTIFACT_DIR / "eps_sweep.json"
OUT_CSV = C.REPORT_DIR / "eps_sweep.csv"


def run(pairs: int = 200, n: int = C.WINDOW_N,
        eps_values=(1e-6, 1e-5, 1e-4, None, 1e-3)) -> pd.DataFrame:
    """`None` means the resolution floor, 0.5/n — the current default."""
    df = load_window(C.REF_STATE, C.REF_YEAR)
    if len(df) < 2 * n:
        raise SystemExit(f"need {2*n} rows for disjoint A/A splits; have {len(df)}")

    rng = np.random.default_rng(C.SEED + 7)
    splits = []
    for _ in range(pairs):
        idx = rng.permutation(len(df))
        splits.append((df.iloc[idx[:n]], df.iloc[idx[n:2 * n]]))

    rows = []
    for eps in eps_values:
        vals = []
        for a, b in splits:
            per_feature = []
            for f in C.NUMERIC:
                per_feature.append(detectors.psi_numeric(
                    a[f].to_numpy(float), b[f].to_numpy(float), eps=eps))
            for f in C.CATEGORICAL:
                per_feature.append(detectors.psi_categorical(
                    a[f].to_numpy(), b[f].to_numpy(), eps=eps))
            vals.append(max(per_feature))
        vals = np.asarray(vals)
        label = "resolution (0.5/n)" if eps is None else f"{eps:.0e}"
        rows.append({
            "eps": label,
            "eps_value": detectors.resolution_eps(n, n) if eps is None else eps,
            "psi_max_median": float(np.median(vals)),
            "psi_max_q99": float(np.quantile(vals, 0.99)),
            "frac_over_folklore_0.2": float((vals > 0.2).mean()),
        })
        print(f"  eps={label:20s} median {rows[-1]['psi_max_median']:.4f}  "
              f"P(>0.2) {rows[-1]['frac_over_folklore_0.2']:.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps({
        "pairs": pairs, "n_per_side": n,
        "sample_resolution_1_over_n": 1.0 / n,
        "rows": rows,
        "interpretation": (
            "The no-drift floor of max-PSI is a function of the smoothing "
            "constant. Any eps far below the sample resolution 1/n inflates it "
            "by charging a large penalty for categories that are absent purely "
            "by chance. The project's original headline used eps=1e-6 and was "
            "an artifact of this; the default is now 0.5/n."),
    }, indent=2))
    print(f"-> {OUT_CSV}")
    return out


if __name__ == "__main__":
    run()
