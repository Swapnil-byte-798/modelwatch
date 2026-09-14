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
import hashlib
import sys
import time

import numpy as np
import pandas as pd

from . import config as C
from . import cbpe, detectors
from .data import available_cells, load_window
from .model import load as load_model, predict

CORPUS_PATH = C.ARTIFACT_DIR / "corpus.parquet"


def cell_generator(state: str, year: int, salt: str = "boot") -> np.random.Generator:
    """An independent RNG stream for one (state, year) cell.

    The full 256-bit SHA-256 digest is handed to SeedSequence as entropy rather
    than truncated to 64 bits: SeedSequence is built to take arbitrary entropy
    and spread it, and discarding three quarters of the digest buys nothing.
    Deriving from the cell's identity (rather than SeedSequence.spawn, whose
    children are ordered) is what keeps a window's draws independent of which
    other cells exist in the corpus.
    """
    digest = hashlib.sha256(f"{salt}|{state}|{year}|{C.SEED}".encode()).digest()
    return np.random.default_rng(np.random.SeedSequence(list(digest)))


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
                       rng: np.random.Generator, chunk: int = 250) -> np.ndarray:
    """Bootstrap distribution of AUC, without re-sorting on every replicate.

    The naive version calls fast_auc per replicate, which re-sorts n points
    every time: 1,000 replicates x 2 sides x 224 windows is ~450,000 sorts and
    measured at 21.7 hours. Since every replicate resamples the SAME fixed
    (y, s), the ordering is invariant -- only the multiplicity of each point
    changes. So sort once, and treat a replicate as a count vector over fixed
    sorted positions:

        concordant = sum_g [ pos_g * (negatives strictly below group g) ]
                     + 0.5 * sum_g [ pos_g * neg_g ]        (ties, half credit)
        AUC        = concordant / (n_pos * n_neg)

    where g indexes groups of tied scores. Verified bit-equal to fast_auc on
    tie-heavy inputs (see tests); 41x faster, 175s -> 4.3s per 1,000 replicates.
    Chunked to bound peak memory at chunk x n floats.
    """
    y = np.asarray(y).astype(np.float64)
    s = np.asarray(s)
    n = y.size
    order = np.argsort(s, kind="mergesort")
    ys = y[order]
    ss = s[order]
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])   # tie-group starts
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)

    out = np.empty(n_boot, dtype=float)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        pos = inv[rng.integers(0, n, (b, n))]
        counts = np.empty((b, n), dtype=np.float64)
        for r in range(b):
            counts[r] = np.bincount(pos[r], minlength=n)
        pos_c = counts * ys
        neg_c = counts - pos_c
        pos_g = np.add.reduceat(pos_c, starts, axis=1)
        neg_g = np.add.reduceat(neg_c, starts, axis=1)
        cum_below = np.cumsum(neg_g, axis=1) - neg_g
        conc = (pos_g * cum_below).sum(axis=1) + 0.5 * (pos_g * neg_g).sum(axis=1)
        n_pos = pos_c.sum(axis=1)
        n_neg = neg_c.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[done:done + b] = np.where(
                (n_pos == 0) | (n_neg == 0), np.nan, conc / (n_pos * n_neg))
        done += b
    return out


def build(limit: int | None = None, with_domain: bool = True,
          n_boot: int = C.BOOTSTRAP_N) -> pd.DataFrame:
    clf, calibrator, meta, splits = load_model()
    holdout = splits[splits["split"] == "holdout"].reset_index(drop=True)
    holdout_y = holdout["y"].to_numpy()
    holdout_scores = predict(clf, holdout)
    holdout_auc = fast_auc(holdout_y, holdout_scores)

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
        cell_rng = cell_generator(state, year)
        # BOTH sides are resampled from this window's own stream. Two properties
        # follow, and both matter:
        #
        #  1. Keyed on cell identity, not loop position. A single sequential
        #     generator would make every window's CI a function of how many
        #     cells sort before it, so extending the corpus from 2 years to 5
        #     would silently rewrite intervals already published. Never key this
        #     on the loop index -- that reintroduces the same coupling.
        #  2. The holdout is resampled per window rather than once and shared.
        #     Sharing one holdout realisation across every window is defensible
        #     for a single marginal CI, but these CIs are aggregated: `harmful`
        #     depends on ci_hi < 0 and the headline base rate is the mean of
        #     those labels. A shared draw correlates all 224 decisions, so one
        #     unlucky holdout resample shifts every label the same way and the
        #     base-rate CI understates the true uncertainty.
        win_boot = bootstrap_auc_dist(y, scores, n_boot, cell_rng)
        hold_boot = bootstrap_auc_dist(holdout_y, holdout_scores, n_boot, cell_rng)
        delta_boot = win_boot - hold_boot
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
