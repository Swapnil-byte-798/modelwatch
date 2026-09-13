"""Confidence-Based Performance Estimation (CBPE), implemented from scratch.

The point of monitoring is that in production you do not have labels when you
need them. CBPE estimates performance from calibrated scores alone:

    if the model says p=0.8 for 100 rows and the model is calibrated,
    roughly 80 of them are positive -- so the confusion matrix can be
    computed in expectation without ever seeing y.

The structural limitation is the interesting part, and it is why this module is
also a *detector*:

    CBPE assumes P(y|X) is unchanged. It re-weights by the observed feature
    distribution, so it tracks degradation caused by covariate shift, and it is
    blind by construction to concept drift -- a change in the X->y relationship.

That gives a free concept-drift signal: where CBPE's estimate is accurate, the
shift is covariate; where realised performance falls but CBPE says nothing is
wrong, P(y|X) has moved. The residual IS the concept-drift detector.

Deliberately hand-written rather than `pip install nannyml`: nannyml is used
once, in a notebook, as a cross-check oracle.
"""
from __future__ import annotations

import numpy as np


def estimated_confusion(p: np.ndarray, threshold: float = 0.5) -> dict:
    """Expected confusion-matrix cells from calibrated probabilities."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    pred_pos = p >= threshold
    tp = float(p[pred_pos].sum())
    fp = float((1.0 - p[pred_pos]).sum())
    fn = float(p[~pred_pos].sum())
    tn = float((1.0 - p[~pred_pos]).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and (precision + recall) > 0
          else float("nan"))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": (tp + tn) / len(p), "precision": precision,
            "recall": recall, "f1": f1}


def estimated_auc(p: np.ndarray, scores: np.ndarray | None = None) -> float:
    """Expected ROC-AUC under the Bernoulli label model implied by `p`.

    AUC is P(score of a positive > score of a negative). Treating each row as
    positive with probability p_i, the expected number of concordant pairs is

        num = sum_{i,j} p_i (1 - p_j) [ s_i > s_j ]  + 0.5 * ties
        den = (sum p)(sum (1-p)) - sum p_i (1 - p_i)

    Computed in O(n log n) with a sort and a running sum rather than the O(n^2)
    double loop. Ties are given the usual 0.5 credit.
    """
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    s = np.asarray(scores if scores is not None else p, dtype=float)
    if p.size == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    s_sorted, p_sorted = s[order], p[order]
    w_neg = 1.0 - p_sorted

    # Walk groups of equal score in ascending order, accumulating negative mass
    # strictly below the current group.
    num = 0.0
    below = 0.0
    i = 0
    n = len(s_sorted)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        grp = slice(i, j + 1)
        grp_pos = p_sorted[grp].sum()
        grp_neg = w_neg[grp].sum()
        # strictly-greater pairs + half credit for within-group ties
        num += grp_pos * below
        num += 0.5 * (grp_pos * grp_neg - float(np.sum(p_sorted[grp] * w_neg[grp])))
        below += grp_neg
        i = j + 1

    total_pos = p.sum()
    total_neg = (1.0 - p).sum()
    den = total_pos * total_neg - float(np.sum(p * (1.0 - p)))
    if den <= 0:
        return float("nan")
    return float(num / den)


def estimate_window(calibrator, X: np.ndarray, raw_scores: np.ndarray) -> dict:
    """Label-free performance estimate for one window."""
    p_cal = calibrator.predict_proba(X)[:, 1]
    out = estimated_confusion(p_cal)
    out["auc"] = estimated_auc(p_cal, raw_scores)
    out["mean_calibrated_p"] = float(p_cal.mean())
    return out
