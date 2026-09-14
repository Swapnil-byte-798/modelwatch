"""Label-blind drift detectors.

Every function here sees only feature values and model scores. None of them may
touch `y`. That separation is the whole experiment: these statistics are the
independent variables, and realised performance degradation is the ground truth
they are graded against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from . import config as C


# --- Univariate (marginal) tests --------------------------------------------
# These test feature marginals only. They cannot see a change in P(y|X); that
# limitation is measured rather than asserted (see benchmark.py).


def resolution_eps(n_ref: int, n_cur: int) -> float:
    """Smoothing floor tied to the sample's own resolution: half a count.

    PSI needs a floor because a category present in one sample and absent from
    the other gives log(0). The floor must scale with n, and this is not a
    detail -- it decides the answer.

    A fixed constant far below 1/n turns every zero-count category into a large
    spurious term: the contribution is about (1/n) * ln(1 / (n * eps)), which
    diverges as eps -> 0. On ACS OCCP (~430 levels at n=5,000) there are dozens
    of zero-count cells by pure sampling, and at eps=1e-6 they supply the
    MAJORITY of the PSI value -- measured at 56% for OCCP, 58% for POBP. The
    original version of this project used 1e-6 and reported a no-drift median
    max-PSI of 0.292; more than half of that was this constant. See DEAD_ENDS.md.

    Half a count is the conventional Jeffreys-style floor: it is the smallest
    quantity the sample could have resolved, so it never invents mass the data
    could not have observed, and it never exceeds a genuine single observation.
    """
    return 0.5 / max(min(int(n_ref), int(n_cur)), 1)

def psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = C.PSI_BINS,
                eps: float | None = None) -> float:
    """Population Stability Index over quantile bins fitted on the reference.

    `eps` defaults to the sample resolution (see resolution_eps); pass an
    explicit value only to reproduce the epsilon sweep in DEAD_ENDS.md.

    One pathology left in deliberately: quantile binning collapses on point
    masses (WKHP has a large spike at 40), so the effective bin count can be
    below `bins`. Note that at 10 bins and n=5,000 the floor never binds here --
    it is the categorical path where eps decides the answer.
    """
    if eps is None:
        eps = resolution_eps(len(ref), len(cur))
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    r, _ = np.histogram(ref, bins=edges)
    c, _ = np.histogram(cur, bins=edges)
    r_pct = np.clip(r / max(r.sum(), 1), eps, None)
    c_pct = np.clip(c / max(c.sum(), 1), eps, None)
    return float(np.sum((c_pct - r_pct) * np.log(c_pct / r_pct)))


def psi_categorical(ref: np.ndarray, cur: np.ndarray,
                    eps: float | None = None) -> float:
    """PSI over category frequencies (no binning).

    This is where the smoothing floor matters. OCCP and POBP carry hundreds of
    levels, so at n=5,000 many categories are absent from one side by chance
    alone; with a floor below 1/n those empty cells dominate the statistic.
    `eps` defaults to the sample resolution.
    """
    if eps is None:
        eps = resolution_eps(len(ref), len(cur))
    cats = np.union1d(np.unique(ref), np.unique(cur))
    r = np.array([(ref == k).sum() for k in cats], dtype=float)
    c = np.array([(cur == k).sum() for k in cats], dtype=float)
    r_pct = np.clip(r / max(r.sum(), 1), eps, None)
    c_pct = np.clip(c / max(c.sum(), 1), eps, None)
    return float(np.sum((c_pct - r_pct) * np.log(c_pct / r_pct)))


def ks(ref: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    res = stats.ks_2samp(ref, cur)
    return float(res.statistic), float(res.pvalue)


def chi_square(ref: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    """Chi-square over the contingency table of category counts.

    OCCP and POBP are high-cardinality (hundreds of levels), which is exactly
    where chi-square becomes unreliable: many cells have expected counts below
    5. That is real-world pathology, not a bug, so the columns stay in and the
    behaviour is reported.
    """
    cats = np.union1d(np.unique(ref), np.unique(cur))
    r = np.array([(ref == k).sum() for k in cats], dtype=float)
    c = np.array([(cur == k).sum() for k in cats], dtype=float)
    keep = (r + c) > 0
    table = np.vstack([r[keep], c[keep]])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return 0.0, 1.0
    try:
        chi2, p, _, _ = stats.chi2_contingency(table)
        return float(chi2), float(p)
    except ValueError:
        return 0.0, 1.0


# --- Multivariate ------------------------------------------------------------

def domain_classifier_auc(ref: pd.DataFrame, cur: pd.DataFrame,
                          folds: int = C.DOMAIN_CLF_FOLDS,
                          seed: int = C.SEED) -> float:
    """Train a classifier to tell reference rows from window rows.

    Its cross-validated ROC-AUC is the drift statistic: 0.5 means the two
    samples are indistinguishable, higher means the joint distribution moved.
    This is the only detector here that can see correlation shift -- a change
    in the dependence structure that leaves every marginal intact.
    """
    X = pd.concat([ref[C.FEATURES], cur[C.FEATURES]], axis=0, ignore_index=True)
    y = np.r_[np.zeros(len(ref)), np.ones(len(cur))]
    clf = HistGradientBoostingClassifier(random_state=seed, max_iter=100)
    scores = cross_val_score(clf, X, y, cv=folds, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


# --- Battery -----------------------------------------------------------------

def run_battery(ref: pd.DataFrame, cur: pd.DataFrame,
                ref_scores: np.ndarray, cur_scores: np.ndarray,
                with_domain: bool = True) -> dict:
    """All label-blind statistics for one window, as a flat record."""
    rec: dict[str, float] = {}
    psis, ks_ds, ks_ps, chi_ps = {}, {}, {}, {}

    for f in C.NUMERIC:
        a, b = ref[f].to_numpy(float), cur[f].to_numpy(float)
        psis[f] = psi_numeric(a, b)
        d, p = ks(a, b)
        ks_ds[f], ks_ps[f] = d, p

    for f in C.CATEGORICAL:
        a, b = ref[f].to_numpy(), cur[f].to_numpy()
        psis[f] = psi_categorical(a, b)
        _, p = chi_square(a, b)
        chi_ps[f] = p

    for f, v in psis.items():
        rec[f"psi__{f}"] = v
    for f, v in ks_ds.items():
        rec[f"ksd__{f}"] = v
    for f, v in ks_ps.items():
        rec[f"ksp__{f}"] = v
    for f, v in chi_ps.items():
        rec[f"chip__{f}"] = v

    # Aggregate policies. psi_max is split by feature type because OCCP and POBP
    # carry hundreds of levels: their category-frequency PSI is inflated by rare
    # categories and will dominate a naive max. Whether that helps or hurts is
    # then a measured result rather than a design assumption.
    rec["psi_max"] = max(psis.values())
    rec["psi_max_numeric"] = max((psis[f] for f in C.NUMERIC), default=0.0)
    rec["psi_max_categorical"] = max((psis[f] for f in C.CATEGORICAL), default=0.0)
    rec["psi_mean"] = float(np.mean(list(psis.values())))
    rec["psi_n_over_folklore"] = sum(v > C.FOLKLORE_PSI_THRESHOLD for v in psis.values())
    rec["ks_max_d"] = max(ks_ds.values()) if ks_ds else 0.0
    rec["ks_min_p"] = min(ks_ps.values()) if ks_ps else 1.0
    rec["chi_min_p"] = min(chi_ps.values()) if chi_ps else 1.0
    rec["min_p_any"] = min(rec["ks_min_p"], rec["chi_min_p"])

    # Prediction-score drift: cheap, and usually the strongest marginal signal
    # because the model compresses every feature into one number.
    rec["score_psi"] = psi_numeric(ref_scores, cur_scores)
    rec["score_mean_shift"] = float(cur_scores.mean() - ref_scores.mean())
    d, p = ks(ref_scores, cur_scores)
    rec["score_ks_d"], rec["score_ks_p"] = d, p

    rec["domain_auc"] = domain_classifier_auc(ref, cur) if with_domain else np.nan
    return rec


def bh_reject(pvalues: list[float], alpha: float = C.FOLKLORE_ALPHA) -> bool:
    """Benjamini-Hochberg: does any hypothesis survive FDR control?

    Note for the README: BH assumes independence or positive regression
    dependence. ACS features are strongly correlated (SCHL/OCCP/WKHP), so BH is
    anti-conservative here; Benjamini-Yekutieli would be the strict choice.
    """
    p = np.sort(np.asarray(pvalues, dtype=float))
    m = p.size
    if m == 0:
        return False
    thresh = alpha * np.arange(1, m + 1) / m
    return bool(np.any(p <= thresh))
