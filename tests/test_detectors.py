"""Detector unit tests.

NOTE ON SYNTHETIC DRIFT -- read before reusing anything here.

This is the ONLY place in the project where drift is injected by the author.
It exists to verify that the detector implementations respond monotonically to
shift magnitude, i.e. that the code is correct. It is a power curve, not
evidence of detection performance, and no number produced here appears in the
README, the model card, or on a resume. Every headline number in this project
comes from real ACS shift with ground-truth labels the detectors never see.
"""
import numpy as np
import pytest

from modelwatch import cbpe, detectors


def test_psi_zero_on_identical_samples():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20_000)
    assert detectors.psi_numeric(a, a.copy()) < 1e-9


def test_psi_monotone_in_shift_magnitude():
    """Power curve: PSI must increase with injected mean shift."""
    rng = np.random.default_rng(1)
    ref = rng.normal(size=20_000)
    vals = [detectors.psi_numeric(ref, rng.normal(loc=d, size=20_000))
            for d in (0.0, 0.25, 0.5, 1.0)]
    assert vals == sorted(vals), vals
    assert vals[-1] > vals[0]


def test_ks_detects_shift_and_not_noise():
    rng = np.random.default_rng(2)
    ref = rng.normal(size=5_000)
    _, p_same = detectors.ks(ref, rng.normal(size=5_000))
    _, p_shift = detectors.ks(ref, rng.normal(loc=0.3, size=5_000))
    assert p_same > 0.01
    assert p_shift < 1e-6


def test_categorical_psi_and_chi_square():
    rng = np.random.default_rng(3)
    ref = rng.choice(["a", "b", "c"], size=10_000, p=[0.5, 0.3, 0.2])
    same = rng.choice(["a", "b", "c"], size=10_000, p=[0.5, 0.3, 0.2])
    moved = rng.choice(["a", "b", "c"], size=10_000, p=[0.2, 0.3, 0.5])
    assert detectors.psi_categorical(ref, same) < detectors.psi_categorical(ref, moved)
    assert detectors.chi_square(ref, moved)[1] < detectors.chi_square(ref, same)[1]


def test_chi_square_survives_unseen_category():
    ref = np.array(["a"] * 100 + ["b"] * 100)
    cur = np.array(["a"] * 100 + ["c"] * 100)
    chi2, p = detectors.chi_square(ref, cur)
    assert np.isfinite(chi2) and 0.0 <= p <= 1.0


def test_bh_is_more_conservative_than_uncorrected():
    p = [0.04] + [0.9] * 40      # one nominally significant among many
    assert any(v < 0.05 for v in p)
    assert not detectors.bh_reject(p)


# --- CBPE --------------------------------------------------------------------

def test_estimated_auc_matches_sklearn_when_labels_are_certain():
    """With p in {0,1} the expectation collapses to the ordinary AUC."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, size=2_000)
    s = y * 0.6 + rng.normal(scale=0.3, size=2_000)
    assert cbpe.estimated_auc(y.astype(float), s) == pytest.approx(
        roc_auc_score(y, s), abs=1e-9)


def test_estimated_auc_handles_ties():
    p = np.array([1.0, 1.0, 0.0, 0.0])
    s = np.array([0.5, 0.5, 0.5, 0.5])     # everything tied -> AUC 0.5
    assert cbpe.estimated_auc(p, s) == pytest.approx(0.5)


def test_estimated_confusion_recovers_calibrated_counts():
    p = np.array([0.9, 0.8, 0.2, 0.1])
    c = cbpe.estimated_confusion(p, threshold=0.5)
    assert c["tp"] == pytest.approx(1.7)
    assert c["fp"] == pytest.approx(0.3)
    assert c["fn"] == pytest.approx(0.3)
    assert c["tn"] == pytest.approx(1.7)
