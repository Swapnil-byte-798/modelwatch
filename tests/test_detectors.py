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


# --- Regression tests for defects found in QA --------------------------------
# Each of these corresponds to a confirmed bug. See DEAD_ENDS.md.

def test_resolution_eps_is_half_a_count():
    assert detectors.resolution_eps(5_000, 5_000) == pytest.approx(0.5 / 5_000)
    # takes the smaller sample: the floor must be resolvable by BOTH sides
    assert detectors.resolution_eps(10_000, 2_000) == pytest.approx(0.5 / 2_000)
    assert detectors.resolution_eps(0, 0) == pytest.approx(0.5)  # no ZeroDivisionError


def test_zero_count_categories_do_not_dominate_psi():
    """The headline-reversing bug: a floor far below 1/n makes absent categories
    supply most of the statistic. With the resolution floor they must not."""
    rng = np.random.default_rng(11)
    n, k = 5_000, 400                       # OCCP-like cardinality
    p = rng.dirichlet(np.ones(k) * 0.3)
    a = rng.choice(k, size=n, p=p)
    b = rng.choice(k, size=n, p=p)          # SAME distribution: no drift
    resolution = detectors.psi_categorical(a, b)
    artifact = detectors.psi_categorical(a, b, eps=1e-6)
    # On real ACS OCCP (465 levels, n=5,000) the constant inflates the no-drift
    # value from 0.144 to 0.279 -- a factor of 1.93. Synthetic data with the same
    # cardinality reproduces 1.57-1.90 depending on sparsity, so assert the
    # direction and a conservative magnitude rather than the exact real ratio.
    assert artifact > 1.4 * resolution, (artifact, resolution)
    # The value that matters: with a defensible floor, no-drift data sits BELOW
    # the folklore alarm threshold. With eps=1e-6 on real OCCP it sits above it,
    # which is what produced the original (wrong) headline.
    assert resolution < 0.2, resolution


def test_psi_numeric_is_insensitive_to_eps_at_deployed_settings():
    """The floor is documented as guarding empty bins, but with 10 quantile bins
    at n=5,000 it never binds. Pin that, so the docs stay honest."""
    rng = np.random.default_rng(12)
    a, b = rng.normal(size=5_000), rng.normal(loc=0.2, size=5_000)
    assert detectors.psi_numeric(a, b) == pytest.approx(
        detectors.psi_numeric(a, b, eps=1e-6), abs=1e-12)


def test_bootstrap_stream_depends_on_cell_not_position():
    """Extending the corpus must not rewrite already-published intervals."""
    from modelwatch.corpus import cell_generator

    def draws(state, year, k=8):
        return cell_generator(state, year).integers(0, 10_000, k).tolist()

    # deterministic for a given cell, in any order, in any process
    assert draws("CA", 2018) == draws("CA", 2018)
    # and distinct across cells
    assert draws("CA", 2018) != draws("CA", 2017)
    assert draws("CA", 2018) != draws("TX", 2018)

    # The property that actually matters: a cell's stream is unaffected by which
    # other cells exist. Simulate two corpora whose cell lists differ.
    corpus_a = [("AL", 2014), ("CA", 2018), ("TX", 2018)]
    corpus_b = [("AL", 2014), ("AL", 2015), ("AL", 2016), ("CA", 2018), ("TX", 2018)]
    got_a = {c: cell_generator(*c).integers(0, 10_000, 5).tolist() for c in corpus_a}
    got_b = {c: cell_generator(*c).integers(0, 10_000, 5).tolist() for c in corpus_b}
    for cell in corpus_a:
        assert got_a[cell] == got_b[cell], cell


def test_window_and_holdout_are_resampled_from_the_same_cell_stream():
    """Both sides must come from the window's own stream, so that windows are
    mutually independent -- their CIs are aggregated into the base rate."""
    import inspect
    from modelwatch import corpus

    src = inspect.getsource(corpus.build)
    # the shared-holdout vector must be gone
    assert "holdout_boot" not in src, "holdout bootstrap is shared across windows again"
    assert src.count("bootstrap_auc_dist(") == 2, src.count("bootstrap_auc_dist(")
    assert "hold_boot = bootstrap_auc_dist(holdout_y" in src


def test_two_cells_produce_independent_bootstrap_distributions():
    from modelwatch.corpus import bootstrap_auc_dist, cell_generator
    rng_state = np.random.default_rng(99)
    y = rng_state.integers(0, 2, 400)
    s = y * 0.5 + rng_state.normal(scale=0.4, size=400)
    a = bootstrap_auc_dist(y, s, 50, cell_generator("CA", 2018))
    b = bootstrap_auc_dist(y, s, 50, cell_generator("TX", 2018))
    assert not np.allclose(a, b)
    # same cell reproduces exactly
    a2 = bootstrap_auc_dist(y, s, 50, cell_generator("CA", 2018))
    assert np.allclose(a, a2)


def test_vectorised_bootstrap_matches_reference_auc():
    """bootstrap_auc_dist must stay bit-equal to the per-replicate reference.

    The fast path sorts once and treats each replicate as counts over fixed
    positions. If a future change breaks the tie accounting this catches it.
    """
    from modelwatch.corpus import bootstrap_auc_dist, fast_auc

    rng = np.random.default_rng(5)
    for _ in range(3):
        y = rng.integers(0, 2, 1_500)
        s = np.round(y * 0.5 + rng.normal(scale=0.4, size=1_500), 1)  # heavy ties
        n = y.size
        ref_rng = np.random.default_rng(21)
        fast_rng = np.random.default_rng(21)
        reference = np.array([fast_auc(y[i], s[i])
                              for i in ref_rng.integers(0, n, (25, n))])
        assert np.allclose(bootstrap_auc_dist(y, s, 25, fast_rng),
                           reference, atol=1e-12)


def test_bootstrap_chunking_does_not_change_results():
    from modelwatch.corpus import bootstrap_auc_dist
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, 600)
    s = y * 0.5 + rng.normal(scale=0.4, size=600)
    a = bootstrap_auc_dist(y, s, 40, np.random.default_rng(3), chunk=7)
    b = bootstrap_auc_dist(y, s, 40, np.random.default_rng(3), chunk=40)
    assert np.allclose(a, b, atol=1e-12)
