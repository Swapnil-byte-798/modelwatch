"""Frozen experimental parameters.

Every number in this file is part of the pre-registration (PREREGISTRATION.md).
Changing one after results exist invalidates the benchmark, so they live here
rather than being passed around as arguments.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "acs_cache"      # raw census downloads, deleted after extraction
WINDOWS_DIR = DATA_DIR / "windows"      # per (state, year) parquet, subsampled
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "report"

# --- Task -------------------------------------------------------------------
TASK = "ACSIncome"
REF_STATE = "CA"
REF_YEAR = 2014
YEARS = [2014, 2015, 2016, 2017, 2018]

# ACSIncome feature set (folktables). RELP is valid for 2014-2018; the ACS
# renamed it RELSHIPP in 2019, which is one reason the corpus stops at 2018.
FEATURES = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]
CATEGORICAL = ["COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"]
NUMERIC = ["AGEP", "WKHP"]

# --- Window construction ----------------------------------------------------
# Fixed n across every window. This is the single most important implementation
# decision in the project: PSI and KS are functions of sample size, so unequal
# window sizes (CA has ~100x the rows of WY) would silently confound every
# detector comparison. Windows with fewer than WINDOW_N rows are dropped.
WINDOW_N = 5_000
WINDOW_N_SWEEP = [2_000, 25_000]        # secondary sensitivity sweep

# The reference cell is drawn larger so that its 20% holdout is exactly
# WINDOW_N rows. Comparing a 1,000-row reference against 5,000-row windows would
# hand every two-sample test an asymmetry that has nothing to do with drift.
REF_N = WINDOW_N * 5                    # 25,000 -> 15k train / 5k calib / 5k holdout

# --- Ground truth -----------------------------------------------------------
# A window is "harmful" iff realised AUC drops by at least HARM_DELTA_AUC
# against the reference holdout AND the bootstrap CI for that drop excludes 0.
HARM_DELTA_AUC = -0.02
BOOTSTRAP_N = 1_000
BOOTSTRAP_CI = 0.95

# --- Detectors --------------------------------------------------------------
PSI_BINS = 10
PSI_EPSILON = 1e-6                      # guards empty reference bins; PSI is
                                        # unbounded without it, which is itself
                                        # a documented pathology (see README)
FOLKLORE_PSI_THRESHOLD = 0.2            # the credit-scoring rule of thumb we test
FOLKLORE_ALPHA = 0.05
DOMAIN_CLF_FOLDS = 5

# --- A/A harness ------------------------------------------------------------
AA_PAIRS = 2_000                        # random within-cell splits: no drift by construction
AA_CALIBRATION_QUANTILE = 0.99

# --- Reproducibility --------------------------------------------------------
SEED = 20260914
TRAIN_FRACTION = 0.6                    # of the reference cell
CALIB_FRACTION = 0.2                    # isotonic calibration for CBPE
# remaining 0.2 is the reference holdout used for all baseline metrics

for _d in (DATA_DIR, CACHE_DIR, WINDOWS_DIR, ARTIFACT_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
