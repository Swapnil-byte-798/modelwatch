# Pre-registration

Committed **before any results exist**. Everything below is fixed. If a decision
here changes after results are in hand, the change is recorded in `DEAD_ENDS.md`
with the reason, and the affected numbers are recomputed from scratch.

The point of this file is to foreclose the garden-of-forking-paths objection:
a benchmark whose harm threshold is chosen after seeing which threshold makes
the detectors look good is not a benchmark.

## Question

When a drift monitor fires, what is the probability that model performance has
actually degraded?

Portfolio drift projects universally report that their detector detects drift
they injected themselves. That measures nothing: the effect size was chosen by
the author, so detection is guaranteed and there is no negative class. This
project measures the decision quality of the monitor instead, against real
distribution shift with ground truth the monitor never sees.

## Corpus

- **Data**: US Census ACS PUMS via `folktables` (Ding et al., 2021), 1-Year
  person files, survey years 2014-2018, all 50 states + DC + PR.
- **Task**: `ACSIncome` — binary, `PINCP > 50000`.
- **Reference cell**: California, 2014. Split 60/20/20 into train / calibration
  / reference-holdout with seed 20260914.
- **Deployment windows**: every other (state, year) cell. Expected N is roughly
  250 before exclusions.
- **Fixed window size**: every window is subsampled without replacement to
  exactly n = 5,000 rows. Cells with fewer than 5,000 rows are excluded and the
  exclusions are listed in the README. Unequal n would confound every detector
  statistic, since PSI and KS both scale with sample size.
- **Secondary sweep**: n in {2,000, 25,000} to show threshold sensitivity to n.

## Model under monitoring

One `HistGradientBoostingClassifier` (scikit-learn defaults except
`random_state=20260914`), trained once on the reference train split, serialised
with joblib and pinned by SHA-256. It is never retrained. The monitored model
is deliberately boring: the monitor is the subject of this project, not the
classifier.

## Ground truth

For each window, realised AUC is computed using the true labels, which no
detector is permitted to see.

```
delta_auc(window) = AUC(window) - AUC(reference_holdout)
harmful(window)   = delta_auc <= -0.02
                    AND the 95% bootstrap CI of delta_auc excludes 0
```

Bootstrap: 1,000 resamples, percentile interval, resampling the window and the
reference holdout independently at their fixed sizes.

The `-0.02` threshold is a materiality judgement fixed in advance, not tuned.
Sensitivity to it is reported at -0.01 and -0.03 as a robustness check.

## Detectors under test

All are label-blind. Each returns a scalar per window.

1. Per-feature PSI (10 quantile bins fitted on the reference), aggregated by max
2. Per-feature two-sample KS statistic D (numeric features), aggregated by max
3. Per-feature chi-square (categorical features), aggregated by min p-value
4. PSI on the model's predicted score
5. Domain classifier: `HistGradientBoostingClassifier` trained to separate
   reference from window, 5-fold CV ROC-AUC as the statistic (the only
   multivariate detector)
6. CBPE-estimated delta AUC (label-free performance estimation)

## Alerting policies scored

- `v0-naive`: PSI > 0.2 on any feature, or any KS/chi-square p < 0.05,
  no multiple-testing correction. This is the folklore monitor, and it is the
  control arm.
- `bh`: same statistics with Benjamini-Hochberg correction across features
- `calibrated`: thresholds set at the 99th percentile of the A/A null at the
  deployed n, with effect-size gating
- each single detector at a sweep of thresholds, for the leaderboard

## Primary metrics

Detectors are scored as **binary classifiers against `harmful`**:

- Precision, recall, **PR-AUC** (not ROC-AUC: the positive class is expected to
  be a minority, and reporting detector ROC-AUC at a low base rate is
  uninformative about alert quality)
- Alerts per window
- Spearman correlation between the detector statistic and true `delta_auc`
- The two quadrant rates: **alarm-no-harm** and **silent-failure**

## A/A harness

2,000 random equal-size splits drawn **within** a single (state, year) cell,
where "no drift" is true by construction. This yields the measured false-alarm
rate of each folklore threshold, and the empirical null used to calibrate
replacement thresholds.

## What this design cannot measure

Stated in advance so that no number of this kind is reported later:

- **Detection delay in days / alerts per week.** ACS is annual and
  cross-sectional. There are at most 5 time points per state and no day-level
  clock. Any such figure would be fabricated.
- **Seasonality.** Annual survey data has no intra-year cycle.
- **Streaming error-stream detectors** (ADWIN, DDM, Page-Hinkley). These need a
  sequential labelled error signal that this data does not contain.
- **Selective labelling / reject inference.** Would require inventing an
  approval propensity, i.e. measuring a bias of our own construction.
- **Training/serving skew.** Requires two genuinely divergent feature code
  paths; injecting skew to detect skew is the self-graded exam this project
  exists to avoid.

## Role of synthetic drift

Synthetic shift appears in exactly one place: `tests/test_detectors.py`, as a
known-effect-size fixture used to verify that detector implementations respond
monotonically to shift magnitude. It never appears in a headline result. Any
figure derived from it is labelled as a power curve, not as evidence of
detection performance.
