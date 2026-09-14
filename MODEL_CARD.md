# Model Card — ACSIncome income classifier (monitored subject)

*Generated from measured results by `python -m modelwatch.model_card`. Do not
edit by hand; edit the run and regenerate.*

Generated: 2026-09-14 11:15 UTC

## Model details

| | |
|---|---|
| Task | ACSIncome — binary, `PINCP > 50000` |
| Architecture | HistGradientBoostingClassifier |
| Training data | ACS PUMS CA_2014, 15,000 rows |
| Calibration split | 5,000 rows (isotonic) |
| Reference holdout | 5,000 rows |
| Features | 10 ACS columns, fed as numeric codes |
| Artifact SHA-256 | `9d2e38bda39b17f17f6ddc64968c3796e474a25b8c8d090a393c18c9c6f2098d` |
| Seed | 20260914 |

## Intended use

A research subject for evaluating **drift monitoring**. It exists so that the
monitor has something to watch. It is not intended for, and must not be used
for, any decision about a person's income, credit, eligibility or employment.

## Data lineage

- Source: US Census ACS PUMS 1-Year person files via `folktables`
- Survey years: 2014, 2015, 2016, 2017, 2018
- Reference cell: CA_2014
- Deployment windows: 224 (state, year) cells, each subsampled to
  exactly 5,000 rows

## Evaluation results

| Metric | Value | How obtained |
|---|---|---|
| Reference holdout AUC | 0.8991 | realised, with labels |
| Mean ΔAUC across windows | -0.0272 | realised, with labels |
| Materially degraded windows | 68.3% (153/224) | realised, with labels |
| CBPE mean absolute error | 0.0149 AUC | label-free estimate vs realised |

## What this monitor cannot detect

Measured, not asserted.

- **False alarms.** On A/A splits where no drift exists by construction, the
  folklore monitor (PSI > 0.2, or any p < 0.05) fires on
  **29.8%** of windows.
- **Silent failures.** At that operating point the monitor stays quiet on
  **0.0%** of all windows while the model was degraded — a miss
  rate of **0.0%** among the windows that were materially degraded.
  (The first denominator is every window; the second is only the harmful ones.)
- **Concept drift.** The marginal tests (PSI, KS, chi-square) examine feature
  marginals only. A change in P(y|X) with the feature distribution unchanged is
  invisible to them by construction — this is a property of the statistics, not
  a tuning failure.
- **CBPE's blind spot.** Confidence-based performance estimation assumes P(y|X)
  is stable, so it also cannot see concept drift. Its residual against realised
  performance is used here as the concept-drift signal precisely because of it.
- **Anything below the detection floor.** At n = 5,000 per window, the
  smallest AUC drop distinguishable from noise is bounded by the bootstrap
  width; drops smaller than |0.02| are not classified as harmful at all.

## Known limitations of the model itself

- High-cardinality ACS columns (OCCP, POBP) are fed as numeric codes rather than
  declared categoricals, because they exceed HistGradientBoosting's 255-bin
  categorical limit. This matches the standard folktables baseline and means the
  model treats occupation codes as ordered when they are not.
- Trained on a single state-year. Generalisation across states is the subject
  under study, not a property being claimed.
- No fairness audit has been performed. Per-group metrics are reported by the
  monitor where available, but this model has not been assessed for disparate
  impact and must not be deployed.

## Governance status

This card is **an input to a model risk review, not the outcome of one.** No
review has taken place. No approver has signed anything. The section headings
follow the documentation structure used by common model-risk frameworks so that
a reviewer would find what they expect, which is a statement about layout and
nothing more.
