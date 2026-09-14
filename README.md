# ModelWatch

**When a drift monitor fires, what is the probability that the model has actually degraded?**

Portfolio drift projects detect drift their own author injected. That measures nothing: the effect size was chosen by the author, so detection is guaranteed and there is no negative class. This project measures the *decision quality of the monitor itself*, against real distribution shift on US Census data, with ground truth the detectors never see.

## Results

- **224 real deployment windows** — one per (state, year) ACS cell, each fixed at 5,000 rows
- **68.3% materially degraded** (95% CI 61.9–74.0%) — the base rate everything divides by
- **69% alert precision** for the standard PSI > 0.2 / p < 0.05 monitor — which is just the base rate restated, because it fires on 99.6% of windows (1 true negative out of 224)
- **29.8% false-alarm rate** on A/A splits where no drift exists by construction
- **0% silent failures** — degraded windows the monitor never flagged
- **CBPE mean absolute error 0.015 AUC** estimating performance with no labels at all

![what the standard drift monitor is worth](report/quadrants.png)

Each point is a real deployment. The monitor sees only the x axis. **68 windows fired with nothing wrong**; **0 degraded windows never fired at all.**

## Where the standard monitor actually fails

**1. Not where this project first claimed.** An earlier version of this README reported that `PSI > 0.2` sits below the no-drift noise floor and fires on 100% of samples where nothing happened. That was an artifact of the PSI smoothing constant, not a property of the threshold. The measured no-drift median max-PSI is **0.157**, which is below the 0.2 rule of thumb, and the A/A false-alarm rate of the combined folklore policy is **29.8%** — driven by the KS/chi-square p-value arm, not by PSI.

The correction is measured rather than asserted. PSI needs a floor for categories absent from one sample; a floor far below the sample resolution `1/n` charges a large penalty for categories missing by pure chance, and ACS occupation codes have ~465 levels at n=5,000. Sweeping it over the same A/A splits:

| smoothing floor | no-drift median max-PSI | fires at 0.2 |
|---|---|---|
| `1e-06` ← original | 0.293 | 100.0% |
| `1e-05` | 0.226 | 98.5% |
| `1e-04` | 0.156 | 0.0% |
| `resolution (0.5/n)` ← **current default** | 0.156 | 0.0% |
| `1e-03` | 0.072 | 0.0% |

The conclusion flips between `1e-5` and `1e-4`. The default is now `0.5/n` — half a count, the smallest quantity the sample could have resolved. See `DEAD_ENDS.md`.

**2. Drift is ubiquitous; harm is not.** This is the failure that survives. Real drift sits far above the null — median max-PSI **5.16** against a no-drift 99th percentile of **0.183**, and only **3 of 224** windows fall below that percentile. So a threshold strict enough to suppress the no-drift null still admits almost every real window.

And the separation the monitor would need simply is not there: max-PSI is **5.51** on windows where the model materially degraded and **4.75** where it did not. A detector that measures how much the inputs moved is answering a different question from the one the pager is asking.

**3. The degradation is spatial, not temporal.** Windows that hold the state fixed and move only through time (4 of them, California 2015-2018) show **0%** material degradation — realised ΔAUC between +0.003 and −0.003. Windows that change state degrade at **66%**. For this model and this task, "harm" is distance from California, not elapsed time. A monitor tuned on calendar drift would have found nothing to alarm about in four years of data.

## Detector leaderboard

Scored as binary classifiers against realised degradation. **A detector that fires at random scores PR-AUC = the base rate = 0.683**, so that is the line to beat, not 0.5. PR-AUC rather than ROC-AUC because the question is what an alert is worth, not how well the statistic ranks overall.

| detector | PR-AUC | lift over base rate | Spearman vs ΔAUC | p |
|---|---|---|---|---|
| `cbpe_predicted_drop` | 0.977 | +0.294 | -0.836 | 7.3e-60 |
| `psi_mean` | 0.850 | +0.167 | -0.460 | 3.7e-13 |
| `psi_max` | 0.826 | +0.142 | -0.357 | 3.8e-08 |
| `psi_max_categorical` | 0.826 | +0.142 | -0.357 | 3.8e-08 |
| `domain_auc` | 0.818 | +0.134 | -0.346 | 1.1e-07 |
| `ks_max_d` | 0.745 | +0.062 | -0.120 *(n.s.)* | 7.4e-02 |
| `psi_max_numeric` | 0.717 | +0.034 | -0.077 *(n.s.)* | 2.5e-01 |
| `neglog_min_p` | 0.690 | +0.006 | -0.142 | 3.4e-02 |
| `score_psi` | 0.673 | -0.010 | -0.126 *(n.s.)* | 5.9e-02 |

Only `cbpe_predicted_drop` clears the base rate by a wide margin. Several marginal detectors show no statistically significant rank correlation with realised degradation at all — marked *(n.s.)*. Note also that `psi_max` and `psi_max_categorical` are identical to three decimals: max-PSI is entirely determined by the high-cardinality categorical columns (OCCP, POBP), and the numeric-only variant is not significant.

## Alerting policies

| policy | precision | recall | alerts/window | silent failures |
|---|---|---|---|---|
| `v0_naive` | 69% | 100% | 100% | 0% |
| `bh_corrected` | 69% | 100% | 99% | 0% |
| `calibrated_psi` | 69% | 100% | 99% | 0% |

## What this project does not claim

- No detection-delay-in-days or alerts-per-week figures. ACS is annual and cross-sectional; there is no day-level clock, so any such number would be fabricated. See `PREREGISTRATION.md`.
- No seasonality analysis, for the same reason.
- No compliance sign-off. `MODEL_CARD.md` is an *input* to a model risk review, not the outcome of one. Nobody has approved anything.
- Synthetic drift appears only in `tests/test_detectors.py`, as a power-curve fixture verifying the detector code responds to shift. It is never a headline result.

## Reproducing

```bash
make data       # download ACS cells (slow, network)
make reproduce  # everything else, fixed seeds
```

Model pinned at SHA-256 `9d2e38bda39b17f1…`, seed 20260914.

See `PREREGISTRATION.md` for the experimental contract, committed before any results existed, and `DEAD_ENDS.md` for what was rejected and why.
