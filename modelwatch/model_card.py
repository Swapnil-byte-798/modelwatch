"""Generate MODEL_CARD.md from measured results.

The standard sections are table stakes. The section that is not standard, and
which is the point of this file, is **What this monitor cannot detect** -- it is
populated from the project's own measured false-alarm and miss rates rather
than written by hand.

A model card that quotes its own miss rate is an honest input to a model risk
review. Note what this file never says: it does not claim sign-off, approval,
audit-readiness, or compliance with any framework. Those are outcomes produced
by a review function that does not exist here. It claims the artifact only.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from . import config as C

CARD_PATH = C.ROOT / "MODEL_CARD.md"

TEMPLATE = """# Model Card — {task} income classifier (monitored subject)

*Generated from measured results by `python -m modelwatch.model_card`. Do not
edit by hand; edit the run and regenerate.*

Generated: {generated}

## Model details

| | |
|---|---|
| Task | {task} — binary, `PINCP > 50000` |
| Architecture | {architecture} |
| Training data | ACS PUMS {ref_cell}, {n_train:,} rows |
| Calibration split | {n_calib:,} rows (isotonic) |
| Reference holdout | {n_holdout:,} rows |
| Features | {n_features} ACS columns, fed as numeric codes |
| Artifact SHA-256 | `{sha}` |
| Seed | {seed} |

## Intended use

A research subject for evaluating **drift monitoring**. It exists so that the
monitor has something to watch. It is not intended for, and must not be used
for, any decision about a person's income, credit, eligibility or employment.

## Data lineage

- Source: US Census ACS PUMS 1-Year person files via `folktables`
- Survey years: {years}
- Reference cell: {ref_cell}
- Deployment windows: {n_windows} (state, year) cells, each subsampled to
  exactly {window_n:,} rows

## Evaluation results

| Metric | Value | How obtained |
|---|---|---|
| Reference holdout AUC | {holdout_auc:.4f} | realised, with labels |
| Mean ΔAUC across windows | {mean_delta:+.4f} | realised, with labels |
| Materially degraded windows | {base_rate_pct:.1f}% ({harm_n}/{n_windows}) | realised, with labels |
| CBPE mean absolute error | {cbpe_mae:.4f} AUC | label-free estimate vs realised |

## What this monitor cannot detect

Measured, not asserted.

- **False alarms.** On A/A splits where no drift exists by construction, the
  folklore monitor (PSI > {psi_thr}, or any p < {alpha}) fires on
  **{fa_naive_pct:.1f}%** of windows.
- **Silent failures.** At that operating point the monitor misses
  **{silent_pct:.1f}%** of all windows that were materially degraded.
- **Concept drift.** The marginal tests (PSI, KS, chi-square) examine feature
  marginals only. A change in P(y|X) with the feature distribution unchanged is
  invisible to them by construction — this is a property of the statistics, not
  a tuning failure.
- **CBPE's blind spot.** Confidence-based performance estimation assumes P(y|X)
  is stable, so it also cannot see concept drift. Its residual against realised
  performance is used here as the concept-drift signal precisely because of it.
- **Anything below the detection floor.** At n = {window_n:,} per window, the
  smallest AUC drop distinguishable from noise is bounded by the bootstrap
  width; drops smaller than |{harm_delta}| are not classified as harmful at all.

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
"""


def generate() -> str:
    meta = json.loads((C.ARTIFACT_DIR / "model_meta.json").read_text())
    bench = json.loads((C.ARTIFACT_DIR / "benchmark.json").read_text())
    try:
        aa = json.loads((C.ARTIFACT_DIR / "aa_summary.json").read_text())
        fa_naive = aa["false_alarm_rate"]["v0_naive"] * 100
    except FileNotFoundError:
        fa_naive = float("nan")

    naive = bench["policies"]["v0_naive"]
    harm_n = naive["tp"] + naive["fn"]

    text = TEMPLATE.format(
        task=meta["task"],
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        architecture=meta["model"],
        ref_cell=meta["reference_cell"],
        n_train=meta["n_train"], n_calib=meta["n_calib"], n_holdout=meta["n_holdout"],
        n_features=len(meta["features"]),
        sha=meta["model_sha256"], seed=meta["seed"],
        years=", ".join(str(y) for y in C.YEARS),
        n_windows=bench["n_windows"], window_n=C.WINDOW_N,
        holdout_auc=bench["reference_holdout_auc"],
        mean_delta=bench["mean_delta_auc"],
        base_rate_pct=bench["base_rate_harmful"] * 100,
        harm_n=harm_n,
        cbpe_mae=bench["cbpe"]["mae_vs_realised"],
        psi_thr=C.FOLKLORE_PSI_THRESHOLD, alpha=C.FOLKLORE_ALPHA,
        fa_naive_pct=fa_naive,
        silent_pct=naive["silent_failure_rate"] * 100,
        harm_delta=abs(C.HARM_DELTA_AUC),
    )
    CARD_PATH.write_text(text)
    print(f"-> {CARD_PATH}")
    return text


if __name__ == "__main__":
    generate()
