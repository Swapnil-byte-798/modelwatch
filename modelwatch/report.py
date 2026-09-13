"""Render README.md from measured results.

No number in the README is ever typed by hand. `make reproduce` regenerates
this file from the artifacts, so if a number in the README disagrees with the
code, the README is stale and the build will say so.
"""
from __future__ import annotations

import json

import pandas as pd

from . import config as C

README = C.ROOT / "README.md"


def _load(name):
    p = C.ARTIFACT_DIR / name
    if not p.exists():
        p = C.REPORT_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def generate() -> str:
    meta = _load("model_meta.json")
    bench = _load("benchmark.json")
    aa = _load("aa_summary.json")
    quad = _load("quadrants.json")
    if not (meta and bench):
        raise SystemExit("run `make corpus && make benchmark` first")

    naive = bench["policies"]["v0_naive"]
    bh = bench["policies"].get("bh_corrected", {})
    lb = pd.DataFrame(bench["leaderboard"])
    base_pct = bench["base_rate_harmful"] * 100
    ci = bench["base_rate_ci"]
    fa = aa["false_alarm_rate"]["v0_naive"] * 100 if aa else float("nan")

    lines = []
    A = lines.append
    A("# ModelWatch")
    A("")
    A("**When a drift monitor fires, what is the probability that the model has "
      "actually degraded?**")
    A("")
    A("Portfolio drift projects detect drift their own author injected. That "
      "measures nothing: the effect size was chosen by the author, so detection "
      "is guaranteed and there is no negative class. This project measures the "
      "*decision quality of the monitor itself*, against real distribution shift "
      "on US Census data, with ground truth the detectors never see.")
    A("")
    A("## Results")
    A("")
    A(f"- **{bench['n_windows']} real deployment windows** — one per (state, year) "
      f"ACS cell, each fixed at {C.WINDOW_N:,} rows")
    A(f"- **{base_pct:.1f}% materially degraded** (95% CI "
      f"{ci[0]*100:.1f}–{ci[1]*100:.1f}%) — the base rate everything divides by")
    A(f"- **{naive['precision']*100:.0f}% alert precision** for the standard "
      f"PSI > {C.FOLKLORE_PSI_THRESHOLD} / p < {C.FOLKLORE_ALPHA} monitor "
      f"(fires on {naive['alert_rate']*100:.0f}% of windows)")
    if aa:
        A(f"- **{fa:.1f}% false-alarm rate** on A/A splits where no drift exists "
          f"by construction")
    A(f"- **{naive['silent_failure_rate']*100:.0f}% silent failures** — degraded "
      f"windows the monitor never flagged")
    A(f"- **CBPE mean absolute error {bench['cbpe']['mae_vs_realised']:.3f} AUC** "
      f"estimating performance with no labels at all")
    A("")
    if quad:
        A("![what the standard drift monitor is worth](report/quadrants.png)")
        A("")
        A(f"Each point is a real deployment. The monitor sees only the x axis. "
          f"**{quad['alarm_no_harm']} windows fired with nothing wrong**; "
          f"**{quad['silent_failure']} degraded windows never fired at all.**")
        A("")
    A("## Detector leaderboard")
    A("")
    A("Scored as binary classifiers against realised degradation. PR-AUC, not "
      "ROC-AUC: harmful windows are the minority class, and ROC-AUC flatters a "
      "detector that a pager would not tolerate.")
    A("")
    A("| detector | PR-AUC | Spearman vs ΔAUC |")
    A("|---|---|---|")
    for r in lb.to_dict("records"):
        A(f"| `{r['detector']}` | {r['pr_auc']:.3f} | {r['spearman_vs_delta_auc']:+.3f} |")
    A("")
    A("## Alerting policies")
    A("")
    A("| policy | precision | recall | alerts/window | silent failures |")
    A("|---|---|---|---|---|")
    for name, p in bench["policies"].items():
        A(f"| `{name}` | {p['precision']*100:.0f}% | {p['recall']*100:.0f}% | "
          f"{p['alert_rate']*100:.0f}% | {p['silent_failure_rate']*100:.0f}% |")
    A("")
    A("## What this project does not claim")
    A("")
    A("- No detection-delay-in-days or alerts-per-week figures. ACS is annual "
      "and cross-sectional; there is no day-level clock, so any such number "
      "would be fabricated. See `PREREGISTRATION.md`.")
    A("- No seasonality analysis, for the same reason.")
    A("- No compliance sign-off. `MODEL_CARD.md` is an *input* to a model risk "
      "review, not the outcome of one. Nobody has approved anything.")
    A("- Synthetic drift appears only in `tests/test_detectors.py`, as a "
      "power-curve fixture verifying the detector code responds to shift. It is "
      "never a headline result.")
    A("")
    A("## Reproducing")
    A("")
    A("```bash")
    A("make data       # download ACS cells (slow, network)")
    A("make reproduce  # everything else, fixed seeds")
    A("```")
    A("")
    A(f"Model pinned at SHA-256 `{meta['model_sha256'][:16]}…`, seed {meta['seed']}.")
    A("")
    A("See `PREREGISTRATION.md` for the experimental contract, committed before "
      "any results existed, and `DEAD_ENDS.md` for what was rejected and why.")
    A("")

    text = "\n".join(lines)
    README.write_text(text)
    print(f"-> {README}")
    return text


if __name__ == "__main__":
    generate()
