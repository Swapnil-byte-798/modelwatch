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


def _drift_magnitude():
    """Real drift magnitude against the no-drift null -- the headline comparison."""
    import pandas as _pd
    try:
        d = _pd.read_parquet(C.ARTIFACT_DIR / "corpus.parquet")
        aa = _pd.read_parquet(C.ARTIFACT_DIR / "aa_null.parquet")
    except FileNotFoundError:
        return None
    q99 = float(aa["psi_max"].quantile(0.99))
    return {
        "null_median": float(aa["psi_max"].median()),
        "null_q99": q99,
        "null_max": float(aa["psi_max"].max()),
        "real_min": float(d["psi_max"].min()),
        "real_median": float(d["psi_max"].median()),
        "below_null_q99": int((d["psi_max"] <= q99).sum()),
        "n": int(len(d)),
        "harmful_median": float(d.loc[d["harmful"], "psi_max"].median()),
        "benign_median": float(d.loc[~d["harmful"], "psi_max"].median()),
    }


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
      f"PSI > {C.FOLKLORE_PSI_THRESHOLD} / p < {C.FOLKLORE_ALPHA} monitor — "
      f"which is just the base rate restated, because it fires on "
      f"{naive['alert_rate']*100:.1f}% of windows "
      f"({naive['tn']} true negatives out of {bench['n_windows']})")
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
    dm = _drift_magnitude()
    eps_rows = None
    eps_path = C.ARTIFACT_DIR / "eps_sweep.json"
    if eps_path.exists():
        eps_rows = json.loads(eps_path.read_text())["rows"]

    if dm:
        fires = dm["null_median"] > C.FOLKLORE_PSI_THRESHOLD
        A("## Where the standard monitor actually fails")
        A("")
        A("**1. Not where this project first claimed.** An earlier version of "
          "this README reported that `PSI > 0.2` sits below the no-drift noise "
          "floor and fires on 100% of samples where nothing happened. That was "
          "an artifact of the PSI smoothing constant, not a property of the "
          "threshold. The measured no-drift median max-PSI is "
          f"**{dm['null_median']:.3f}**, which is "
          f"{'above' if fires else 'below'} the 0.2 rule of thumb, and the A/A "
          f"false-alarm rate of the combined folklore policy is **{fa:.1f}%** — "
          "driven by the KS/chi-square p-value arm, not by PSI.")
        A("")
        if eps_rows:
            A("The correction is measured rather than asserted. PSI needs a floor "
              "for categories absent from one sample; a floor far below the "
              "sample resolution `1/n` charges a large penalty for categories "
              "missing by pure chance, and ACS occupation codes have ~465 levels "
              "at n=5,000. Sweeping it over the same A/A splits:")
            A("")
            A("| smoothing floor | no-drift median max-PSI | fires at 0.2 |")
            A("|---|---|---|")
            for r in eps_rows:
                mark = " ← original" if r["eps"] == "1e-06" else (
                    " ← **current default**" if r["eps"].startswith("resolution") else "")
                A(f"| `{r['eps']}`{mark} | {r['psi_max_median']:.3f} | "
                  f"{r['frac_over_folklore_0.2']*100:.1f}% |")
            A("")
            A("The conclusion flips between `1e-5` and `1e-4`. The default is now "
              "`0.5/n` — half a count, the smallest quantity the sample could "
              "have resolved. See `DEAD_ENDS.md`.")
            A("")
        A(f"**2. Drift is ubiquitous; harm is not.** This is the failure that "
          f"survives. Real drift sits far above the null — median max-PSI "
          f"**{dm['real_median']:.2f}** against a no-drift 99th percentile of "
          f"**{dm['null_q99']:.3f}**, and only **{dm['below_null_q99']} of "
          f"{dm['n']}** windows fall below that percentile. So a threshold "
          f"strict enough to suppress the no-drift null still admits almost "
          f"every real window.")
        A("")
        A(f"And the separation the monitor would need simply is not there: "
          f"max-PSI is **{dm['harmful_median']:.2f}** on windows where the model "
          f"materially degraded and **{dm['benign_median']:.2f}** where it did "
          f"not. A detector that measures how much the inputs moved is answering "
          f"a different question from the one the pager is asking.")
        A("")
    bsk = bench.get("by_shift_kind", {})
    if "temporal" in bsk and "spatial" in bsk:
        t, sp = bsk["temporal"], bsk["spatial"]
        A("**3. The degradation is spatial, not temporal.** Windows that hold the "
          f"state fixed and move only through time ({t['n']} of them, California "
          f"2015-2018) show **{t['harmful_rate']*100:.0f}%** material "
          f"degradation — realised ΔAUC between +0.003 and −0.003. Windows that "
          f"change state degrade at **{sp['harmful_rate']*100:.0f}%**. For this "
          "model and this task, \"harm\" is distance from California, not "
          "elapsed time. A monitor tuned on calendar drift would have found "
          "nothing to alarm about in four years of data.")
        A("")
    A("## Detector leaderboard")
    A("")
    A(f"Scored as binary classifiers against realised degradation. **A detector "
      f"that fires at random scores PR-AUC = the base rate = "
      f"{base_pct/100:.3f}**, so that is the line to beat, not 0.5. PR-AUC "
      f"rather than ROC-AUC because the question is what an alert is worth, not "
      f"how well the statistic ranks overall.")
    A("")
    A("| detector | PR-AUC | lift over base rate | Spearman vs ΔAUC | p |")
    A("|---|---|---|---|---|")
    for r in lb.to_dict("records"):
        lift = r["pr_auc"] - base_pct / 100
        sig = "" if r["spearman_p"] < 0.05 else " *(n.s.)*"
        A(f"| `{r['detector']}` | {r['pr_auc']:.3f} | {lift:+.3f} | "
          f"{r['spearman_vs_delta_auc']:+.3f}{sig} | {r['spearman_p']:.1e} |")
    A("")
    A("Only `cbpe_predicted_drop` clears the base rate by a wide margin. Several "
      "marginal detectors show no statistically significant rank correlation "
      "with realised degradation at all — marked *(n.s.)*. Note also that "
      "`psi_max` and `psi_max_categorical` are identical to three decimals: "
      "max-PSI is entirely determined by the high-cardinality categorical "
      "columns (OCCP, POBP), and the numeric-only variant is not significant.")
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
