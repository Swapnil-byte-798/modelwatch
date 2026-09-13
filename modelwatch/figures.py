"""The one figure that carries the project.

Scatter of every deployment window: the folklore drift statistic on x, the
degradation it was supposed to predict on y. Two quadrants are the whole
argument -- windows where the monitor fired and nothing was wrong, and windows
where the model degraded and the monitor stayed silent.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

CORPUS_PATH = C.ARTIFACT_DIR / "corpus.parquet"
QUADRANT_PNG = C.REPORT_DIR / "quadrants.png"


def quadrant_plot(threshold: float | None = None, out=QUADRANT_PNG) -> dict:
    df = pd.read_parquet(CORPUS_PATH)
    thr = threshold if threshold is not None else C.FOLKLORE_PSI_THRESHOLD
    x = df["psi_max"].to_numpy()
    y = df["delta_auc"].to_numpy()
    harmful = df["harmful"].to_numpy().astype(bool)
    fired = x > thr
    n = len(df)

    counts = {
        "alarm_no_harm": int(np.sum(fired & ~harmful)),
        "caught": int(np.sum(fired & harmful)),
        "silent_failure": int(np.sum(~fired & harmful)),
        "quiet_ok": int(np.sum(~fired & ~harmful)),
    }

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=160)
    ax.axhspan(C.HARM_DELTA_AUC, y.min() - 0.02 if y.size else -0.1,
               color="#c0392b", alpha=0.05)
    ax.axhline(C.HARM_DELTA_AUC, color="#c0392b", lw=1, ls="--",
               label=f"harm threshold ({C.HARM_DELTA_AUC:+.2f} AUC)")
    ax.axvline(thr, color="#2c3e50", lw=1, ls="--",
               label=f"PSI folklore threshold ({thr})")

    ax.scatter(x[~harmful], y[~harmful], s=16, alpha=0.55, c="#7f8c8d",
               label="no material harm", edgecolors="none")
    ax.scatter(x[harmful], y[harmful], s=26, alpha=0.85, c="#c0392b",
               label="materially degraded", edgecolors="none")

    xmax = max(x.max() * 1.05, thr * 1.6) if x.size else 1.0
    ax.set_xlim(0, xmax)
    ax.annotate(f"ALARM / NO HARM\n{counts['alarm_no_harm']} windows "
                f"({100*counts['alarm_no_harm']/n:.0f}%)",
                xy=(thr + 0.02 * xmax, max(y.max(), 0.005) * 0.75),
                fontsize=9, color="#2c3e50", weight="bold")
    ax.annotate(f"SILENT FAILURE\n{counts['silent_failure']} windows "
                f"({100*counts['silent_failure']/n:.0f}%)",
                xy=(0.02 * xmax, min(y.min(), C.HARM_DELTA_AUC) * 0.9),
                fontsize=9, color="#c0392b", weight="bold")

    ax.set_xlabel("max per-feature PSI  (label-blind: what the monitor sees)")
    ax.set_ylabel("realised ΔAUC vs reference  (ground truth: what actually happened)")
    ax.set_title(f"What the standard drift monitor is worth\n"
                 f"{n} real ACS deployment windows, "
                 f"{100*harmful.mean():.0f}% materially degraded", fontsize=11)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

    counts["threshold"] = thr
    counts["n"] = n
    (C.REPORT_DIR / "quadrants.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps(counts, indent=2))
    print(f"-> {out}")
    return counts


if __name__ == "__main__":
    quadrant_plot()
