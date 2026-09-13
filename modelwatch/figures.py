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

    aa_path = C.ARTIFACT_DIR / "aa_null.parquet"
    aa_max = None
    if aa_path.exists():
        aa_max = float(pd.read_parquet(aa_path)["psi_max"].max())

    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=160)
    ax.axhline(C.HARM_DELTA_AUC, color="#c0392b", lw=1, ls="--",
               label=f"harm threshold ({C.HARM_DELTA_AUC:+.2f} AUC)")

    # The entire no-drift null, drawn to scale. This is the point of the figure:
    # every A/A sample -- where nothing happened by construction -- lives in the
    # sliver at the left edge, and every real window is far to its right.
    if aa_max is not None:
        ax.axvspan(0, aa_max, color="#2c3e50", alpha=0.12, zorder=0)
        ax.annotate(f"no-drift null\n(all {len(pd.read_parquet(aa_path))} A/A samples\n"
                    f"fall below {aa_max:.2f})",
                    xy=(aa_max, y.min()), xytext=(aa_max + 0.35, y.min() * 0.97),
                    fontsize=8, color="#2c3e50",
                    arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=0.8))
    ax.axvline(thr, color="#2c3e50", lw=1.2, ls="--",
               label=f"PSI folklore threshold ({thr})")

    ax.scatter(x[~harmful], y[~harmful], s=30, alpha=0.75, c="#95a5a6",
               label=f"no material harm ({counts['alarm_no_harm']})",
               edgecolors="white", linewidths=0.4)
    ax.scatter(x[harmful], y[harmful], s=34, alpha=0.85, c="#c0392b",
               label=f"materially degraded ({counts['caught']})",
               edgecolors="white", linewidths=0.4)

    # Median PSI per class: the overlap is the argument.
    med_h = float(np.median(x[harmful])) if harmful.any() else None
    med_b = float(np.median(x[~harmful])) if (~harmful).any() else None
    if med_h and med_b:
        ax.axvline(med_h, color="#c0392b", lw=1, alpha=0.55)
        ax.axvline(med_b, color="#7f8c8d", lw=1, alpha=0.55)
        ax.annotate(f"median PSI: degraded {med_h:.1f}  vs  unharmed {med_b:.1f}\n"
                    f"the monitor cannot tell these apart",
                    xy=((med_h + med_b) / 2, y.max()), xytext=(0.30, 0.03),
                    textcoords="axes fraction", fontsize=9, color="#2c3e50",
                    weight="bold", ha="left",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec="#bdc3c7", alpha=0.92))

    ax.set_xlim(-0.15, max(x.max() * 1.04, thr * 2))
    ax.set_xlabel("max per-feature PSI   (label-blind: all the monitor sees)")
    ax.set_ylabel("realised ΔAUC vs reference   (ground truth, never shown to the monitor)")
    ax.set_title(
        f"Every window alerts. {counts['alarm_no_harm']} of {n} had nothing wrong.\n"
        f"{n} real ACS deployment windows, {100*harmful.mean():.0f}% materially degraded",
        fontsize=11.5)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.92)
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
