"""Draw the README figures from output/backtest_*.csv (run backtest.py first)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT, DOCS = ROOT / "output", ROOT / "docs"
COLORS = {"model": "#2f6fdf", "blend": "#16a085", "vegas": "#7f8c8d"}
LABELS = {"model": "Stats model (no betting data)", "blend": "Stats + Vegas line", "vegas": "Vegas favorite"}


def main():
    DOCS.mkdir(exist_ok=True)
    s = pd.read_csv(OUT / "backtest_summary.csv")
    s = s[s.season != "all"]
    bt = pd.read_csv(OUT / "backtest_predictions.csv")
    bt = bt[bt.result != 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    for who in ("vegas", "blend", "model"):
        d = s[s.who == who]
        ax1.plot(d.season.astype(str), d.accuracy * 100, marker="o", lw=2, color=COLORS[who], label=LABELS[who])
    ax1.set_ylabel("Winners picked correctly (%)")
    ax1.set_title("Walk-forward backtest by season (2026 = weeks 1-10)", loc="left", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(frameon=False, fontsize=9, loc="lower right")
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)

    conf = np.maximum(bt.p_model, 1 - bt.p_model)
    right = ((bt.p_model > 0.5) == (bt.result > 0)).astype(float)
    bins = np.array([0.5, 0.6, 0.7, 0.8, 1.0])
    idx = np.digitize(conf, bins[1:-1])
    pred = [conf[idx == i].mean() for i in range(4)]
    act = [right[idx == i].mean() for i in range(4)]
    n = [(idx == i).sum() for i in range(4)]
    ax2.plot([0.5, 0.95], [0.5, 0.95], ls="--", color="#999", lw=1)
    ax2.scatter(pred, act, s=[x / 4 for x in n], color=COLORS["model"], zorder=3)
    for p, a, k in zip(pred, act, n):
        ax2.annotate(f"{k} games", (p, a), textcoords="offset points", xytext=(8, -12), fontsize=8, color="#555")
    ax2.set_xlabel("Stated win probability of the pick")
    ax2.set_ylabel("Actual win rate")
    ax2.set_title("Calibration of the stats model", loc="left", fontsize=11)
    ax2.set_xlim(0.5, 0.95)
    ax2.set_ylim(0.5, 0.95)
    ax2.grid(alpha=0.3)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS / "backtest.png", dpi=150)
    print("wrote", DOCS / "backtest.png")


if __name__ == "__main__":
    main()
