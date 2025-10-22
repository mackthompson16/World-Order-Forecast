from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from .utils import WarPeriod, WW1, WW2


def _shade_war(ax, war: WarPeriod, ymin: float = 0.0, ymax: float = 1.0):
    ax.axvspan(war.start, war.end, color="gray", alpha=0.2, ymin=ymin, ymax=ymax, label=war.label)


def plot_composite(df: pd.DataFrame, selected: List[str], outpath: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    dashed_small = {"RUSSIA", "FRANCE", "GERMANY"}
    for country in selected:
        sub = df[df["country"] == country].dropna(subset=["WorldOrderIndex"])  # already smoothed upstream if desired
        if sub.empty:
            continue
        if country in dashed_small:
            ax.plot(
                sub["year"],
                sub["WorldOrderIndex"],
                label=country,
                linewidth=1.5,
                linestyle="--",
            )
        else:
            ax.plot(
                sub["year"],
                sub["WorldOrderIndex"],
                label=country,
                linewidth=2.5,
                linestyle="-",
            )

    _shade_war(ax, WW1)
    _shade_war(ax, WW2)
    ax.set_title("World Order Index (smoothed)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Index (0-1)")
    ax.set_xlim(df["year"].min(), df["year"].max())
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_raw_metrics(df: pd.DataFrame, country: str, outpath: Path):
    metrics = [
        "education",
        "competitiveness",
        "innovation",
        "economic_index",
        "trade_share",
        "military",
        "financial_center",
        "reserve_currency",
    ]
    titles = [
        "Education",
        "Competitiveness (Polity)",
        "Technology/Innovation",
        "Economic Output Share",
        "Trade Share",
        "Military",
        "Financial Center",
        "Reserve Currency",
    ]

    sub = df[df["country"] == country]
    if sub.empty:
        return

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, (m, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        s = sub[["year", m]].dropna()
        if not s.empty:
            ax.plot(s["year"], s[m], color="#1f77b4")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1)
    for ax in axes:
        ax.set_xlim(sub["year"].min(), sub["year"].max())
    fig.suptitle(f"Raw Metrics (smoothed): {country}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
