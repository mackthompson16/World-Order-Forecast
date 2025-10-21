from pathlib import Path
from typing import List
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_top(series: pd.DataFrame, areas: pd.DataFrame, out_path: Path) -> None:
    rank_col = "Average" if "Average" in areas.columns else ("Area" if "Area" in areas.columns else areas.columns[-1])
    ranked = areas.sort_values(rank_col, ascending=False)["Country"].tolist()
    plt.figure(figsize=(12, 7))
    plotted = 0
    for c in ranked:
        if plotted >= 5:
            break
        s = series[series["Country"] == c].sort_values("Year")
        if "AvailableCount" in s.columns:
            s = s[s["AvailableCount"] >= 3]
        if s.empty:
            continue
        plt.plot(s["Year"], s["CompositeStanding"], label=c)
        plotted += 1
    plt.title("Empire Composite Standing — Top 5 by Area (dynamic start)")
    plt.xlabel("Year")
    plt.ylabel("Normalized Composite Standing (0–1)")
    years = pd.to_numeric(series["Year"], errors="coerce").dropna()
    if not years.empty:
        xmin, xmax = int(years.min()), int(years.max())
        plt.xlim(xmin, xmax)
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_country_metric_breakdowns(composite: pd.DataFrame, countries: List[str], out_dir: Path) -> None:
    df = composite.copy()
    norm_cols = [c for c in df.columns if c.endswith("_norm")]
    ord_names = ["first", "second", "third", "fourth", "fifth"]
    for idx, country in enumerate(countries):
        sub = df[df["Country"] == country].copy()
        if sub.empty:
            continue
        sub["Year"] = pd.to_numeric(sub["Year"], errors="coerce")
        sub = sub.dropna(subset=["Year"]).sort_values("Year")
        if "AvailableCount" in sub.columns:
            valid_sub = sub[sub["AvailableCount"] >= 3].copy()
        else:
            valid_sub = sub.copy()
        if valid_sub.empty:
            continue
        start_year = int(valid_sub["Year"].min())
        plot_df = sub[sub["Year"] >= start_year].copy()
        used_norm_cols = [c for c in norm_cols if plot_df[c].notna().any()]
        if not used_norm_cols:
            continue
        plt.figure(figsize=(12, 7))
        for nc in used_norm_cols:
            series = plot_df[["Year", nc]].dropna()
            if series.empty:
                continue
            label = nc.replace("_norm", "")
            plt.plot(series["Year"], series[nc], label=label)
        plt.title(f"{country} — Metric Standings (start at ≥3 metrics)")
        plt.xlabel("Year")
        plt.ylabel("Normalized Standing (0–1)")
        plt.ylim(0, 1)
        xmin, xmax = int(plot_df["Year"].min()), int(plot_df["Year"].max())
        plt.xlim(xmin - 1, xmax + 1)
        plt.legend(ncol=2, fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        label = ord_names[idx] if idx < len(ord_names) else f"rank_{idx+1}"
        fname = out_dir / f"{label}.png"
        plt.savefig(fname, dpi=150)
        plt.close()


__all__ = ["plot_top", "plot_country_metric_breakdowns"]


def plot_fixed_countries(composite: pd.DataFrame, countries: List[str], out_path: Path) -> None:
    plt.figure(figsize=(12, 7))
    for name in countries:
        s = composite[composite["Country"] == name].copy()
        if s.empty:
            continue
        s = s.sort_values("Year")
        # Clamp to start at 1950
        s = s[pd.to_numeric(s["Year"], errors="coerce") >= 1950]
        if "AvailableCount" in s.columns:
            s = s[s["AvailableCount"] >= 3]
        if s.empty:
            continue
        plt.plot(s["Year"], s["CompositeStanding"], label=name)
    plt.title("Empire Composite Standing — Selected Countries (dynamic start)")
    plt.xlabel("Year")
    plt.ylabel("Normalized Composite Standing (0–1)")
    years = pd.to_numeric(composite["Year"], errors="coerce").dropna()
    if not years.empty:
        xmin, xmax = max(1950, int(years.min())), int(years.max())
        plt.xlim(xmin, xmax)
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
