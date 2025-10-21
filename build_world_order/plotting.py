from typing import List, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dirs, rolling_smooth, canonical_country


TARGET_COUNTRIES = ["RUSSIA", "GERMANY", "CHINA", "USA", "FRANCE", "NETHERLANDS", "INDIA"]


def _filter_countries(df: pd.DataFrame, countries: Sequence[str]) -> pd.DataFrame:
    keys = [canonical_country(c) for c in countries]
    return df[df["country"].isin(keys)].copy()


def plot_world_order_composite(
    metrics_df: pd.DataFrame,
    out_dir: str,
    smooth_window: int = 5,
    countries: Sequence[str] = TARGET_COUNTRIES,
    start_year: int = 1800,
    end_year: int | None = None,
) -> str:
    ensure_dirs(out_dir)
    df = metrics_df.copy()
    metric_cols = [c for c in [
        "Education", "Military", "EconomicIndex", "TradeShare", "ReserveCurrency", "FinancialCenter"
    ] if c in df.columns]

    df["Composite"] = df[metric_cols].mean(axis=1, skipna=True)
    df = _filter_countries(df, countries)
    # Start from requested year threshold
    df = df[df["year"] >= start_year]
    if end_year is not None:
        df = df[df["year"] <= end_year]

    plt.figure(figsize=(10, 6))
    for country, sub in df.groupby("country"):
        sub = sub.sort_values("year")
        ys = rolling_smooth(sub["Composite"], window=smooth_window)
        plt.plot(sub["year"], ys, label=country)

    plt.title("World Order Composite Standing (Smoothed)")
    plt.xlabel("Year")
    plt.ylabel("Composite Score (0-1)")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    out_path = f"{out_dir}/World_Order_Graph.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_raw_metric_diagnostics(metrics_df: pd.DataFrame, out_dir: str, countries: Sequence[str] = TARGET_COUNTRIES, smooth_window: int = 5, start_year: int = 1800) -> None:
    out_root = f"{out_dir}/raw_metrics"
    ensure_dirs(out_root)

    metric_cols = [
        "Education", "Military", "EconomicIndex", "TradeShare", "ReserveCurrency", "FinancialCenter"
    ]
    df = _filter_countries(metrics_df, countries)
    df = df[df["year"] >= start_year]

    for country, sub in df.groupby("country"):
        sub = sub.sort_values("year")
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
        axes = axes.ravel()
        for i, m in enumerate(metric_cols):
            ax = axes[i]
            if m in sub.columns:
                # Smooth each metric series before plotting
                y = rolling_smooth(sub[m], window=smooth_window)
                ax.plot(sub["year"], y, color="tab:blue")
                ax.set_title(m)
                ax.grid(True, alpha=0.3)
            else:
                ax.set_title(m)
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.grid(True, alpha=0.3)
        for j in range(len(metric_cols), len(axes)):
            axes[j].axis('off')
        fig.suptitle(f"Raw Metric Diagnostics — {country}")
        fig.supxlabel("Year")
        fig.supylabel("Score (0-1)")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = f"{out_root}/{country}_raw_metrics.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
