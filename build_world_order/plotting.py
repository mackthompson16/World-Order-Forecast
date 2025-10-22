from typing import List, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dirs, rolling_smooth, canonical_country


TARGET_COUNTRIES = [
    "RUSSIA",
    "GERMANY",
    "CHINA",
    "USA",
    "FRANCE",
    "NETHERLANDS",
    "INDIA",
    "UNITED KINGDOM",
]


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
        "Education", "Military", "EconomicIndex", "TradeShare", "ReserveCurrency", "FinancialCenter", "Innovation", "Competitiveness"
    ] if c in df.columns]

    precomputed = "WorldOrderIndex" in df.columns
    if precomputed:
        df["Composite"] = df["WorldOrderIndex"]
    # Create aliases to match requested weighting schema
    if "Innovation" in df.columns:
        df["Technology"] = df["Innovation"]
    if "EconomicIndex" in df.columns:
        df["EconomicOutput"] = df["EconomicIndex"]

    # Weighted composite with renormalization over available metrics
    weights = {
        "Education": 0.15,
        "Competitiveness": 0.15,
        "Technology": 0.15,
        "EconomicOutput": 0.15,
        "TradeShare": 0.10,
        "Military": 0.10,
        "FinancialCenter": 0.10,
        "ReserveCurrency": 0.10,
    }
    # Select available columns from the weighting schema
    available_cols = [k for k in weights.keys() if k in df.columns]
    # Weighted sum
    value_mat = df[available_cols]
    weight_vec = pd.Series({k: weights[k] for k in available_cols})
    weighted_sum = (value_mat * weight_vec).sum(axis=1, skipna=True)
    # Sum of weights present (non-null per row)
    present_weights = value_mat.notna().astype(float) * weight_vec
    present_weights = present_weights.sum(axis=1)
    if not precomputed:
        df["Composite"] = weighted_sum.divide(present_weights).where(present_weights > 0)

    df = _filter_countries(df, countries)
    # Start from requested year threshold
    df = df[df["year"] >= start_year]
    if end_year is not None:
        df = df[df["year"] <= end_year]

    plt.figure(figsize=(10, 6))

    # Emphasize these countries (bold lines)
    emphasize = {canonical_country(c) for c in ["CHINA", "UNITED KINGDOM", "USA", "GERMANY"]}

    for country, sub in df.groupby("country"):
        sub = sub.sort_values("year")
        ys = rolling_smooth(sub["Composite"], window=smooth_window)
        if country in emphasize:
            plt.plot(
                sub["year"], ys,
                label=country,
                linewidth=3.0,
                linestyle='-',
                alpha=0.95,
                zorder=3,
            )
        else:
            plt.plot(
                sub["year"], ys,
                label=country,
                linewidth=1.0,
                linestyle='--',
                alpha=0.8,
                zorder=2,
            )

    # Shade notable global periods and label them
    ax = plt.gca()
    periods = [
        (1914, 1918, "WW1"),
        (1939, 1945, "WW2"),
    ]
    for start, end, label in periods:
        ax.axvspan(start, end, color="grey", alpha=0.15, zorder=1)
        xmid = (start + end) / 2.0
        ymin, ymax = ax.get_ylim()
        y = ymin + 0.94 * (ymax - ymin)
        ax.text(
            xmid,
            y,
            label,
            ha="center",
            va="top",
            fontsize=8,
            color="dimgray",
            zorder=4,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
        )

    plt.title("World Order Composite Standing (Smoothed)")
    plt.xlabel("Year")
    plt.ylabel("Composite Score (0-1)")
    leg = plt.legend(ncol=2, fontsize=8)
    # Bold legend labels for emphasized countries
    for txt in leg.get_texts():
        if canonical_country(txt.get_text()) in emphasize:
            txt.set_fontweight('bold')
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
        "Education", "Military", "EconomicIndex", "TradeShare", "ReserveCurrency", "FinancialCenter", "Innovation", "Competitiveness"
    ]
    df = _filter_countries(metrics_df, countries)
    df = df[df["year"] >= start_year]

    for country, sub in df.groupby("country"):
        sub = sub.sort_values("year")
        # Dynamic grid size based on metric count
        n = len(metric_cols)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), sharex=True)
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
