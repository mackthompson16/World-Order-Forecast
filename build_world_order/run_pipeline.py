from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data_loading import load_all
from .compute_metrics import build_metrics, compute_composite
from .utils import SELECTED_COUNTRIES
from .normalize import fill_gaps
from .plotting import plot_composite, plot_raw_metrics


def main(data_dir: Path, results_dir: Path, smooth_window: int = 5):
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = results_dir / "raw_metrics"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dfs, grid = load_all(data_dir)
    panel = build_metrics(dfs, grid)

    # Smooth with centered rolling window for all metric columns
    metric_cols = [
        "education",
        "military",
        "economic_index",
        "trade_share",
        "reserve_currency",
        "financial_center",
        "innovation",
        "competitiveness",
    ]

    panel = panel.sort_values(["country", "year"]).copy()
    for c in metric_cols:
        # Centered rolling mean; preserve NaNs at edges
        panel[c] = panel.groupby("country")[c].transform(lambda s: s.rolling(smooth_window, center=True, min_periods=1).mean())
        # Re-apply inside-only interpolation to avoid extended edges
        panel[c] = panel.groupby("country")[c].transform(fill_gaps)

    # Compute composite with renormalized weights per-row
    panel = compute_composite(panel)

    # Filter selected countries and year bounds are already enforced by loaders
    selected = panel[panel["country"].isin(SELECTED_COUNTRIES)].copy()

    # Persist CSV per spec
    metrics_csv = results_dir / "metrics.csv"
    selected.to_csv(metrics_csv, index=False)

    # Also save full panel for reference
    full_csv = results_dir / "full_metrics.csv"
    panel.to_csv(full_csv, index=False)

    # Plots
    composite_png = results_dir / "world_order_index.png"
    plot_composite(selected, SELECTED_COUNTRIES, composite_png)

    for country in SELECTED_COUNTRIES:
        out = results_dir / f"raw_metrics_{country.replace(' ', '_')}.png"
        plot_raw_metrics(selected, country, out)

    print(f"Wrote: {metrics_csv}")
    print(f"Wrote: {full_csv}")
    print(f"Wrote figures to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build World Order metrics and plots")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing input CSVs")
    parser.add_argument("--out-dir", type=Path, default=Path("build_world_order") / "results", help="Output directory")
    parser.add_argument("--smooth", type=int, default=5, help="Centered rolling window for smoothing (years)")
    args = parser.parse_args()

    main(args.data_dir, args.out_dir, smooth_window=args.smooth)

