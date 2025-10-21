import os
import sys
from typing import List

import pandas as pd

from .loaders import load_education, load_military, load_gmd
from .metrics import compute_metrics
from .plotting import plot_world_order_composite, plot_raw_metric_diagnostics, TARGET_COUNTRIES
from .diagnostics import compute_data_coverage, plot_data_coverage
from .utils import ensure_dirs


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run(selected_countries: List[str] = None) -> None:
    if selected_countries is None:
        selected_countries = TARGET_COUNTRIES

    edu_path = os.path.join(DATA_DIR, "Education.csv")
    mil_path = os.path.join(DATA_DIR, "military.csv")
    gmd_path = os.path.join(DATA_DIR, "GMD.csv")

    # Load
    edu = load_education(edu_path)
    mil = load_military(mil_path)
    gmd = load_gmd(gmd_path)

    # Compute metrics
    metrics_df = compute_metrics(edu, mil, gmd)

    # Save intermediate (optional, uncomment if needed)
    # ensure_dirs(RESULTS_DIR)
    # metrics_df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)

    # Plots
    ensure_dirs(RESULTS_DIR)
    composite_path = plot_world_order_composite(
        metrics_df,
        RESULTS_DIR,
        smooth_window=11,
        countries=selected_countries,
        start_year=1800,
        end_year=2024,
    )
    plot_raw_metric_diagnostics(metrics_df, RESULTS_DIR, countries=selected_countries)

    # Coverage timeline for inputs used by metrics (raw availability counts)
    coverage_df = compute_data_coverage(edu_path, mil_path, gmd_path)
    coverage_path = plot_data_coverage(coverage_df, RESULTS_DIR, start_year=1800, end_year=2024)

    print(f"Saved composite graph to: {composite_path}")
    print(f"Raw metric diagnostics saved under: {os.path.join(RESULTS_DIR, 'raw_metrics')}")
    print(f"Saved data coverage timeline to: {coverage_path}")


if __name__ == "__main__":
    run()
