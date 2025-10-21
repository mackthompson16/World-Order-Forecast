from pathlib import Path
from typing import Optional

import pandas as pd

# Import via modularized pipeline
from src.pipeline.panel import collect_panel, summarize_schema, pivot_metrics
from src.pipeline.coverage import compute_coverage_from_wide, plot_coverage_combined
from src.pipeline.normalize import compute_composite
from src.pipeline.winners import (
    select_top5_by_area,
    compute_winners_by_metric_year,
    compute_top5_by_year,
)
from src.pipeline.plots import plot_top, plot_country_metric_breakdowns, plot_fixed_countries


def run(data_dir: Optional[str] = None) -> None:
    data_root = Path(data_dir or Path("data")).resolve()
    results_dir = Path("results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] (orchestrator) Data root: {data_root}")
    panel, problems = collect_panel(data_root)
    if panel.empty:
        print("[INFO] (orchestrator) No data parsed; exiting.")
        if problems:
            pd.DataFrame(problems, columns=["file", "note"]).to_csv(results_dir / "parsing_warnings.csv", index=False)
        return

    summarize_schema(panel).to_csv(results_dir / "parsed_schema_summary.csv", index=False)
    panel[["Country"]].drop_duplicates().sort_values("Country").to_csv(results_dir / "countries_list.csv", index=False)

    wide = pivot_metrics(panel)
    coverage = compute_coverage_from_wide(wide)
    coverage.to_csv(results_dir / "data_coverage_by_metric_year.csv", index=False)
    plot_coverage_combined(coverage, Path("coverage_all_metrics_counts.png"), logy=True)

    composite = compute_composite(wide)
    composite.to_csv(results_dir / "empire_composite.csv", index=False)

    # Build Top 5 ranking CSV for reference (Coverage then Area)
    top_series, areas = select_top5_by_area(composite)
    areas.to_csv(results_dir / "country_area_ranking.csv", index=False)
    # Plot fixed countries instead of Top 5: US, UK, China, Russia, India
    fixed_countries = [
        "United States",
        "United Kingdom",
        "China",
        "Russia",
        "India",
    ]
    plot_fixed_countries(composite, fixed_countries, Path("empire_standings_top5.png"))

    winners_by_metric = compute_winners_by_metric_year(composite)
    winners_by_metric.to_csv(results_dir / "winners_by_metric_year.csv", index=False)
    top5_by_year = compute_top5_by_year(composite)
    top5_by_year.to_csv(results_dir / "top5_by_year.csv", index=False)

    winners_dir = results_dir / "winners"
    winners_dir.mkdir(parents=True, exist_ok=True)
    # Use the same fixed set for winners images, in this order
    fixed_countries = [
        "United States",
        "United Kingdom",
        "China",
        "Russia",
        "India",
    ]
    plot_country_metric_breakdowns(composite, fixed_countries, winners_dir)

    if problems:
        pd.DataFrame(problems, columns=["file", "note"]).to_csv(results_dir / "parsing_warnings.csv", index=False)
        print("[WARN] (orchestrator) Parsing issues detected. See results/parsing_warnings.csv")

    print("[INFO] (orchestrator) Done. CSVs in results/, images at repo root.")
