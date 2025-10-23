from pathlib import Path
import argparse

from .clean_data import build_clean_data, interpolate_chat_inplace
from .metrics import write_metrics
from .plotting import plot_composite, plot_metrics_grid, plot_top25_country_grids


def run(data_dir: Path, out_dir: Path, smooth: int = 5, rebuild_clean: bool = False):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build clean data and ensure CHAT is mapped and interpolated in-place
    clean_path = build_clean_data(data_dir, overwrite=rebuild_clean)
    chat_path = interpolate_chat_inplace(data_dir)

    # Metrics
    metrics_path = out_dir / "metrics.csv"
    metrics_path = write_metrics(clean_path, chat_path, metrics_path)

    # Plots (use INDEX from metrics.csv for composite)
    import pandas as pd
    metrics = pd.read_csv(metrics_path)
    comp_fig = plot_composite(metrics[[c for c in ["country_name","ISO3","year","INDEX"] if c in metrics.columns]], out_dir, smooth=smooth)
    grid_fig = plot_metrics_grid(metrics, out_dir, smooth=smooth)
    top25_dir = plot_top25_country_grids(metrics, out_dir, smooth=smooth)

    print(f"Metrics: {metrics_path}")
    print(f"Saved plots: {comp_fig}, {grid_fig}")
    print(f"Saved top-25 grids in: {top25_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run world order pipeline")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with input CSVs")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing window for plots")
    parser.add_argument("--rebuild-clean", action="store_true", help="Force rebuilding clean_data.csv")
    args = parser.parse_args()
    run(Path(args.data_dir), Path(args.out_dir), smooth=args.smooth, rebuild_clean=args.rebuild_clean)


if __name__ == "__main__":
    main()
