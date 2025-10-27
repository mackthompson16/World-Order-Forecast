from pathlib import Path
import argparse

from .metrics import write_metrics
from .plotting import plot_composite, plot_metrics_grid, plot_top25_country_grids, plot_geography_index
from .geography import write_clean_geography


def run(data_dir: Path | None = None, out_dir: Path | None = None, smooth: int = 5, rebuild_clean: bool = False):
    # Defaults: data under ./data, results under build_world_order/results
    data_dir = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[1] / "data"
    out_dir = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use prebuilt clean data and CHAT; do not rebuild here
    clean_path = data_dir / "clean_data.csv"
    chat_path = data_dir / "CHAT.csv"
    if not clean_path.exists():
        raise FileNotFoundError(f"Missing {clean_path}. Run: python -m build_world_order.clean_data --data-dir {data_dir}")
    if not chat_path.exists():
        raise FileNotFoundError(f"Missing {chat_path}. Ensure CHAT.csv is present in the data directory.")

    # Metrics
    metrics_path = out_dir / "metrics.csv"
    metrics_path = write_metrics(clean_path, chat_path, metrics_path)

    # Plots (use INDEX from metrics.csv for composite)
    import pandas as pd
    metrics = pd.read_csv(metrics_path)
    comp_fig = plot_composite(metrics[[c for c in ["country_name","ISO3","year","INDEX"] if c in metrics.columns]], out_dir, smooth=smooth)
    grid_fig = plot_metrics_grid(metrics, out_dir, smooth=smooth)
    top25_dir = plot_top25_country_grids(metrics, out_dir, smooth=smooth)

    # Geography (optional): if data/geography_data exists, build clean_geography.csv
    geo_dir = data_dir / "geography_data"
    if not geo_dir.exists():
        # Also accept geography_Data (case variants)
        alt = data_dir / "geography_Data"
        if alt.exists():
            geo_dir = alt
    if geo_dir.exists():
        geo_path = write_clean_geography(geo_dir, data_dir)
        if geo_path:
            print(f"Geography: {geo_path}")
            try:
                import pandas as pd
                geo_df = pd.read_csv(geo_path)
                geo_fig = plot_geography_index(geo_df, out_dir, smooth=smooth)
                print(f"Saved geography plot: {geo_fig}")
            except Exception as e:
                print(f"Geography plot failed: {e}")

    print(f"Metrics: {metrics_path}")
    print(f"Saved plots: {comp_fig}, {grid_fig}")
    print(f"Saved top-25 grids in: {top25_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run world order pipeline")
    parser.add_argument("--data-dir", type=str, required=False, help="Directory with input CSVs (default: ./data)")
    parser.add_argument("--out-dir", type=str, required=False, help="Output directory for results (default: build_world_order/results)")
    parser.add_argument("--smooth", type=int, default=5, help="Smoothing window for plots")
    parser.add_argument("--rebuild-clean", action="store_true", help="(deprecated) Cleaning runs only via clean_data module")
    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else None
    out_dir = Path(args.out_dir) if args.out_dir else None
    run(data_dir, out_dir, smooth=args.smooth, rebuild_clean=args.rebuild_clean)


if __name__ == "__main__":
    main()
