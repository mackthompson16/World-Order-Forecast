"""
Forecast utilities: produce per-country projected series to 2054 using a trained model.

Generates projection tuples suitable for plotting functions in plotting.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from .train_forecast import (
    ForecastTCN,
    make_country_windows,
    METRIC_COLS,  # K channels used in training (includes INDEX)
)


def _load_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "ISO3" in df.columns:
        df["ISO3"] = df["ISO3"].astype(str).str.upper()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.sort_values(["ISO3", "year"]).reset_index(drop=True)
    return df


@dataclass
class Projection:
    iso: str
    end_year: int
    years_hist: np.ndarray
    vals_hist: np.ndarray
    years_fore: np.ndarray
    vals_fore: np.ndarray


def build_country_projection(
    model: ForecastTCN,
    metrics_df: pd.DataFrame,
    iso: str,
    focus_metric: str = "INDEX",
    window: int = 50,
    horizon: int = 30,
    max_year: int = 2024,
    require_2024: bool = True,
) -> Optional[Projection]:
    """
    Build projection for one ISO3 code.

    - Uses rows where at least 4 component metrics are present to determine eligibility.
    - Prefers the window that ends exactly at max_year; if not found and require_2024=False,
      falls back to the last window <= max_year.
    - Offsets forecast to branch from last actual at end_year for continuity.
    """
    use_cols = [c for c in METRIC_COLS if c in metrics_df.columns]
    df = metrics_df[metrics_df["ISO3"] == iso].copy().sort_values("year")
    if df.empty or len(df) < window:
        return None

    # eligibility by composite rule: >= 4 component metrics present that year
    df["metric_count"] = df[use_cols].notna().sum(axis=1)

    # historical values up to max_year with eligibility
    hist = df[(df["year"] <= max_year) & (df["metric_count"] >= 4)].copy()
    if hist.empty:
        return None

    # Build an input window ending at max_year (or fallback <= max_year if allowed),
    # without requiring future targets.
    years = df["year"].to_numpy()
    # eligible end rows are where year <= max_year
    end_indices = np.where(years <= max_year)[0]
    if end_indices.size == 0:
        return None
    end_idx = int(end_indices[-1])  # last row <= max_year
    end_year = int(years[end_idx])
    if require_2024 and end_year != max_year:
        return None
    start_idx = end_idx - window + 1
    if start_idx < 0:
        return None
    win_df = df.iloc[start_idx:end_idx+1].copy()
    # Inputs: zero-fill NaNs, build mask
    K = len(METRIC_COLS)
    x_arr = np.zeros((1, window, K), dtype=np.float32)
    m_arr = np.zeros((1, window, K), dtype=bool)
    for k, col in enumerate(METRIC_COLS):
        vals = pd.to_numeric(win_df.get(col), errors="coerce").to_numpy()
        mask = ~np.isnan(vals)
        vals = np.nan_to_num(vals, nan=0.0)
        x_arr[0, :, k] = vals.astype(np.float32)
        m_arr[0, :, k] = mask
    geo = pd.to_numeric(win_df.get("geography_index"), errors="coerce").to_numpy()
    geo_last = float(geo[~np.isnan(geo)][-1]) if (~np.isnan(geo)).any() else 0.0

    device = next(model.parameters()).device
    with torch.no_grad():
        xb = torch.tensor(x_arr, dtype=torch.float32, device=device)
        mb = torch.tensor(m_arr, dtype=torch.bool, device=device)
        gb = torch.tensor([geo_last], dtype=torch.float32, device=device)
        pred = model(xb, mb, gb).cpu().numpy()[0]
    Kp = pred.shape[1]
    metric_to_idx = {m: i for i, m in enumerate(METRIC_COLS[:Kp])}
    fidx = metric_to_idx.get(focus_metric, Kp - 1)
    pred_series = pred[:, fidx]

    # continuity: last actual at end_year
    last_actual_row = hist[hist["year"] == end_year]
    if not last_actual_row.empty and focus_metric in hist.columns and pd.notna(last_actual_row.iloc[-1][focus_metric]):
        last_actual = float(last_actual_row.iloc[-1][focus_metric])
        offset = last_actual - float(pred_series[0])
        pred_series = pred_series + offset

    years_hist = hist["year"].to_numpy()
    vals_hist = hist[focus_metric].to_numpy()
    years_fore = np.arange(end_year + 1, end_year + horizon + 1)
    return Projection(
        iso=iso,
        end_year=end_year,
        years_hist=years_hist,
        vals_hist=vals_hist,
        years_fore=years_fore,
        vals_fore=pred_series,
    )


def build_projections_for_isos(
    ckpt_path: Path,
    metrics_csv: Path = Path("data/results/metrics.csv"),
    isos: Optional[List[str]] = None,
    focus_metric: str = "INDEX",
    window: int = 50,
    horizon: int = 30,
    max_year: int = 2024,
    require_2024: bool = True,
) -> Dict[str, Projection]:
    """Load model checkpoint and produce projections for selected isos."""
    state = torch.load(ckpt_path, map_location="cpu")
    K = int(state.get("in_channels", len(METRIC_COLS)))
    horizon_ckpt = int(state.get("horizon", horizon))
    cfg = state.get("config", {})
    model = ForecastTCN(
        in_channels=K,
        horizon=horizon_ckpt,
        use_gaussian_fade=cfg.get("use_gaussian_fade", True),
        gaussian_sigma_frac=cfg.get("gaussian_sigma_frac", 0.25),
    )
    model.load_state_dict(state["model_state"])
    model.eval()

    df = _load_metrics(metrics_csv)
    if isos is None:
        isos = sorted(df["ISO3"].dropna().unique().tolist())
    out: Dict[str, Projection] = {}
    for iso in isos:
        proj = build_country_projection(
            model=model,
            metrics_df=df,
            iso=iso,
            focus_metric=focus_metric,
            window=window,
            horizon=horizon_ckpt,
            max_year=max_year,
            require_2024=require_2024,
        )
        if proj is not None:
            out[iso] = proj
    return out


if __name__ == "__main__":
    # Minimal CLI: prefer latest run checkpoint; fallback to data/results/forecast_tcn.pt
    run_ckpts = sorted(Path("data/results").glob("run_*/forecast_tcn_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if run_ckpts:
        ckpt = run_ckpts[0]
    else:
        ckpt = Path("data/results/forecast_tcn.pt")
        if not ckpt.exists():
            raise SystemExit("No checkpoint found in data/results.")
    # Build all projections; highlight GBR/USA/CHN in the plot
    projs = build_projections_for_isos(ckpt, isos=None, require_2024=False)
    # Auto-plot
    try:
        from .plotting import plot_projection_2054
        out = plot_projection_2054(projs, Path("build_world_order/results/projection_2054.png"), highlight_iso=["USA","CHN","GBR"], smooth=5)
        print({k: {"end_year": v.end_year, "hist": len(v.years_hist), "fore": len(v.years_fore)} for k, v in projs.items()})
        print("wrote", out)
    except Exception as e:
        print("built projections only", {k: v.end_year for k,v in projs.items()}, "error plotting:", e)
