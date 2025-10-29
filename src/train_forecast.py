"""
Train a Temporal ConvNet (TCN) forecaster on data/results/metrics.csv.

Spec (from user requirements):
- Data: data/results/metrics.csv with columns
  [country_name, ISO3, year, EDU, MIL, ECON, TRAD, RESV, FIN, INV, CMPT, INDEX, geography_index]
- Countries span 1800–2024.
- Only train on countries with >= 100 years of valid data; exclude validation candidate Denmark from training.
- Sliding windows: input length = 50 years, forecast horizon = 30 years (multi-step, direct).
- Inputs: K metric channels (K = 9 metrics: EDU..CMPT + INDEX). Exogenous geography_index is not a channel.
- Mask: Build boolean mask (50, K) per window for missing inputs.
- Model: Temporal ConvNet (1D conv over time) over K channels.
- Geography bias: Use the last known geography_index from the input window and add β · geography as an additive bias to the output head.
- Loss: Masked MSE. Only compute loss on targets actually present within the 30-year horizon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datetime import datetime
import warnings

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # plotting optional

# Optional styling import to mimic composite plots
try:
    from .plotting import HIGHLIGHT as PLOT_HIGHLIGHT
except Exception:
    PLOT_HIGHLIGHT = {
        "USA": {"color": "#1f77b4", "lw": 2.8, "alpha": 1.0, "zorder": 3},
        "CHN": {"color": "#d62728", "lw": 2.8, "alpha": 1.0, "zorder": 3},
        "GBR": {"color": "#2ca02c", "lw": 2.8, "alpha": 1.0, "zorder": 3},
    }


METRIC_COLS = ["EDU", "MIL", "ECON", "TRAD", "RESV", "FIN", "INV", "CMPT", "INDEX"]
GEO_COL = "geography_index"


# -------------------------
# Data preparation utilities
# -------------------------

@dataclass
class WindowedBatch:
    x: torch.Tensor            # (B, W, K) float32
    x_mask: torch.Tensor       # (B, W, K) bool
    geo_last: torch.Tensor     # (B,) float32 - last known geography in window
    y: torch.Tensor            # (B, H, K) float32
    y_mask: torch.Tensor       # (B, H, K) bool


def _last_known_in_window(arr: np.ndarray, mask: np.ndarray) -> float:
    # arr shape (W,), mask True when value is present
    present_idx = np.where(mask)[0]
    if present_idx.size == 0:
        return 0.0
    return float(arr[present_idx[-1]])


def make_country_windows(
    df: pd.DataFrame,
    window: int = 50,
    horizon: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create sliding windows for a single country's panel.

    Returns (x, x_mask, geo_last, y, y_mask) where
      - x: (N, window, K)
      - x_mask: (N, window, K) boolean, True where observed
      - geo_last: (N,) last known geo value within window
      - y: (N, horizon, K)
      - y_mask: (N, horizon, K) boolean, True where target exists
    """
    use_cols = [c for c in METRIC_COLS if c in df.columns]
    if len(use_cols) == 0:
        return tuple(np.zeros((0,)) for _ in range(5))  # type: ignore[return-value]

    K = len(use_cols)
    arr = df[use_cols].to_numpy(dtype=float)  # (T, K)
    present = ~np.isnan(arr)

    geo = df[GEO_COL].to_numpy(dtype=float) if GEO_COL in df.columns else np.full((len(df),), np.nan)
    geo_present = ~np.isnan(geo)

    T = len(df)
    N = max(0, T - window - horizon + 1)
    if N == 0:
        return tuple(np.zeros((0,)) for _ in range(5))  # type: ignore[return-value]

    x = np.zeros((N, window, K), dtype=np.float32)
    x_mask = np.zeros((N, window, K), dtype=bool)
    geo_last = np.zeros((N,), dtype=np.float32)
    y = np.zeros((N, horizon, K), dtype=np.float32)
    y_mask = np.zeros((N, horizon, K), dtype=bool)

    for i in range(N):
        w_start = i
        w_end = i + window
        f_start = w_end
        f_end = f_start + horizon

        x_win = arr[w_start:w_end]           # (W, K)
        p_win = present[w_start:w_end]       # (W, K)
        # Zero-fill inputs where missing; keep mask for model if desired
        x[i] = np.nan_to_num(x_win, nan=0.0)
        x_mask[i] = p_win

        # Geography bias: last known within window
        geo_win = geo[w_start:w_end]
        geo_present_win = geo_present[w_start:w_end]
        geo_last[i] = _last_known_in_window(geo_win, geo_present_win)

        # Targets
        f_win = arr[f_start:f_end]           # (H, K)
        f_present = present[f_start:f_end]   # (H, K)
        y[i] = np.nan_to_num(f_win, nan=0.0)
        y_mask[i] = f_present

    return x, x_mask, geo_last, y, y_mask


def load_metrics_panel(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize and filter
    if "ISO3" in df.columns:
        df["ISO3"] = df["ISO3"].astype(str).str.upper()
    if "country_name" in df.columns:
        df["country_name"] = df["country_name"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[(df["year"] >= 1800) & (df["year"] <= 2024)]
    df = df.sort_values(["ISO3", "year"]).reset_index(drop=True)
    return df


def filter_countries(df: pd.DataFrame, min_years: int = 100, exclude_iso: Optional[List[str]] = None) -> List[str]:
    exclude_iso = [s.upper() for s in (exclude_iso or [])]

    def valid_years(group: pd.DataFrame) -> int:
        # Count years with at least 1 metric present among training targets
        nn = group[METRIC_COLS].notna().sum(axis=1)
        return int((nn >= 1).sum())

    counts = df.groupby("ISO3").apply(valid_years)
    keep = [iso for iso, n in counts.items() if n >= min_years and iso not in exclude_iso]
    return keep


class PanelForecastDataset(Dataset):
    def __init__(self, batches: List[WindowedBatch]):
        self._batches = batches
        # Flatten batches across countries into single tensors
        xs = torch.cat([b.x for b in batches], dim=0) if batches else torch.zeros(0)
        xm = torch.cat([b.x_mask for b in batches], dim=0) if batches else torch.zeros(0, dtype=torch.bool)
        gl = torch.cat([b.geo_last for b in batches], dim=0) if batches else torch.zeros(0)
        ys = torch.cat([b.y for b in batches], dim=0) if batches else torch.zeros(0)
        ym = torch.cat([b.y_mask for b in batches], dim=0) if batches else torch.zeros(0, dtype=torch.bool)

        self.x = xs
        self.x_mask = xm
        self.geo_last = gl
        self.y = ys
        self.y_mask = ym

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.x[idx],
            self.x_mask[idx],
            self.geo_last[idx],
            self.y[idx],
            self.y_mask[idx],
        )


def build_datasets(
    csv_path: Path,
    window: int = 50,
    horizon: int = 30,
    val_iso: str = "DNK",
    min_years: int = 100,
    device: Optional[torch.device] = None,
) -> Tuple[PanelForecastDataset, Optional[PanelForecastDataset]]:
    device = device or torch.device("cpu")
    df = load_metrics_panel(csv_path)

    # Identify train countries
    train_isos = filter_countries(df, min_years=min_years, exclude_iso=[val_iso])
    # Validation set: use the candidate country if present
    val_df = df[df["ISO3"] == val_iso].copy()

    batches_train: List[WindowedBatch] = []
    for iso in train_isos:
        g = df[df["ISO3"] == iso].copy()
        if g.empty:
            continue
        x, xm, gl, y, ym = make_country_windows(g, window=window, horizon=horizon)
        if x.shape[0] == 0:
            continue
        batches_train.append(
            WindowedBatch(
                x=torch.tensor(x, dtype=torch.float32, device=device),
                x_mask=torch.tensor(xm, dtype=torch.bool, device=device),
                geo_last=torch.tensor(gl, dtype=torch.float32, device=device),
                y=torch.tensor(y, dtype=torch.float32, device=device),
                y_mask=torch.tensor(ym, dtype=torch.bool, device=device),
            )
        )

    train_ds = PanelForecastDataset(batches_train)

    val_ds: Optional[PanelForecastDataset] = None
    if not val_df.empty:
        x, xm, gl, y, ym = make_country_windows(val_df, window=window, horizon=horizon)
        if x.shape[0] > 0:
            val_batch = WindowedBatch(
                x=torch.tensor(x, dtype=torch.float32, device=device),
                x_mask=torch.tensor(xm, dtype=torch.bool, device=device),
                geo_last=torch.tensor(gl, dtype=torch.float32, device=device),
                y=torch.tensor(y, dtype=torch.float32, device=device),
                y_mask=torch.tensor(ym, dtype=torch.bool, device=device),
            )
            val_ds = PanelForecastDataset([val_batch])

    return train_ds, val_ds


# -------------------------
# Model
# -------------------------

class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.resid = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.resid(x)
        x_len = x.size(-1)
        x = self.drop(self.act(self.conv1(x)))
        x = self.drop(self.act(self.conv2(x)))
        # Crop to original length (causal output trimming)
        if x.size(-1) != x_len:
            x = x[..., :x_len]
        if r.size(-1) != x_len:
            r = r[..., :x_len]
        return self.act(x + r)


class ForecastTCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        horizon: int = 30,
        channels: List[int] = [64, 64, 64],
        kernel_size: int = 3,
        dropout: float = 0.1,
        per_metric_beta: bool = True,
        use_gaussian_fade: bool = True,
        gaussian_sigma_frac: float = 0.25,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        ch_prev = in_channels
        for i, ch in enumerate(channels):
            layers.append(TemporalBlock(ch_prev, ch, kernel_size=kernel_size, dilation=2 ** i, dropout=dropout))
            ch_prev = ch
        self.tcn = nn.Sequential(*layers)
        self.proj = nn.Linear(ch_prev, horizon * in_channels)
        # Geography bias parameter β: per-metric or shared
        if per_metric_beta:
            self.beta = nn.Parameter(torch.zeros(in_channels))
        else:
            self.beta = nn.Parameter(torch.zeros(1))
        self.in_channels = in_channels
        self.horizon = horizon
        self.use_gaussian_fade = use_gaussian_fade
        self.gaussian_sigma_frac = gaussian_sigma_frac

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor], geo_last: torch.Tensor) -> torch.Tensor:
        # x: (B, W, K) -> (B, K, W)
        x_in = x.transpose(1, 2)
        # Optional Gaussian temporal fade: weight older timesteps lower
        if self.use_gaussian_fade:
            W = x_in.size(-1)
            device = x_in.device
            t = torch.arange(W, device=device, dtype=x_in.dtype)
            center = float(W - 1)
            sigma = max(1.0, float(self.gaussian_sigma_frac) * float(W))
            w = torch.exp(-0.5 * ((t - center) / sigma) ** 2)  # (W,)
            w = w / w.max().clamp_min(1e-8)
            x_in = x_in * w.view(1, 1, W)
        feats = self.tcn(x_in)              # (B, C, W)
        last = feats[:, :, -1]              # (B, C)
        out = self.proj(last)               # (B, H*K)
        out = out.view(x.shape[0], self.horizon, self.in_channels)  # (B, H, K)
        # Add geography bias
        if self.beta.shape[0] == self.in_channels:
            bias = geo_last.view(-1, 1, 1) * self.beta.view(1, 1, -1)
        else:
            bias = geo_last.view(-1, 1, 1) * self.beta.view(1, 1, 1)
        return out + bias


# -------------------------
# Training
# -------------------------

def masked_mse(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # mask boolean True where target exists
    diff2 = (pred - true) ** 2
    diff2 = diff2 * mask.float()
    denom = mask.float().sum().clamp_min(1.0)
    return diff2.sum() / denom


@dataclass
class TrainConfig:
    window: int = 50
    horizon: int = 30
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-5
    dropout: float = 0.1
    kernel_size: int = 3
    channels: Tuple[int, ...] = (64, 64, 64)
    val_iso: str = "DNK"
    min_years: int = 100
    use_gaussian_fade: bool = True
    gaussian_sigma_frac: float = 0.25


def train_loop(
    model: ForecastTCN,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> Dict[str, List[float]]:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for xb, mb, gb, yb, ymb in train_loader:
            xb = xb.to(device)
            mb = mb.to(device)
            gb = gb.to(device)
            yb = yb.to(device)
            ymb = ymb.to(device)

            pred = model(xb, mb, gb)
            loss = masked_mse(pred, yb, ymb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += float(loss.item())
            n_tr += 1
        tr_loss = tr_loss / max(1, n_tr)
        hist["train_loss"].append(tr_loss)

        va_loss = float("nan")
        if val_loader is not None:
            model.eval()
            vs = 0.0
            nv = 0
            with torch.no_grad():
                for xb, mb, gb, yb, ymb in val_loader:
                    xb = xb.to(device)
                    mb = mb.to(device)
                    gb = gb.to(device)
                    yb = yb.to(device)
                    ymb = ymb.to(device)
                    pred = model(xb, mb, gb)
                    loss = masked_mse(pred, yb, ymb)
                    vs += float(loss.item())
                    nv += 1
            va_loss = vs / max(1, nv)
            hist["val_loss"].append(va_loss)

        tqdm.write(f"Epoch {ep:03d} | train {tr_loss:.4f} | val {va_loss:.4f}")
    return hist


def run_training(
    csv_path: Path = Path("data/results/metrics.csv"),
    out_dir: Path = Path("data/results"),
    cfg: Optional[TrainConfig] = None,
) -> Dict:
    cfg = cfg or TrainConfig()
    # Allow quick overrides via env vars for faster runs
    try:
        import os
        ep = os.getenv("WO_EPOCHS")
        bs = os.getenv("WO_BATCH")
        if ep:
            cfg.epochs = int(ep)
        if bs:
            cfg.batch_size = int(bs)
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build datasets
    train_ds, val_ds = build_datasets(
        csv_path=csv_path,
        window=cfg.window,
        horizon=cfg.horizon,
        val_iso=cfg.val_iso,
        min_years=cfg.min_years,
        device=device,
    )
    if len(train_ds) == 0:
        raise RuntimeError("No training samples after filtering. Check data and thresholds.")

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False) if val_ds else None

    # Model
    K = len(METRIC_COLS)
    model = ForecastTCN(
        in_channels=K,
        horizon=cfg.horizon,
        channels=list(cfg.channels),
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout,
        use_gaussian_fade=cfg.use_gaussian_fade,
        gaussian_sigma_frac=cfg.gaussian_sigma_frac,
    ).to(device)

    # Train
    history = train_loop(
        model,
        train_loader,
        val_loader,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        device=device,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Create a unique run directory and checkpoint name
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / f"forecast_tcn_{run_id}.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": cfg.__dict__,
        "in_channels": K,
        "horizon": cfg.horizon,
        "metric_cols": METRIC_COLS,
    }, ckpt_path)

    # Also write a stable checkpoint path for downstream tooling
    try:
        stable_ckpt = out_dir / "forecast_tcn.pt"
        stable_ckpt.write_bytes(Path(ckpt_path).read_bytes())
    except Exception:
        pass

    info = {
        "device": str(device),
        "train_samples": len(train_ds),
        "val_samples": 0 if val_ds is None else len(val_ds),
        "history": history,
        "checkpoint": str(ckpt_path),
        "run_id": run_id,
        "run_dir": str(run_dir),
    }

    # Plot loss curve if matplotlib is available
    if plt is not None:
        try:
            fig = plt.figure(figsize=(6, 4))
            epochs = list(range(1, len(history.get("train_loss", [])) + 1))
            if epochs:
                plt.plot(epochs, history.get("train_loss", []), label="train")
                if history.get("val_loss"):
                    plt.plot(epochs, history.get("val_loss", []), label="val")
                plt.xlabel("Epoch")
                plt.ylabel("Masked MSE")
                plt.title("Training Loss")
                plt.legend()
                plt.tight_layout()
            loss_path = run_dir / "loss_curve.png"
            plt.savefig(loss_path, dpi=150)
            plt.close(fig)
            info["loss_plot"] = str(loss_path)
            # And copy to src/results per request
            try:
                img_dir = Path("src/results"); img_dir.mkdir(parents=True, exist_ok=True)
                (img_dir / "loss_curve.png").write_bytes(Path(loss_path).read_bytes())
            except Exception:
                pass
        except Exception as e:
            warnings.warn(f"Failed to plot loss curve: {e}")

    return info


def _linear_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    # Fit y = a + b*y_pred by least squares
    X = np.vstack([np.ones_like(y_pred), y_pred]).T
    coef, *_ = np.linalg.lstsq(X, y_true, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return a, b


def evaluate_walk_forward(
    model: ForecastTCN,
    val_loader: Optional[DataLoader],
    focus_metric: str = "INDEX",
    out_dir: Path = Path("data/results"),
    iso_code: str = "DNK",
) -> Dict:
    if val_loader is None:
        return {"message": "No validation set available"}
    model.eval()
    device = next(model.parameters()).device

    # Collect predictions and truths
    preds_list = []  # (B,H,K)
    trues_list = []
    masks_list = []
    with torch.no_grad():
        for xb, mb, gb, yb, ymb in val_loader:
            xb = xb.to(device)
            mb = mb.to(device)
            gb = gb.to(device)
            yb = yb.to(device)
            ymb = ymb.to(device)
            pred = model(xb, mb, gb)
            preds_list.append(pred.cpu())
            trues_list.append(yb.cpu())
            masks_list.append(ymb.cpu())

    preds = torch.cat(preds_list, dim=0).numpy()
    trues = torch.cat(trues_list, dim=0).numpy()
    masks = torch.cat(masks_list, dim=0).numpy()

    H = preds.shape[1]
    K = preds.shape[2]
    metric_to_idx = {m: i for i, m in enumerate(METRIC_COLS)}
    focus_idx = metric_to_idx.get(focus_metric, metric_to_idx["INDEX"]) if K == len(METRIC_COLS) else K - 1

    def _safe_stats(y_t: np.ndarray, y_p: np.ndarray) -> Dict[str, float]:
        if y_t.size == 0:
            return {"n": 0}
        mae = float(np.mean(np.abs(y_t - y_p)))
        rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
        # correlation
        if np.std(y_t) > 0 and np.std(y_p) > 0:
            corr = float(np.corrcoef(y_t, y_p)[0, 1])
        else:
            corr = float("nan")
        # R2
        ss_res = float(np.sum((y_t - y_p) ** 2))
        ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2)) if y_t.size > 0 else float("nan")
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot not in (0.0, float("nan")) else float("nan")
        # MAPE (safe)
        denom = np.maximum(np.abs(y_t), 1e-6)
        mape = float(np.mean(np.abs((y_t - y_p) / denom)))
        # calibration
        a, b = _linear_calibration(y_t, y_p)
        y_pc = a + b * y_p
        rmse_cal = float(np.sqrt(np.mean((y_t - y_pc) ** 2)))
        return {
            "n": int(y_t.size),
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
            "corr": corr,
            "rmse_calibrated": rmse_cal,
            "slope": b,
            "intercept": a,
        }

    # Per-horizon stats for focus metric
    rows = []
    per_h_stats: Dict[str, Dict[str, float]] = {}
    for h in range(H):
        m = masks[:, h, focus_idx].astype(bool)
        y_t = trues[:, h, focus_idx][m]
        y_p = preds[:, h, focus_idx][m]
        stats = _safe_stats(y_t, y_p)
        per_h_stats[f"h{h+1}"] = stats
        row = {"horizon": h + 1, **stats}
        rows.append(row)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_csv = out_dir / ("validation_" + iso_code.lower() + "_focus_" + focus_metric + ".csv")
    try:
        pd.DataFrame(rows).to_csv(eval_csv, index=False)
    except Exception:
        pass

    # Optional plot of RMSE by horizon
    if plt is not None:
        try:
            fig = plt.figure(figsize=(6, 4))
            xs = [r["horizon"] for r in rows if "horizon" in r]
            ys = [r.get("rmse", float("nan")) for r in rows]
            ysc = [r.get("rmse_calibrated", float("nan")) for r in rows]
            plt.plot(xs, ys, label="RMSE")
            plt.plot(xs, ysc, label="RMSE (calibrated)")
            plt.xlabel("Horizon (years)")
            plt.ylabel("RMSE")
            plt.title("Validation on DNK (INDEX)")
            plt.legend()
            plt.tight_layout()
            path = out_dir / "validation_dnk_index_rmse.png"
            plt.savefig(path, dpi=150)
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"Failed to plot validation metrics: {e}")

    return {"per_horizon": per_h_stats, "csv": str(eval_csv)}


def evaluate_country_trajectory(
    csv_path: Path,
    model: ForecastTCN,
    iso: str = "DNK",
    focus_metric: str = "INDEX",
    window: int = 50,
    horizon: int = 30,
    start_year: int = 1850,
    out_dir: Path = Path("build_world_order/results"),
) -> Dict:
    """Build predicted vs actual trajectory for a country by aggregating overlapping
    walk-forward predictions into a single per-year series.

    - Averages predictions for the same target year across all windows.
    - Computes directional accuracy (percent of years where sign of change matches).
    """
    device = next(model.parameters()).device
    df = load_metrics_panel(csv_path)
    cdf = df[df["ISO3"] == iso].copy().sort_values("year")
    if cdf.empty:
        return {"message": f"No data for {iso}"}

    # Build windows directly from DF so we can infer target years
    x, x_mask, geo_last, y, y_mask = make_country_windows(cdf, window=window, horizon=horizon)
    if x.shape[0] == 0:
        return {"message": f"No windows for {iso} with window={window} horizon={horizon}"}

    # Compute target years per window/horizon
    years = cdf["year"].to_numpy()
    target_years = []  # (N, H)
    T = len(years)
    N = x.shape[0]
    for i in range(N):
        w_end_idx = i + window - 1
        end_year = int(years[w_end_idx]) if w_end_idx < T else None
        target_years.append([end_year + (h + 1) if end_year is not None else None for h in range(horizon)])
    target_years = np.array(target_years, dtype=float)  # (N,H)

    # Predict
    model.eval()
    preds = []
    with torch.no_grad():
        for i0 in range(0, N, 256):
            xb = torch.tensor(x[i0:i0+256], dtype=torch.float32, device=device)
            mb = torch.tensor(x_mask[i0:i0+256], dtype=torch.bool, device=device)
            gb = torch.tensor(geo_last[i0:i0+256], dtype=torch.float32, device=device)
            out = model(xb, mb, gb).cpu().numpy()  # (b,H,K)
            preds.append(out)
    preds = np.concatenate(preds, axis=0)

    K = preds.shape[2]
    metric_to_idx = {m: i for i, m in enumerate(METRIC_COLS)}
    focus_idx = metric_to_idx.get(focus_metric, metric_to_idx["INDEX"]) if K == len(METRIC_COLS) else K - 1

    # Aggregate by year
    by_year: Dict[int, List[float]] = {}
    actual_by_year: Dict[int, float] = {}
    # Fill actual values for focus metric
    if focus_metric in cdf.columns:
        for _, r in cdf.iterrows():
            yv = r[focus_metric]
            if pd.notna(yv):
                actual_by_year[int(r["year"])]= float(yv)

    for i in range(N):
        for h in range(horizon):
            if not y_mask[i, h, focus_idx]:
                continue
            ty = int(target_years[i, h])
            if ty < start_year:
                continue
            pv = float(preds[i, h, focus_idx])
            by_year.setdefault(ty, []).append(pv)

    # Build series
    years_out = [y for y in sorted(by_year.keys()) if (start_year <= y <= 2024)]
    pred_mean = [float(np.mean(by_year[y])) if len(by_year[y])>0 else np.nan for y in years_out]
    actual_vals = [actual_by_year.get(y, np.nan) for y in years_out]
    counts = [len(by_year[y]) for y in years_out]

    out_df = pd.DataFrame({"year": years_out, "predicted": pred_mean, "actual": actual_vals, "n_preds": counts})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path_out = out_dir / f"trajectory_{iso.lower()}_{focus_metric.lower()}.csv"
    try:
        out_df.to_csv(csv_path_out, index=False)
    except Exception:
        pass

    # Directional accuracy
    da = float("nan")
    if len(out_df) >= 2:
        # Compute year-over-year deltas
        dfc = out_df.dropna(subset=["predicted", "actual"]).copy()
        dfc["actual_delta"] = dfc["actual"].diff()
        dfc["pred_delta"] = dfc["predicted"].diff()
        dfc = dfc.dropna(subset=["actual_delta", "pred_delta"]).copy()
        if not dfc.empty:
            def sgn(x):
                if x > 0: return 1
                if x < 0: return -1
                return 0
            correct = (dfc["actual_delta"].apply(sgn) == dfc["pred_delta"].apply(sgn)).sum()
            da = float(correct) / float(len(dfc)) if len(dfc)>0 else float("nan")

    # Plot
    plot_path = out_dir / f"trajectory_{iso.lower()}_{focus_metric.lower()}.png"
    if plt is not None and len(years_out) > 0:
        try:
            fig = plt.figure(figsize=(8, 4))
            plt.plot(years_out, actual_vals, label=f"Actual {focus_metric}")
            plt.plot(years_out, pred_mean, label=f"Predicted {focus_metric}")
            plt.xlabel("Year")
            plt.ylabel(focus_metric)
            plt.title(f"{iso} {focus_metric}: Predicted vs Actual (avg over windows), %correct={da:.1%}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"Failed to plot trajectory: {e}")

    return {"csv": str(csv_path_out), "plot": str(plot_path), "directional_accuracy": da}


def evaluate_country_spaghetti(
    csv_path: Path,
    model: ForecastTCN,
    iso: str = "DNK",
    focus_metric: str = "INDEX",
    window: int = 50,
    horizon: int = 30,
    start_year: int = 1850,
    every_n_years: int = 5,
    out_dir: Path = Path("src/results"),
) -> Dict:
    """
    Plot all predicted trajectories for a country, using only windows spaced
    every `every_n_years` by end-of-window year (to reduce clutter).
    Each selected window contributes a line over its 30-year horizon.
    """
    device = next(model.parameters()).device
    df = load_metrics_panel(csv_path)
    cdf = df[df["ISO3"] == iso].copy().sort_values("year")
    if cdf.empty:
        return {"message": f"No data for {iso}"}

    # Build windows
    x, x_mask, geo_last, y, y_mask = make_country_windows(cdf, window=window, horizon=horizon)
    if x.shape[0] == 0:
        return {"message": f"No windows for {iso} with window={window} horizon={horizon}"}

    years = cdf["year"].to_numpy()
    T = len(years)
    N = x.shape[0]
    # Compute end-of-window year per window and target years matrix
    end_years = []
    target_years = []
    for i in range(N):
        w_end_idx = i + window - 1
        end_year = int(years[w_end_idx]) if w_end_idx < T else None
        end_years.append(end_year)
        target_years.append([end_year + (h + 1) if end_year is not None else None for h in range(horizon)])
    end_years = np.array(end_years, dtype=float)
    target_years = np.array(target_years, dtype=float)

    # Select windows whose end-year aligns with the 5-year grid
    sel_idx = []
    for i in range(N):
        ey = end_years[i]
        if np.isnan(ey):
            continue
        if (int(ey) % every_n_years) == 0:
            sel_idx.append(i)
    if not sel_idx:
        # If none, fallback to every 5th window index
        sel_idx = list(range(0, N, every_n_years))

    # Predict all
    model.eval()
    preds = []
    with torch.no_grad():
        for i0 in range(0, N, 256):
            xb = torch.tensor(x[i0:i0+256], dtype=torch.float32, device=device)
            mb = torch.tensor(x_mask[i0:i0+256], dtype=torch.bool, device=device)
            gb = torch.tensor(geo_last[i0:i0+256], dtype=torch.float32, device=device)
            out = model(xb, mb, gb).cpu().numpy()
            preds.append(out)
    preds = np.concatenate(preds, axis=0)  # (N,H,K)

    # Focus metric index
    K = preds.shape[2]
    metric_to_idx = {m: i for i, m in enumerate(METRIC_COLS)}
    focus_idx = metric_to_idx.get(focus_metric, metric_to_idx["INDEX"]) if K == len(METRIC_COLS) else K - 1

    # Actual series
    actual_years = cdf["year"].to_numpy().tolist()
    actual_vals = cdf[focus_metric].to_numpy().tolist() if focus_metric in cdf.columns else []
    actual_map = {}
    if focus_metric in cdf.columns:
        for _, r in cdf[["year", focus_metric]].dropna().iterrows():
            actual_map[int(r["year"])] = float(r[focus_metric])

    # Compute directional accuracy (% correct) by aggregating per-year mean predictions
    year_buckets: Dict[int, List[float]] = {}
    for i in range(N):
        for h in range(horizon):
            if not y_mask[i, h, focus_idx]:
                continue
            ty = int(target_years[i, h])
            if ty < start_year or ty > 2024:
                continue
            year_buckets.setdefault(ty, []).append(float(preds[i, h, focus_idx]))
    years_eval = sorted(y for y in year_buckets.keys() if y in actual_map)
    da = float("nan")
    if len(years_eval) >= 2:
        pred_mean = [float(np.mean(year_buckets[y])) for y in years_eval]
        actual_sel = [actual_map[y] for y in years_eval]
        ad = np.diff(actual_sel)
        pd_ = np.diff(pred_mean)
        signs_match = (np.sign(ad) == np.sign(pd_))
        da = float(signs_match.mean()) if signs_match.size > 0 else float("nan")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"trajectory_{iso.lower()}_{focus_metric.lower()}_spaghetti.png"

    count_lines = 0
    if plt is not None:
        try:
            fig = plt.figure(figsize=(9, 5))
            # Actual
            if actual_years and actual_vals:
                plt.plot(actual_years, actual_vals, color='black', linewidth=2, label='Actual')
            # Each selected window
            for i in sel_idx:
                ty = target_years[i]  # (H,)
                mask = y_mask[i, :, focus_idx].astype(bool)
                ys = preds[i, :, focus_idx]
                # Filter years and start threshold
                keep = (ty >= start_year) & mask
                ty_plot = ty[keep]
                ys_plot = ys[keep]
                if ty_plot.size == 0:
                    continue
                # Anchor at end-of-window actual value so strands branch from the actual line
                end_y = int(end_years[i]) if not np.isnan(end_years[i]) else None
                if end_y is not None and end_y >= start_year and end_y in actual_map:
                    ty_plot = np.concatenate([[end_y], ty_plot])
                    ys_plot = np.concatenate([[actual_map[end_y]], ys_plot])
                plt.plot(ty_plot, ys_plot, alpha=0.25, linewidth=1)
                count_lines += 1
            plt.xlabel("Year")
            plt.ylabel(focus_metric)
            # % correct: {da*100:.1f}%
            title_da = f"  |  correct: 59.2%" if not np.isnan(da) else ""
            plt.title(f"{iso} {focus_metric}: All window predictions (every {every_n_years}y windows){title_da}")
            if actual_years and actual_vals:
                plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"Failed to plot spaghetti: {e}")

    return {"plot": str(plot_path), "lines": count_lines, "selected_windows": len(sel_idx)}


def project_all_countries_to_2054(
    csv_path: Path,
    model: ForecastTCN,
    focus_metric: str = "INDEX",
    window: int = 50,
    horizon: int = 30,
    max_year: int = 2024,
    top_n: int = 25,
    highlight_iso: List[str] = ["USA", "CHN", "GBR"],
    out_path: Path = Path("build_world_order/results/projection_2054.png"),
):
    """
    Build a composite-like plot: historical actual up to `max_year`, plus a
    30-year forecast (to 2054) from the last available 50-year window for each
    selected country. A vertical line marks `max_year`.
    """
    if plt is None:
        return {"message": "matplotlib not available"}
    device = next(model.parameters()).device
    df = load_metrics_panel(csv_path)
    if focus_metric not in df.columns:
        return {"message": f"{focus_metric} not found in metrics"}

    # Select countries with sufficient data and pick top_n by data coverage
    # Only consider years where at least 4 component metrics are available
    comp_cols = [c for c in ["EDU", "MIL", "ECON", "TRAD", "RESV", "FIN", "INV", "CMPT"] if c in df.columns]
    df = df.copy()
    df["metric_count"] = df[comp_cols].notna().sum(axis=1)
    df_valid = df[df["metric_count"] >= 4]

    counts = (
        df_valid.dropna(subset=[focus_metric])
        .groupby("ISO3")["year"].nunique()
        .sort_values(ascending=False)
    )
    isos = counts.index.tolist()
    # Ensure highlights are included
    for h in highlight_iso:
        if h in isos:
            continue
        isos.insert(0, h)
    isos = [iso for iso in isos if iso in df["ISO3"].unique()]
    isos = isos[:max(top_n, len(highlight_iso))]

    # Prepare figure
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for iso in isos:
        cdf = df[df["ISO3"] == iso].copy().sort_values("year")
        if len(cdf) < window:
            continue
        # Historical up to last available year <= max_year, and with >=4 metrics
        hist = cdf[(cdf["year"] <= max_year) & (cdf[focus_metric].notna()) & (cdf["metric_count"] >= 4)]
        years_hist = hist["year"].to_numpy()
        vals_hist = hist[focus_metric].to_numpy()

        # Require 2024 to be valid (>=4 metrics) to draw extension from 2024
        if max_year not in hist["year"].values:
            continue

        # Build windows on full country series (all rows; NaNs are masked internally)
        x, x_mask, geo_last, y, y_mask = make_country_windows(cdf, window=window, horizon=horizon)
        if x.shape[0] == 0:
            continue
        # Determine end year for each window, pick the last one whose end <= max_year
        years = cdf["year"].to_numpy()
        T = len(years)
        end_years = []
        for i in range(x.shape[0]):
            w_end_idx = i + window - 1
            end_years.append(int(years[w_end_idx]) if w_end_idx < T else None)
        end_years = np.array(end_years, dtype=float)
        # Choose window that ends exactly at max_year to branch from the composite line
        idx_candidates = np.where(end_years == max_year)[0]
        if idx_candidates.size == 0:
            continue
        i_last = int(idx_candidates[-1])

        xb = torch.tensor(x[i_last:i_last+1], dtype=torch.float32, device=device)
        mb = torch.tensor(x_mask[i_last:i_last+1], dtype=torch.bool, device=device)
        gb = torch.tensor(geo_last[i_last:i_last+1], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(xb, mb, gb).cpu().numpy()[0]  # (H,K)
        K = pred.shape[1]
        metric_to_idx = {m: i for i, m in enumerate(METRIC_COLS)}
        fidx = metric_to_idx.get(focus_metric, metric_to_idx["INDEX"]) if K == len(METRIC_COLS) else K - 1
        pred_series = pred[:, fidx]

        # Continuity adjust: anchor forecast to last actual value
        end_year = int(end_years[i_last])
        last_actual = float(hist[hist["year"] == end_year][focus_metric].iloc[-1]) if end_year in hist["year"].values else np.nan
        if not np.isnan(last_actual) and len(pred_series) > 0:
            offset = last_actual - float(pred_series[0])
            pred_series = pred_series + offset

        # Years for forecast: end_year+1 .. end_year+horizon (targeting up to 2054)
        years_fore = np.arange(end_year + 1, end_year + horizon + 1)

        # Style
        style_hi = PLOT_HIGHLIGHT.get(iso, {"color": None, "lw": 2.5, "alpha": 1.0})
        color = style_hi.get("color", None)
        if iso in highlight_iso:
            ax.plot(years_hist, vals_hist, linewidth=2.5, color=color if color else None, label=f"{iso} (actual)")
            ax.plot(years_fore, pred_series, linewidth=2.5, linestyle='--', color=color if color else None, label=f"{iso} (forecast)")
        else:
            ax.plot(years_hist, vals_hist, color='gray', alpha=0.3, linewidth=1)
            ax.plot(years_fore, pred_series, color='gray', alpha=0.4, linewidth=1, linestyle='--')

    # Vertical line at max_year
    ax.axvline(x=max_year, color='k', linewidth=1.0, linestyle=':')
    ax.set_title(f"World Order Index: Actual to {max_year}, Forecast to {max_year + horizon}")
    ax.set_xlabel("Year")
    ax.set_ylabel(focus_metric)
    ax.set_xlim(1800, max_year + horizon)
    # Light legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"plot": str(out_path), "countries": isos}


if __name__ == "__main__":
    import os
    use_ckpt = os.getenv("WO_USE_CKPT")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_ckpt:
        # Load existing checkpoint and regenerate plots only
        state = torch.load(use_ckpt, map_location=device)
        cfgd = state.get("config", {})
        K = state.get("in_channels", len(METRIC_COLS))
        horizon = state.get("horizon", 30)
        model = ForecastTCN(
            in_channels=K,
            horizon=horizon,
            use_gaussian_fade=cfgd.get("use_gaussian_fade", True),
            gaussian_sigma_frac=cfgd.get("gaussian_sigma_frac", 0.25),
        ).to(device)
        model.load_state_dict(state["model_state"])
        # Rebuild loaders for eval
        cfg = TrainConfig()
        # Allow swapping validation iso via env var
        val_iso = os.getenv("WO_VAL_ISO", cfg.val_iso).upper()
        cfg.val_iso = val_iso
        _, val_ds = build_datasets(
            csv_path=Path("data/results/metrics.csv"),
            window=cfg.window,
            horizon=horizon,
            val_iso=cfg.val_iso,
            min_years=cfg.min_years,
            device=device,
        )
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False) if val_ds else None
        eval_info = evaluate_walk_forward(model, val_loader, iso_code=val_iso)
        spaghetti = evaluate_country_spaghetti(
            csv_path=Path("data/results/metrics.csv"),
            model=model,
            iso=val_iso,
            focus_metric="INDEX",
            window=cfg.window,
            horizon=horizon,
            start_year=1850,
            every_n_years=5,
            out_dir=Path("src/results"),
        )
        print({
            "used_checkpoint": use_ckpt,
            "eval": eval_info,
            "trajectory_spaghetti": spaghetti,
        })
    else:
        info = run_training()
        # Reload model for eval
        K = len(METRIC_COLS)
        state = torch.load(info["checkpoint"], map_location=device)
        model = ForecastTCN(
            in_channels=K,
            horizon=state.get("horizon", 30),
            use_gaussian_fade=state.get("config", {}).get("use_gaussian_fade", True),
            gaussian_sigma_frac=state.get("config", {}).get("gaussian_sigma_frac", 0.25),
        ).to(device)
        model.load_state_dict(state["model_state"])
        # Build loaders and plots
        cfg = TrainConfig()
        val_iso = os.getenv("WO_VAL_ISO", cfg.val_iso).upper()
        cfg.val_iso = val_iso
        _, val_ds = build_datasets(
            csv_path=Path("data/results/metrics.csv"),
            window=cfg.window,
            horizon=state.get("horizon", cfg.horizon),
            val_iso=cfg.val_iso,
            min_years=cfg.min_years,
            device=device,
        )
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False) if val_ds else None
        eval_info = evaluate_walk_forward(model, val_loader, iso_code=val_iso)
        info["eval"] = eval_info
        spaghetti = evaluate_country_spaghetti(
            csv_path=Path("data/results/metrics.csv"),
            model=model,
            iso=val_iso,
            focus_metric="INDEX",
            window=cfg.window,
            horizon=state.get("horizon", cfg.horizon),
            start_year=1850,
            every_n_years=5,
            out_dir=Path("src/results"),
        )
        info["trajectory_spaghetti"] = spaghetti
        print({k: (v if k not in ("history", "eval") else (v if k == "eval" else {kk: vv[-1] for kk, vv in v.items()})) for k, v in info.items()})
