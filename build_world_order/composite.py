from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict


def compute_composite(metrics: pd.DataFrame) -> pd.DataFrame:
    df = metrics.copy()
    # Map component names to weights; handle synonyms
    components: Dict[str, float] = {
        "EDU": 0.15,
        "CMPT": 0.15,  # Competitiveness
        "INV": 0.15,   # Technology/Innovation
        "ECON": 0.15,  # EconomicOutput
        "TRAD": 0.10,
        "MIL": 0.10,
        "FIN": 0.10,
        "RESV": 0.10,
    }

    comp_cols = [c for c in components.keys() if c in df.columns]
    weights = np.array([components[c] for c in comp_cols], dtype=float)
    if not len(comp_cols):
        df["WorldOrderIndex"] = np.nan
        return df

    values = df[comp_cols].to_numpy(dtype=float)
    # For each row, renormalize weights over available components
    mask = ~np.isnan(values)
    # Apply mask to weights per row and renormalize
    weighted_mask = mask * weights[None, :]
    sum_w = weighted_mask.sum(axis=1)
    sum_w[sum_w == 0] = np.nan
    norm_w = np.divide(weighted_mask, sum_w[:, None])
    contrib = np.nansum(np.nan_to_num(values) * norm_w, axis=1)
    df["WorldOrderIndex"] = contrib
    return df


def write_composite(metrics_path: Path, out_path: Path) -> Path:
    metrics = pd.read_csv(metrics_path)
    comp = compute_composite(metrics)
    cols = [c for c in ["country_name", "ISO3", "year", "WorldOrderIndex"] if c in comp.columns]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comp[cols].to_csv(out_path, index=False)
    return out_path

