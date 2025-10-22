from __future__ import annotations

import numpy as np
import pandas as pd


def per_year_minmax(series: pd.Series, year_index: pd.Series) -> pd.Series:
    """Min–max normalize values per year across countries.

    For each year, computes (x - min) / (max - min). If a year has only one
    non-null value or max == min, return 0.5 for those non-null entries.

    Parameters
    - series: numeric values aligned to rows
    - year_index: year values aligned to rows
    """
    df = pd.DataFrame({"val": series, "year": year_index})

    def _norm(group: pd.DataFrame) -> pd.Series:
        vals = group["val"].astype(float)
        nonnull = vals.dropna()
        if nonnull.empty:
            return pd.Series(index=group.index, dtype=float)
        vmin = nonnull.min()
        vmax = nonnull.max()
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
            out = pd.Series(np.where(vals.notna(), 0.5, np.nan), index=group.index)
        else:
            out = (vals - vmin) / (vmax - vmin)
        return out

    return df.groupby("year", group_keys=False).apply(_norm).astype(float)


def fill_gaps(series: pd.Series) -> pd.Series:
    """Fill gaps according to rules:
    - Forward-fill consecutive gaps of length <= 3
    - Linearly interpolate longer internal gaps (>3)
    - Do not extrapolate beyond first/last real values
    """
    s = series.copy()
    # Forward-fill small gaps only (limit=3)
    s = s.ffill(limit=3)
    # Linear interpolate remaining internal NaNs
    s = s.interpolate(method="linear", limit_area="inside")
    return s

