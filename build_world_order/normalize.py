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


def enforce_min_countries_and_interpolate(
    df: pd.DataFrame,
    value_col: str,
    year_col: str = "year",
    country_col: str = "country",
    min_countries: int = 4,
) -> pd.DataFrame:
    """Pre-process a long panel before normalization.

    - Determine the first year where at least `min_countries` have non-null values.
    - Mask values prior to that year to NaN (insufficient cross-section).
    - Within each country, linearly interpolate missing values with `limit_area='inside'`.
      Do not extend before first or after last observed value.

    Returns a copy of df with `value_col` modified.
    """
    out = df.copy()
    counts = out.groupby(year_col)[value_col].apply(lambda s: s.notna().sum())
    years_ok = counts.index[counts >= min_countries]
    if len(years_ok) == 0:
        # Nothing qualifies; blank the column
        out[value_col] = np.nan
        return out
    start_year = int(years_ok.min())
    # Mask years before start_year
    out.loc[out[year_col] < start_year, value_col] = np.nan
    # Interpolate per country inside observed ranges
    out.sort_values([country_col, year_col], inplace=True)
    out[value_col] = (
        out.groupby(country_col, group_keys=False)[value_col]
        .apply(lambda s: s.interpolate(method="linear", limit_area="inside"))
        .astype(float)
    )
    return out
