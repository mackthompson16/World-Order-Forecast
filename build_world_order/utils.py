import pandas as pd
import numpy as np
from typing import Iterable, Optional


def interpolate_panel(df: pd.DataFrame, group_cols: Iterable[str], value_cols: Iterable[str]) -> pd.DataFrame:
    out = df.sort_values(list(group_cols) + ["year"]).copy()
    for col in value_cols:
        if col in out.columns:
            # Coerce to numeric for proper interpolation
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = (
                out.groupby(list(group_cols))[col]
                .apply(lambda s: s.interpolate(method="linear", limit_area="inside"))
                .reset_index(level=list(range(len(group_cols))), drop=True)
            )
    return out


def forward_fill_limited(df: pd.DataFrame, group_cols: Iterable[str], value_cols: Iterable[str], max_years: int, year_col: str = "year") -> pd.DataFrame:
    out = df.sort_values(list(group_cols) + [year_col]).copy()
    for col in value_cols:
        if col not in out.columns:
            continue
        # Track last non-null value and year per group
        def ffill_limit(group: pd.DataFrame) -> pd.Series:
            last_val = np.nan
            last_year: Optional[float] = None
            res = []
            for _, row in group.iterrows():
                y = row[year_col]
                v = row[col]
                if pd.notna(v):
                    last_val = v
                    last_year = y
                    res.append(v)
                else:
                    if last_year is not None and (y - last_year) <= max_years:
                        res.append(last_val)
                    else:
                        res.append(np.nan)
            return pd.Series(res, index=group.index)

        out[col] = out.groupby(list(group_cols), group_keys=False).apply(ffill_limit)
    return out


def yearwise_min_max_norm(series: pd.Series, year: pd.Series) -> pd.Series:
    df = pd.DataFrame({"val": series, "year": year})
    def norm_year(g: pd.DataFrame) -> pd.Series:
        vals = g["val"].dropna()
        if len(vals) < 4:
            return pd.Series([np.nan] * len(g), index=g.index)
        vmin = vals.min()
        vmax = vals.max()
        if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
            return pd.Series([np.nan] * len(g), index=g.index)
        return (g["val"] - vmin) / (vmax - vmin)
    out = df.groupby("year", group_keys=False).apply(norm_year)
    return out


def moving_average(s: pd.Series, window: int) -> pd.Series:
    if window is None or window <= 1:
        return s
    return s.rolling(window=window, min_periods=max(1, window // 2), center=True).mean()

