import os
import math
from typing import Iterable, Dict

import numpy as np
import pandas as pd


# Canonical name normalization with a few common aliases
_ALIAS_MAP: Dict[str, str] = {
    # United States aliases
    "US": "USA",
    "U.S.": "USA",
    "U.S": "USA",
    "USA": "USA",
    "UNITED STATES": "USA",
    "UNITED STATES OF AMERICA": "USA",
    # ISO3 codes (selected, expandable)
    "DEU": "GERMANY",
    "CHN": "CHINA",
    "RUS": "RUSSIA",
    "FRA": "FRANCE",
    "NLD": "NETHERLANDS",
    "IND": "INDIA",
    "GBR": "UNITED KINGDOM",
    "UKM": "UNITED KINGDOM",
    "CAN": "CANADA",
    "AUS": "AUSTRALIA",
    "JPN": "JAPAN",
    "KOR": "SOUTH KOREA",
    "KAZ": "KAZAKHSTAN",
    "BRA": "BRAZIL",
    "MEX": "MEXICO",
    "TUR": "TURKEY",
    "ITA": "ITALY",
    "ESP": "SPAIN",
    "SWE": "SWEDEN",
    "NOR": "NORWAY",
    "DNK": "DENMARK",
    "CHE": "SWITZERLAND",
    "NGA": "NIGERIA",
    "ZAF": "SOUTH AFRICA",
    "SAU": "SAUDI ARABIA",
    "ARE": "UNITED ARAB EMIRATES",
    "IRN": "IRAN",
    "IRQ": "IRAQ",
    "ISR": "ISRAEL",
    "EGY": "EGYPT",
    "IDN": "INDONESIA",
    "VNM": "VIETNAM",
    "SGP": "SINGAPORE",
    "HKG": "HONG KONG",
    "NPL": "NEPAL",
    "PAK": "PAKISTAN",
    "BGD": "BANGLADESH",
    "PHL": "PHILIPPINES",
    "THA": "THAILAND",
    "MYS": "MALAYSIA",
    "NOR": "NORWAY",
    "POL": "POLAND",
    "NZE": "NEW ZEALAND",
    "NZL": "NEW ZEALAND",
    "BEL": "BELGIUM",
    "AUT": "AUSTRIA",
    "GRC": "GREECE",
    "PRT": "PORTUGAL",
    "NGA": "NIGERIA",
    "ETH": "ETHIOPIA",
    "DZA": "ALGERIA",
    "MAR": "MOROCCO",
    "ARG": "ARGENTINA",
    "CHL": "CHILE",
    "COL": "COLOMBIA",
    "PER": "PERU",
    "URY": "URUGUAY",
    "VEN": "VENEZUELA",
    # Common alternates that might appear
    "PRC": "CHINA",
    "PEOPLE'S REPUBLIC OF CHINA": "CHINA",
    "PEOPLES REPUBLIC OF CHINA": "CHINA",
    "RUSSIAN FEDERATION": "RUSSIA",
    "SOVIET UNION": "RUSSIA",
    "UK": "UNITED KINGDOM",
    "U.K.": "UNITED KINGDOM",
    "GREAT BRITAIN": "UNITED KINGDOM",
    "BRITAIN": "UNITED KINGDOM",
    "NETHERLANDS": "NETHERLANDS",
    "THE NETHERLANDS": "NETHERLANDS",
    "HOLLAND": "NETHERLANDS",
}


def canonical_country(name: str) -> str:
    if name is None:
        return name
    key = str(name).strip().upper()
    return _ALIAS_MAP.get(key, key)


def forward_fill_by_group(df: pd.DataFrame, group_cols: Iterable[str], sort_cols: Iterable[str], cols_to_ffill: Iterable[str]) -> pd.DataFrame:
    """Forward-fill specified columns within groups, after sorting.

    Extends previous year value forward when current is null.
    """
    df = df.sort_values(list(group_cols) + list(sort_cols)).copy()
    df[list(cols_to_ffill)] = (
        df.groupby(list(group_cols), dropna=False)[list(cols_to_ffill)].ffill()
    )
    return df


def min_max_norm_by_year(df: pd.DataFrame, year_col: str, value_col: str, out_col: str) -> pd.DataFrame:
    """Min-max normalize `value_col` within each year into `out_col`.

    If max == min (flat series) or only one non-null value, set 0.5.
    """
    def _norm(s: pd.Series) -> pd.Series:
        vmin = s.min(skipna=True)
        vmax = s.max(skipna=True)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return pd.Series([np.nan] * len(s), index=s.index)
        denom = vmax - vmin
        if not np.isfinite(denom) or denom == 0:
            # collapse to mid value if no dispersion
            return pd.Series([0.5 if pd.notna(x) else np.nan for x in s], index=s.index)
        return (s - vmin) / denom

    df[out_col] = df.groupby(year_col, dropna=False)[value_col].transform(_norm)
    return df


def share_by_year(df: pd.DataFrame, year_col: str, value_col: str, out_col: str) -> pd.DataFrame:
    """Compute share of `value_col` within each year into `out_col`.
    Guards against zero denominators.
    """
    totals = df.groupby(year_col, dropna=False)[value_col].transform(lambda s: s.sum(skipna=True))
    with np.errstate(divide='ignore', invalid='ignore'):
        share = np.where((totals != 0) & np.isfinite(totals), df[value_col] / totals, np.nan)
    df[out_col] = share
    return df


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def rolling_smooth(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()
