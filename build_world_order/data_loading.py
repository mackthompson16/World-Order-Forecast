from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .utils import canonical_country, YEAR_MIN, YEAR_MAX


DataFrames = Dict[str, pd.DataFrame]


def _clip_years(df: pd.DataFrame, year_col: str = "year") -> pd.DataFrame:
    if year_col in df.columns:
        df = df[df[year_col].between(YEAR_MIN, YEAR_MAX)]
    return df


def load_gmd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Canonical country names
    name_col = None
    for c in ["countryname", "country_name", "country"]:
        if c in df.columns:
            name_col = c
            break
    if name_col is None:
        raise ValueError("GMD.csv missing country name column")
    df["country"] = df[name_col].map(canonical_country)
    # Keep relevant columns
    keep = [
        "country", "year",
        "rGDP_USD", "USDfx", "cgovdebt_GDP",
        "exports_USD", "imports_USD",
    ]
    present = [c for c in keep if c in df.columns]
    df = df[present].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = _clip_years(df)
    return df


def load_education(path: Path) -> pd.DataFrame:
    # This file is wide with year columns; reshape to long
    wide = pd.read_csv(path)
    # Identify name column
    name_col = None
    for c in ["country", "country_name", "country name", "Country", "Country Name"]:
        if c in wide.columns:
            name_col = c
            break
    if name_col is None:
        raise ValueError("Education.csv missing country name column")
    # All columns that are years
    year_cols = [c for c in wide.columns if str(c).isdigit()]
    long = wide.melt(id_vars=[name_col], value_vars=year_cols, var_name="year", value_name="education")
    long["country"] = long[name_col].map(canonical_country)
    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    long["education"] = pd.to_numeric(long["education"], errors="coerce")
    long = long.dropna(subset=["country", "year"]).reset_index(drop=True)
    long = _clip_years(long)
    return long[["country", "year", "education"]]


def load_military(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Country code columns vary; prefer ISO-like or stateabb
    if "stateabb" in df.columns:
        df["country"] = df["stateabb"].map(canonical_country)
    elif "country" in df.columns:
        df["country"] = df["country"].map(canonical_country)
    else:
        raise ValueError("military.csv missing country identifier column")
    # Standardize column names
    if "cinc" not in df.columns and "CINC" in df.columns:
        df["cinc"] = df["CINC"]
    keep = ["country", "year", "cinc"]
    df = df[keep].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["cinc"] = pd.to_numeric(df["cinc"], errors="coerce")
    df = _clip_years(df)
    return df


def load_polity(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Polity dataset columns
    name_col = None
    for c in ["country", "country_name", "Country"]:
        if c in df.columns:
            name_col = c
            break
    if name_col is None:
        name_col = "country"  # will fail if not present
    if name_col not in df.columns:
        raise ValueError("polity.csv missing country column")
    # year may be split into byear/eyear spans; we take 'eyear' if present else need to derive
    if "eyear" in df.columns:
        df["year"] = pd.to_numeric(df["eyear"], errors="coerce")
    elif "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    else:
        raise ValueError("polity.csv missing year/eyear")
    df["country"] = df[name_col].map(canonical_country)
    # Clean polity score and sentinel values (-66, -77, -88)
    if "polity" not in df.columns:
        raise ValueError("polity.csv missing 'polity' column")
    df["polity"] = pd.to_numeric(df["polity"], errors="coerce")
    df.loc[df["polity"].isin([-66, -77, -88]), "polity"] = np.nan
    df = _clip_years(df)
    return df[["country", "year", "polity"]]


def load_chat(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect columns: country_name, year, many features
    name_col = None
    for c in ["country_name", "country", "Country"]:
        if c in df.columns:
            name_col = c
            break
    if name_col is None or "year" not in df.columns:
        raise ValueError("CHAT.csv missing country_name/year")
    df["country"] = df[name_col].map(canonical_country)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    # Drop id columns and keep numeric feature columns only
    drop_like = {name_col, "year", "country"}
    feature_cols = [c for c in df.columns if c not in drop_like]
    # Coerce to numeric
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["country", "year"] + feature_cols]
    df = _clip_years(df)
    return df


def load_all(data_dir: Path) -> Tuple[DataFrames, pd.DataFrame]:
    """Load all required datasets.

    Returns (dfs, years_grid) where years_grid contains all country-year pairs
    in the inclusive [YEAR_MIN, YEAR_MAX] range for countries observed.
    """
    gmd = load_gmd(data_dir / "GMD.csv")
    edu = load_education(data_dir / "Education.csv")
    mil = load_military(data_dir / "military.csv")
    pol = load_polity(data_dir / "polity.csv")
    chat = load_chat(data_dir / "CHAT.csv")

    # Determine universe of countries from any source
    countries = pd.Index(
        pd.concat([gmd["country"], edu["country"], mil["country"], pol["country"], chat["country"]]).dropna().unique()
    )
    years = pd.Index(range(YEAR_MIN, YEAR_MAX + 1))
    idx = pd.MultiIndex.from_product([countries, years], names=["country", "year"])
    grid = idx.to_frame(index=False)

    dfs: DataFrames = {"gmd": gmd, "education": edu, "military": mil, "polity": pol, "chat": chat}
    return dfs, grid

