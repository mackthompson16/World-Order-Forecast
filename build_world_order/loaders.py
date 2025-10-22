from typing import List, Optional

import numpy as np
import pandas as pd

from .utils import canonical_country, forward_fill_by_group


def load_education(path: str) -> pd.DataFrame:
    """Load Education.csv (wide format), return long format with columns: country, year, education.

    Expected columns: ccode, country name, 1500, 1501, ...
    We only rely on country name; numeric years are melted.
    """
    df = pd.read_csv(path)
    # Identify year columns (numeric-like)
    year_cols = [c for c in df.columns if str(c).isdigit()]
    value_name = "education"
    long_df = df.melt(
        id_vars=[c for c in df.columns if c not in year_cols],
        value_vars=year_cols,
        var_name="year",
        value_name=value_name,
    )
    # Clean types
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
    # Build canonical country
    cname_col = None
    for cand in ["country name", "country", "Country", "country_name"]:
        if cand in long_df.columns:
            cname_col = cand
            break
    if cname_col is None:
        raise ValueError("Education.csv missing 'country name' column")

    long_df["country"] = long_df[cname_col].astype(str).map(canonical_country)
    # Keep needed columns
    edu = long_df[["country", "year", value_name]].dropna(subset=["year"])\
        .sort_values(["country", "year"]).reset_index(drop=True)

    # Forward-fill within each country
    edu[value_name] = pd.to_numeric(edu[value_name], errors="coerce")
    # Forward-fill within each country (only use year-1, year-2, ...)
    edu = forward_fill_by_group(edu, ["country"], ["year"], [value_name])
    return edu


def load_military(path: str) -> pd.DataFrame:
    """Load military.csv, -9 treated as null. Return columns: country, year, milex, milper.

    Schema: stateabb,ccode,year,milex,milper,irst,pec,tpop,upop,cinc,version
    """
    df = pd.read_csv(path)
    # Treat -9 as null for numeric fields
    numeric_cols = [c for c in ["milex", "milper", "cinc"] if c in df.columns]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] == -9, c] = np.nan

    # Build country name: prefer stateabb if present, else fallback to any name column
    if "stateabb" in df.columns:
        country_series = df["stateabb"].astype(str)
    elif "country" in df.columns:
        country_series = df["country"].astype(str)
    else:
        # Fallback: try ccode as string
        country_series = df.get("ccode", pd.Series(np.nan, index=df.index)).astype(str)

    out = pd.DataFrame({
        "country": country_series.map(canonical_country),
        "year": pd.to_numeric(df.get("year"), errors="coerce"),
        "milex": df.get("milex"),
        "milper": df.get("milper"),
        "cinc": df.get("cinc"),
    })
    out = out.dropna(subset=["year"]).sort_values(["country", "year"]).reset_index(drop=True)
    out = forward_fill_by_group(out, ["country"], ["year"], [c for c in ["milex", "milper", "cinc"] if c in out.columns])
    return out


def load_gmd(path: str) -> pd.DataFrame:
    """Load GMD.csv and return standardized columns for metrics.

    We rely on columns mentioned in the provided schema. Missing ones will remain NaN.
    Canonicalizes country names from `countryname` if exists; fallback to ISO3.
    """
    df = pd.read_csv(path)
    # Establish country field
    if "countryname" in df.columns:
        countries = df["countryname"].astype(str)
    elif "ISO3" in df.columns:
        countries = df["ISO3"].astype(str)
    else:
        # Fallback to any likely column name
        for cand in ["country", "Country", "name"]:
            if cand in df.columns:
                countries = df[cand].astype(str)
                break
        else:
            countries = pd.Series([None] * len(df))

    out_cols = [
        "rGDP_USD", "exports_USD", "imports_USD", "USDfx", "infl",
        "CA_GDP", "M2", "M3", "govdef_GDP", "cgovdebt_GDP", "cgovdebt", "finv_GDP",
    ]
    # Coerce numeric
    for c in out_cols + ["year"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out = pd.DataFrame({
        "country": countries.map(canonical_country),
        "year": df.get("year"),
        "rGDP_USD": df.get("rGDP_USD"),
        "exports_USD": df.get("exports_USD"),
        "imports_USD": df.get("imports_USD"),
        "USDfx": df.get("USDfx"),
        "infl": df.get("infl"),
        "CA_GDP": df.get("CA_GDP"),
        "M2": df.get("M2"),
        "M3": df.get("M3"),
        "govdef_GDP": df.get("govdef_GDP"),
        "cgovdebt_GDP": df.get("cgovdebt_GDP"),
        "cgovdebt": df.get("cgovdebt"),
        "finv_GDP": df.get("finv_GDP"),
    })

    out = out.dropna(subset=["year"]).sort_values(["country", "year"]).reset_index(drop=True)
    # Forward-fill within country for all relevant columns
    fcols = [c for c in out.columns if c not in ("country", "year")]
    out = forward_fill_by_group(out, ["country"], ["year"], fcols)
    return out


def load_chat(path: str) -> pd.DataFrame:
    """Load CHAT.csv and return wide DataFrame: country, year, <numeric columns...>.

    - country_name, year are expected columns
    - All other columns treated as candidate numeric features
    - Forward-fill within country by year for stability
    """
    df = pd.read_csv(path, low_memory=False)
    # Establish country and year
    cname_col = None
    for cand in ["country_name", "country", "Country"]:
        if cand in df.columns:
            cname_col = cand
            break
    if cname_col is None:
        raise ValueError("CHAT.csv missing 'country_name' column")

    if "year" not in df.columns:
        raise ValueError("CHAT.csv missing 'year' column")

    out = df.copy()
    out["country"] = out[cname_col].astype(str).map(canonical_country)
    # Clean year: strip thousands separators and coerce; drop out-of-range
    out["year"] = (
        out["year"].astype(str).str.replace(",", "", regex=False)
    )
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    # Guard unrealistic years produced by parsing glitches
    out.loc[(out["year"] < 1500) | (out["year"] > 2100), "year"] = np.nan

    # Identify numeric feature columns: exclude country/year and any non-feature ids
    exclude = {cname_col, "country", "year"}
    feature_cols = [c for c in out.columns if c not in exclude]
    # Coerce to numeric where possible
    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["year"]).sort_values(["country", "year"]).reset_index(drop=True)
    out = forward_fill_by_group(out, ["country"], ["year"], feature_cols)
    return out[["country", "year"] + feature_cols]


def load_polity(path: str) -> pd.DataFrame:
    """Load polity.csv and return columns: country, year, polity.
    Uses 'country' as name, 'eyear' or 'year' if present for the time.
    """
    df = pd.read_csv(path, low_memory=False)
    # Country
    if "country" in df.columns:
        countries = df["country"].astype(str)
    else:
        for cand in ["country_name", "scode", "p5"]:
            if cand in df.columns:
                countries = df[cand].astype(str)
                break
        else:
            countries = pd.Series([None] * len(df))

    # Year selection: polity datasets often store end year (eyear)
    year_col = "year" if "year" in df.columns else ("eyear" if "eyear" in df.columns else None)
    if year_col is None:
        raise ValueError("polity.csv missing 'year' or 'eyear' column")

    # Clean year string then coerce and bound-check
    year_series = df[year_col].astype(str).str.replace(",", "", regex=False)
    year_series = pd.to_numeric(year_series, errors="coerce")
    year_series = year_series.where((year_series >= 1500) & (year_series <= 2100))

    out = pd.DataFrame({
        "country": countries.map(canonical_country),
        "year": year_series,
        "polity": pd.to_numeric(df.get("polity"), errors="coerce"),
    })
    out = out.dropna(subset=["year"]).sort_values(["country", "year"]).reset_index(drop=True)
    # No forward-fill per request; use only observed values
    return out
