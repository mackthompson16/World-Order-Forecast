from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .utils import canonical_country, ensure_dirs


def _education_long_raw(edu_path: str) -> pd.DataFrame:
    df = pd.read_csv(edu_path)
    year_cols = [c for c in df.columns if str(c).isdigit()]
    cname_col = None
    for cand in ["country name", "country", "Country", "country_name"]:
        if cand in df.columns:
            cname_col = cand
            break
    if cname_col is None:
        raise ValueError("Education.csv missing 'country name' column")

    long_df = df.melt(
        id_vars=[c for c in df.columns if c not in year_cols],
        value_vars=year_cols,
        var_name="year",
        value_name="education",
    )
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
    long_df["country"] = long_df[cname_col].astype(str).map(canonical_country)
    long_df["education"] = pd.to_numeric(long_df["education"], errors="coerce")
    return long_df[["country", "year", "education"]].dropna(subset=["year"]).copy()


def _military_raw(mil_path: str) -> pd.DataFrame:
    df = pd.read_csv(mil_path)
    if "stateabb" in df.columns:
        countries = df["stateabb"].astype(str)
    elif "country" in df.columns:
        countries = df["country"].astype(str)
    else:
        countries = df.get("ccode", pd.Series(np.nan, index=df.index)).astype(str)

    out = pd.DataFrame({
        "country": countries.map(canonical_country),
        "year": pd.to_numeric(df.get("year"), errors="coerce"),
        "milex": pd.to_numeric(df.get("milex"), errors="coerce"),
        "milper": pd.to_numeric(df.get("milper"), errors="coerce"),
    })
    # Treat -9 as missing
    for c in ["milex", "milper"]:
        if c in out.columns:
            out.loc[out[c] == -9, c] = np.nan
    return out.dropna(subset=["year"]).copy()


def _gmd_raw(gmd_path: str) -> pd.DataFrame:
    df = pd.read_csv(gmd_path)
    if "countryname" in df.columns:
        countries = df["countryname"].astype(str)
    elif "ISO3" in df.columns:
        countries = df["ISO3"].astype(str)
    else:
        for cand in ["country", "Country", "name"]:
            if cand in df.columns:
                countries = df[cand].astype(str)
                break
        else:
            countries = pd.Series([None] * len(df))

    cols = [
        "rGDP_USD", "exports_USD", "imports_USD", "USDfx", "infl",
        "CA_GDP", "M2", "govdef_GDP",
    ]
    for c in cols + ["year"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out = pd.DataFrame({
        "country": countries.map(canonical_country),
        "year": df.get("year"),
    })
    for c in cols:
        out[c] = df.get(c)
    return out.dropna(subset=["year"]).copy()


def compute_data_coverage(edu_path: str, mil_path: str, gmd_path: str) -> pd.DataFrame:
    """Return DataFrame: label, year, countries_available."""
    edu = _education_long_raw(edu_path)
    mil = _military_raw(mil_path)
    gmd = _gmd_raw(gmd_path)

    records = []

    # Education
    ed_counts = edu.dropna(subset=["education"]).groupby("year")["country"].nunique()
    for y, n in ed_counts.items():
        records.append({"label": "Education", "year": int(y), "countries_available": int(n)})

    # Military
    for c in ["milex", "milper", "cinc"]:
        if c in mil.columns:
            cnt = mil.dropna(subset=[c]).groupby("year")["country"].nunique()
            for y, n in cnt.items():
                records.append({"label": c, "year": int(y), "countries_available": int(n)})

    # GMD variables
    for c in ["rGDP_USD", "exports_USD", "imports_USD", "USDfx", "infl", "CA_GDP", "M2", "govdef_GDP", "cgovdebt"]:
        if c in gmd.columns:
            cnt = gmd.dropna(subset=[c]).groupby("year")["country"].nunique()
            for y, n in cnt.items():
                records.append({"label": c, "year": int(y), "countries_available": int(n)})

    coverage = pd.DataFrame.from_records(records)
    return coverage.sort_values(["label", "year"]).reset_index(drop=True)


def plot_data_coverage(coverage_df: pd.DataFrame, out_dir: str, start_year: int = 1800, end_year: int | None = 2024) -> str:
    ensure_dirs(out_dir)
    df = coverage_df.copy()
    df = df[df["year"] >= start_year]
    if end_year is not None:
        df = df[df["year"] <= end_year]

    plt.figure(figsize=(10, 6))
    for label, sub in df.groupby("label"):
        sub = sub.sort_values("year")
        plt.plot(sub["year"], sub["countries_available"], label=label)

    plt.title("Data Availability by Year")
    plt.xlabel("Year")
    plt.ylabel("Countries with Available Data")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    out_path = f"{out_dir}/Data_Availability_Timeline.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
