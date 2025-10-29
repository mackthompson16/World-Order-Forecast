from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data_loading import load_country_reference
from .utils import interpolate_panel, yearwise_min_max_norm


def _list_geo_files(geo_dir: Path) -> List[Path]:
    geo_dir = Path(geo_dir)
    if not geo_dir.exists():
        return []
    return sorted([p for p in geo_dir.glob("*.csv") if p.is_file()])


def _sanitize_key(name: str) -> str:
    s = str(name).strip().lower()
    for ch in ["/", "\\", "(", ")", ",", ":", ";", "-", "%", " "]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _map_indicator_key(indicator_name: str, file_stem: str) -> str:
    base = _sanitize_key(indicator_name or file_stem)
    # Try to map to expected canonical names for fill policy
    name = base
    if any(k in base for k in ["land_area", "land_area_sq", "land_area_km"]):
        return "land_area"
    if "precip" in base:
        return "precipitation"
    if "natural" in base and ("gdp" in base or "pct" in base or "percent" in base):
        return "Natural_GDP_PCT"
    if "ag" in base and ("land" in base or "agricultural_land" in base):
        return "Ag_land"
    if "arable" in base and ("pct" in base or "percent" in base):
        return "Arable_PCT"
    if "forest" in base:
        return "forest_area"
    return base


def _load_geo_csv(path: Path) -> Tuple[str, pd.DataFrame]:
    df = pd.read_csv(path)
    # Expected schema: Country Name, Country Code, Indicator Name, Indicator Code, year...
    # Determine indicator key
    indicator_name = None
    if "Indicator Name" in df.columns and not df["Indicator Name"].dropna().empty:
        uniq = df["Indicator Name"].dropna().astype(str).unique()
        # Use the first indicator name
        indicator_name = uniq[0] if len(uniq) >= 1 else None
    key = _map_indicator_key(indicator_name or path.stem, path.stem)

    # Melt wide year columns
    id_cols = [c for c in ["Country Name", "Country Code", "Indicator Name", "Indicator Code"] if c in df.columns]
    year_cols = [c for c in df.columns if c not in id_cols]
    long = df.melt(id_vars=id_cols, value_vars=year_cols, var_name="year", value_name=key)
    long.rename(columns={"Country Name": "country_name", "Country Code": "ISO3"}, inplace=True)
    long["ISO3"] = long["ISO3"].astype(str).str.upper()
    # Coerce values and years
    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    long[key] = pd.to_numeric(long[key], errors="coerce")
    long = long.dropna(subset=["ISO3", "year"]).copy()
    return key, long[["ISO3", "country_name", "year", key]]


def _fit_and_fill_linear(group: pd.Series, years: pd.Series) -> pd.Series:
    s = group.copy()
    x = years.values.astype(float)
    y = s.values.astype(float)
    mask = np.isfinite(y)
    if mask.sum() >= 2:
        coef = np.polyfit(x[mask], y[mask], 1)
        y_pred = coef[0] * x + coef[1]
        y[~mask] = y_pred[~mask]
        return pd.Series(y, index=group.index)
    else:
        # fallback: constant mean fill
        mean = np.nanmean(y)
        if np.isnan(mean):
            return group
        return pd.Series(np.where(np.isfinite(y), y, mean), index=group.index)


def write_clean_geography(geo_dir: Path, data_dir: Path) -> Path | None:
    geo_dir = Path(geo_dir)
    data_dir = Path(data_dir)
    files = _list_geo_files(geo_dir)
    if not files:
        return None

    # Load reference for names (and set of valid ISO3 country codes)
    ref, _ = load_country_reference(data_dir)
    iso_to_name = dict(ref[["abv", "name"]].values)
    valid_iso = set(ref["abv"].astype(str).str.upper().tolist())

    # Load all datasets
    pieces: Dict[str, pd.DataFrame] = {}
    for p in files:
        key, df = _load_geo_csv(p)
        pieces[key] = df

    # Build base years 1800..2024 for all countries referenced in geography files
    years = np.arange(1800, 2025, dtype=int)
    # Only include actual countries present in country_id (exclude regions/aggregates like AFE, AFW, etc.)
    all_iso = sorted({iso for df in pieces.values() for iso in df["ISO3"].unique() if str(iso).upper() in valid_iso})
    base = pd.MultiIndex.from_product([all_iso, years], names=["ISO3", "year"]).to_frame(index=False)
    base["country_name"] = base["ISO3"].map(iso_to_name)

    # Merge each indicator onto base
    out = base.copy()
    for key, df in pieces.items():
        out = out.merge(df[["ISO3", "year", key]], on=["ISO3", "year"], how="left")

    # Interpolate linearly inside gaps for all indicators
    value_cols = [c for c in out.columns if c not in ("ISO3", "year", "country_name")]
    out = interpolate_panel(out, ["ISO3"], value_cols)

    # Take Ag_land/land_area and forest_area/land_area (%) BEFORE forward/back fill
    if "land_area" in out.columns:
        if "Ag_land" in out.columns:
            with np.errstate(invalid="ignore", divide="ignore"):
                out["Ag_land"] = (out["Ag_land"] / out["land_area"]) * 100.0
        if "forest_area" in out.columns:
            with np.errstate(invalid="ignore", divide="ignore"):
                out["forest_area"] = (out["forest_area"] / out["land_area"]) * 100.0

    # Fill to full range according to policies (forward and back to 1800–2024)
    # land_area: static forward/back fill
    if "land_area" in out.columns:
        out["land_area"] = (
            out.sort_values(["ISO3", "year"])  # type: ignore[index]
            .groupby("ISO3")["land_area"]
            .apply(lambda s: s.ffill().bfill())
            .reset_index(level=0, drop=True)
        )
    # Linear regression for the rest
    for reg_col in [c for c in ["Ag_land", "Arable_PCT", "forest_area", "precipitation", "Natural_GDP_PCT"] if c in out.columns]:
        out[reg_col] = (
            out.sort_values(["ISO3", "year"])  # type: ignore[index]
            .groupby("ISO3")[reg_col]
            .apply(lambda s: _fit_and_fill_linear(s, out.loc[s.index, "year"]))
            .reset_index(level=0, drop=True)
        )

    # Normalize per year all indicator columns
    norm_cols: List[str] = []
    for col in value_cols:
        out[col] = yearwise_min_max_norm(out[col], out["year"])
        norm_cols.append(col)

    # Compute geography index as mean of normalized indicators
    if norm_cols:
        out["index"] = out[norm_cols].mean(axis=1, skipna=True)
    else:
        out["index"] = np.nan

    # Output schema: Country, abv, year, indicators..., index
    out = out.rename(columns={"country_name": "Country", "ISO3": "abv"})
    cols = ["Country", "abv", "year"] + norm_cols + ["index"]
    out_path = data_dir / "clean_geography.csv"
    out[cols].to_csv(out_path, index=False)
    return out_path
