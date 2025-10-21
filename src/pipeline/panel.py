from pathlib import Path
from typing import Tuple, List, Optional

import pandas as pd

from src.pipeline.country_utils import standardize_country
from src.pipeline.data_loaders import (
    _safe_read,
    metric_from_filename,
    parse_military_strength_excel,
    parse_global_debt_xls,
    parse_global_debt_csv,
    parse_financial_marketcap_csv,
    parse_cofer_reserve_currency,
    parse_gdp_wdi_csv,
    normalize_dataset,
)

SUPPORTED_EXT = {".csv", ".xls", ".xlsx", ".dta"}


def summarize_schema(panel: pd.DataFrame) -> pd.DataFrame:
    return (
        panel.groupby("Metric").agg(
            n_rows=("Value", "size"),
            min_year=("Year", "min"),
            max_year=("Year", "max"),
            n_countries=("Country", pd.Series.nunique),
        ).reset_index()
    )


def collect_panel(data_root: Path) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    frames: List[pd.DataFrame] = []
    problems: List[Tuple[str, str]] = []
    reserve_path: Optional[Path] = None

    if not data_root.exists():
        return pd.DataFrame(columns=["Country", "Year", "Metric", "Value"]), [
            (str(data_root), "Data directory does not exist")
        ]

    # Recurse into subfolders to support data/<Metric>/* layouts
    for path in sorted(p for p in data_root.rglob("*") if p.is_file()):
        if path.name.startswith("~$"):
            continue
        if path.suffix.lower() not in SUPPORTED_EXT:
            continue

        # Allow directory-driven metric naming: use parent directory as a hint
        parent_name = path.parent.name
        metric = metric_from_filename(path)
        if parent_name in {
            "GDP",
            "GlobalDebt",
            "MilitaryStrength",
            "Innovation",
            "Education",
            "Competitiveness",
            "ReserveCurrency",
            "ReservePower",
            "FinancialCenters",
        }:
            metric = parent_name

        # Split Competitiveness into GCI vs component files
        if metric == "Competitiveness":
            stem_lower = path.stem.lower()
            metric = "Competitiveness_GCI" if stem_lower == "gci" else "Competitiveness_component"

        # Exclude GCI (use components only)
        if metric == "Competitiveness_GCI":
            problems.append((str(path), "Competitiveness GCI excluded; using components average"))
            continue

        # Defer COFER for later (needs GDP weighting for EUR)
        if metric == "ReserveCurrency":
            reserve_path = path
            continue

        # Financial centers (WDI market cap)
        if metric == "FinancialCenters" and path.suffix.lower() == ".csv" and path.stem.lower() == "market_capitalization":
            mc = parse_financial_marketcap_csv(path)
            if mc is None:
                problems.append((str(path), "FinancialCenters market_capitalization not parsed"))
            else:
                mc["Country"] = mc["Country"].apply(standardize_country)
                mc = mc.dropna(subset=["Country"]).copy()
                frames.append(mc)
            continue
        # Exclude stockflows source for now
        if metric == "FinancialCenters" and path.suffix.lower() == ".csv" and path.stem.lower() == "stockflows":
            problems.append((str(path), "FinancialCenters stockflows excluded"))
            continue

        # Education: prefer primary-enrollment.csv over legacy AverageYearsSchooling
        if metric == "Education":
            stem_lower = path.stem.lower()
            if stem_lower.startswith("averageyearsschooling"):
                problems.append((str(path), "Education legacy AverageYearsSchooling excluded; using primary-enrollment"))
                continue

        # Military sheet layout
        if (metric == "MilitaryStrength" and path.suffix.lower() in {".xlsx", ".xls"}) or (
            path.suffix.lower() == ".xlsx" and path.stem == "MillitaryStrength"
        ):
            ms = parse_military_strength_excel(path)
            if ms is None:
                problems.append((str(path), "Military sheet not parsed"))
                continue
            ms["Country"] = ms["Country"].apply(standardize_country)
            ms = ms.dropna(subset=["Country"]).copy()
            frames.append(ms)
            continue

        # Global debt loaders
        if path.suffix.lower() == ".xls" and (path.stem == "globalDebt1950" or metric == "GlobalDebt"):
            gd = parse_global_debt_xls(path)
            if gd is None:
                problems.append((str(path), "GG_DEBT_GDP sheet not parsed"))
            else:
                gd["Country"] = gd["Country"].apply(standardize_country)
                gd = gd.dropna(subset=["Country"]).copy()
                frames.append(gd)
            continue
        if path.suffix.lower() == ".csv" and (path.stem == "globalDebt" or metric == "GlobalDebt"):
            gd = parse_global_debt_csv(path)
            if gd is None:
                problems.append((str(path), "globalDebt.csv not parsed"))
            else:
                gd["Country"] = gd["Country"].apply(standardize_country)
                gd = gd.dropna(subset=["Country"]).copy()
                frames.append(gd)
            continue

        # GDP WDI CSV override
        if metric == "GDP" and path.suffix.lower() == ".csv" and path.stem.lower() == "gdp":
            gdp_csv = parse_gdp_wdi_csv(path)
            if gdp_csv is None:
                problems.append((str(path), "GDP WDI CSV not parsed"))
            else:
                gdp_csv["Country"] = gdp_csv["Country"].apply(standardize_country)
                gdp_csv = gdp_csv.dropna(subset=["Country"]).copy()
                frames.append(gdp_csv)
            continue

        # Generic loader
        df = _safe_read(path)
        if df is None:
            problems.append((str(path), "Failed to read"))
            continue
        norm = normalize_dataset(df, metric, path.name)
        if norm is None:
            problems.append((str(path), "Unrecognized schema; skipped"))
            continue
        norm["Country"] = norm["Country"].apply(standardize_country)
        norm = norm.dropna(subset=["Country"]).copy()
        frames.append(norm)

    if not frames and reserve_path is None:
        return pd.DataFrame(columns=["Country", "Year", "Metric", "Value"]), problems

    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["Country", "Year", "Metric", "Value"])

    # Parse COFER reserves if present, using GDP to weight EUR across core members
    if reserve_path is not None:
        try:
            reserves = parse_cofer_reserve_currency(reserve_path, panel)
            if reserves is not None and not reserves.empty:
                frames.append(reserves)
                panel = pd.concat([panel, reserves], ignore_index=True)
            else:
                problems.append((str(reserve_path), "COFER parsed empty or failed"))
        except Exception as e:
            problems.append((str(reserve_path), f"COFER parsing error: {e}"))

    panel = panel.dropna(subset=["Year"]).reset_index(drop=True)
    return panel, problems


def pivot_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    # Build wide matrix Country, Year, metric columns. Missing -> -1 sentinel per requirements.
    pivot = panel.pivot_table(index=["Country", "Year"], columns="Metric", values="Value", aggfunc="mean").reset_index()
    metric_cols = [c for c in pivot.columns if c not in {"Country", "Year"}]
    for m in metric_cols:
        pivot[m] = pivot[m].fillna(-1)
    return pivot


__all__ = ["collect_panel", "summarize_schema", "pivot_metrics"]
