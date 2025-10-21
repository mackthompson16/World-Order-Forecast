from pathlib import Path
from typing import Optional, Tuple, List
import re

import numpy as np
import pandas as pd

from src.pipeline.country_utils import standardize_country


def _safe_read(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".xls", ".xlsx"}:
            return pd.read_excel(path, sheet_name=0)
        if path.suffix.lower() == ".dta":
            return pd.read_stata(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path.name}: {e}")
        return None
    return None


def metric_from_filename(path: Path) -> str:
    base = path.stem
    mapping = {
        "MillitaryStrength": "MilitaryStrength",
        "globalDebt1950": "GlobalDebt",
        "globalDebt": "GlobalDebt",
        "GDP": "GDP",
        "ReserveCurrency": "ReserveCurrency",
        "Innovation": "Innovation",
        "Education": "Education",
        "Competitiveness": "Competitiveness",
        "FinancialCenters": "FinancialCenters",
    }
    return mapping.get(base, base)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_military_strength_excel(path: Path) -> Optional[pd.DataFrame]:
    try:
        df_raw = pd.read_excel(path, sheet_name="Constant (2023) US$", header=None)
    except Exception as e:
        print(f"[WARN] Failed to read {path.name} (military sheet): {e}")
        return None
    header_row_idx = 6
    if header_row_idx >= len(df_raw):
        header_row_idx = 0
    header = df_raw.iloc[header_row_idx]
    def count_years(s):
        return sum(bool(re.fullmatch(r"\d{4}", str(x))) for x in s)
    if count_years(header) < 3:
        for r in range(min(15, len(df_raw))):
            if count_years(df_raw.iloc[r]) >= 3:
                header_row_idx = r
                header = df_raw.iloc[r]
                break
    data = df_raw.iloc[header_row_idx + 1:].copy()
    data.columns = header.values
    first_col = data.columns[0]
    data.rename(columns={first_col: "Country"}, inplace=True)
    data = data[~data["Country"].isna()].copy()
    year_cols = [c for c in data.columns if re.fullmatch(r"\d{4}", str(c))]
    if not year_cols:
        return None
    long_df = data.melt(id_vars=["Country"], value_vars=year_cols, var_name="Year", value_name="Value")
    long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")
    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
    long_df.dropna(subset=["Year"], inplace=True)
    long_df["Metric"] = "MilitaryStrength"
    return long_df[["Country", "Year", "Value", "Metric"]]


def parse_global_debt_xls(path: Path) -> Optional[pd.DataFrame]:
    try:
        df_raw = pd.read_excel(path, sheet_name="GG_DEBT_GDP", header=None)
    except Exception as e:
        print(f"[WARN] Failed to read {path.name} (GG_DEBT_GDP): {e}")
        return None
    def count_years(s):
        return sum(bool(re.fullmatch(r"\d{4}", str(x))) for x in s)
    header_row_idx = None
    for r in range(min(20, len(df_raw))):
        if count_years(df_raw.iloc[r]) >= 3:
            header_row_idx = r
            break
    if header_row_idx is None:
        header_row_idx = 0
    header = df_raw.iloc[header_row_idx]
    data = df_raw.iloc[header_row_idx + 1:].copy()
    data.columns = header.values
    first_col = data.columns[0]
    data.rename(columns={first_col: "Country"}, inplace=True)
    data = data[~data["Country"].isna()].copy()
    year_cols = [c for c in data.columns if re.fullmatch(r"\d{4}", str(c))]
    if not year_cols:
        return None
    long_df = data.melt(id_vars=["Country"], value_vars=year_cols, var_name="Year", value_name="Value")
    long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")
    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
    long_df.dropna(subset=["Year"], inplace=True)
    long_df["Metric"] = "GlobalDebt"
    return long_df[["Country", "Year", "Value", "Metric"]]


def parse_global_debt_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df_raw = pd.read_csv(path, header=None)
    except Exception as e:
        print(f"[WARN] Failed to read {path.name}: {e}")
        return None
    if df_raw.empty:
        return None
    header = df_raw.iloc[0]
    data = df_raw.iloc[1:].copy()
    data.columns = header.values
    first_col = data.columns[0]
    data.rename(columns={first_col: "Country"}, inplace=True)
    data = data[~data["Country"].isna()].copy()
    year_cols = []
    col_map = {}
    for c in data.columns[1:]:
        s = str(c).strip()
        if re.fullmatch(r"\d{4}", s):
            year_cols.append(c)
            col_map[c] = int(s)
        else:
            m = re.search(r"(\d{4})", s)
            if m:
                y = int(m.group(1))
                year_cols.append(c)
                col_map[c] = y
    if not year_cols:
        print(f"[INFO] {path.name}: no year-like columns detected after header row")
        return None
    data = data.rename(columns=col_map)
    use_cols = ["Country"] + sorted(set(col_map.values()))
    data = data[use_cols]
    long_df = data.melt(id_vars=["Country"], var_name="Year", value_name="Value")
    long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")
    long_df["Value"] = pd.to_numeric(long_df["Value"].replace({"no data": np.nan}), errors="coerce")
    long_df.dropna(subset=["Year"], inplace=True)
    long_df["Metric"] = "GlobalDebt"
    return long_df[["Country", "Year", "Value", "Metric"]]


def parse_financial_stockflows_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path.name}: {e}")
        return None
    if df.empty:
        return None
    country_col = None
    for c in ["Economy_Label", "Country", "Country Name", "Entity"]:
        if c in df.columns:
            country_col = c
            break
    if country_col is None:
        print(f"[INFO] {path.name}: no country-like column detected")
        return None
    year_map = {}
    for c in df.columns:
        m = re.match(r"(\d{4})_Percentage_of_total_world_Value$", str(c))
        if m:
            y = int(m.group(1))
            year_map[c] = y
    if not year_map:
        print(f"[INFO] {path.name}: no year Value columns detected")
        return None
    sub = df[[country_col] + list(year_map.keys())].copy()
    sub.rename(columns={country_col: "Country"}, inplace=True)
    sub = sub.rename(columns=year_map)
    long_df = sub.melt(id_vars=["Country"], var_name="Year", value_name="Value")
    long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")
    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
    long_df = long_df.dropna(subset=["Year"]).reset_index(drop=True)
    long_df["Metric"] = "FinancialCenters"
    return long_df[["Country", "Year", "Value", "Metric"]]


def parse_financial_marketcap_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path.name}: {e}")
        return None
    if df.empty:
        return None
    country_col = None
    for c in ["Country Name", "Country", "Entity"]:
        if c in df.columns:
            country_col = c
            break
    if country_col is None:
        print(f"[INFO] {path.name}: no 'Country Name' column; cannot parse")
        return None
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    if not year_cols:
        print(f"[INFO] {path.name}: no 4-digit year columns detected")
        return None
    sub = df[[country_col] + year_cols].copy()
    sub.rename(columns={country_col: "Country"}, inplace=True)
    long_df = sub.melt(id_vars=["Country"], value_vars=year_cols, var_name="Year", value_name="Value")
    long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")
    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
    long_df = long_df.dropna(subset=["Year"]).reset_index(drop=True)
    long_df["Metric"] = "FinancialCenters"
    return long_df[["Country", "Year", "Value", "Metric"]]


def parse_gdp_wdi_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read {path.name}: {e}")
        return None
    if df.empty:
        return None
    country_col = None
    for c in ["Country Name", "Country", "Entity"]:
        if c in df.columns:
            country_col = c
            break
    if country_col is None:
        print(f"[INFO] {path.name}: no 'Country Name' column; cannot parse")
        return None
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    if not year_cols:
        print(f"[INFO] {path.name}: no 4-digit year columns detected")
        return None
    sub = df[[country_col] + year_cols].copy()
    sub.rename(columns={country_col: "Country"}, inplace=True)
    long_df = sub.melt(id_vars=["Country"], value_vars=year_cols, var_name="Year", value_name="Value")
    long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")
    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
    long_df = long_df.dropna(subset=["Year"]).reset_index(drop=True)
    long_df["Metric"] = "GDP"
    return long_df[["Country", "Year", "Value", "Metric"]]


def parse_cofer_reserve_currency(path: Path, panel_current: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"[WARN] Failed to read {path.name}: {e}")
        return None
    cols = list(df.columns)
    year_cols = [c for c in cols if re.fullmatch(r"\d{4}", str(c))]
    q_cols = [c for c in cols if re.fullmatch(r"\d{4}-Q[1-4]", str(c))]
    if not year_cols and not q_cols:
        print(f"[INFO] COFER: no year/quarter columns detected in {path.name}")
        return None
    if "SERIES_CODE" not in df.columns:
        print("[INFO] COFER: SERIES_CODE not found; cannot identify currency rows")
        return None
    wanted = {
        "CI_USD": ("USD", "United States"),
        "CI_GBP": ("GBP", "United Kingdom"),
        "CI_JPY": ("JPY", "Japan"),
        "CI_CHF": ("CHF", "Switzerland"),
        "CI_CAD": ("CAD", "Canada"),
        "CI_AUD": ("AUD", "Australia"),
        "CI_CNY": ("CNY", "China"),
        "CI_EUR": ("EUR", None),
    }
    rows = {}
    for key in wanted.keys():
        mask = df["SERIES_CODE"].astype(str).str.contains(key) & df["SERIES_CODE"].astype(str).str.contains("SHRO_PT")
        sub = df[mask]
        if sub.empty:
            continue
        rows[key] = sub.iloc[0]
    if not rows:
        print("[INFO] COFER: No currency share rows found")
        return None
    years = sorted({int(c) for c in year_cols}) if year_cols else []
    years_from_q = sorted({int(c.split('-')[0]) for c in q_cols}) if q_cols else []
    years = sorted(set(years) | set(years_from_q))
    if not years:
        return None
    share_by_year = {y: {} for y in years}
    for key, row in rows.items():
        for y in years:
            val = np.nan
            if str(y) in df.columns:
                try:
                    v = pd.to_numeric(row[str(y)], errors="coerce")
                    if pd.notna(v):
                        val = float(v)
                except Exception:
                    pass
            if pd.isna(val):
                qs = [f"{y}-Q{q}" for q in (4, 3, 2, 1)]
                found = False
                for qc in qs:
                    if qc in df.columns:
                        v = pd.to_numeric(row[qc], errors="coerce")
                        if pd.notna(v):
                            val = float(v)
                            found = True
                            break
                if not found:
                    vals = []
                    for qc in qs:
                        if qc in df.columns:
                            vv = pd.to_numeric(row[qc], errors="coerce")
                            if pd.notna(vv):
                                vals.append(float(vv))
                    if vals:
                        val = float(np.mean(vals))
            if pd.notna(val):
                share_by_year[y][key] = val
    norm_yearly = {}
    for y, shares in share_by_year.items():
        if not shares:
            continue
        leader = max(shares.values())
        if leader <= 0 or not np.isfinite(leader):
            continue
        norm_yearly[y] = {k: v / leader for k, v in shares.items()}
    if not norm_yearly:
        return None
    eur_members = ["Germany", "France", "Italy", "Spain", "Netherlands"]
    gdp = (
        panel_current[panel_current["Metric"] == "GDP"][['Country', 'Year', 'Value']].copy()
        if not panel_current.empty
        else pd.DataFrame(columns=['Country', 'Year', 'Value'])
    )
    gdp_pivot = None
    if not gdp.empty:
        gdp_pivot = gdp.pivot_table(index='Year', columns='Country', values='Value', aggfunc='mean')
    rows_out = []
    for y, shares in norm_yearly.items():
        for key, (code, country) in wanted.items():
            if key == 'CI_EUR':
                continue
            if key not in shares or country is None:
                continue
            rows_out.append({"Country": country, "Year": y, "Value": shares[key], "Metric": "ReservePower"})
        if 'CI_EUR' in shares:
            eur_val = shares['CI_EUR']
            if gdp_pivot is not None and y in gdp_pivot.index:
                vals = gdp_pivot.loc[y, eur_members]
                vals = vals.dropna()
                if not vals.empty and float(vals.sum()) > 0:
                    weights = vals / float(vals.sum())
                    for ctry, w in weights.items():
                        rows_out.append({"Country": ctry, "Year": y, "Value": eur_val * float(w), "Metric": "ReservePower"})
                else:
                    w = 1.0 / len(eur_members)
                    for ctry in eur_members:
                        rows_out.append({"Country": ctry, "Year": y, "Value": eur_val * w, "Metric": "ReservePower"})
            else:
                w = 1.0 / len(eur_members)
                for ctry in eur_members:
                    rows_out.append({"Country": ctry, "Year": y, "Value": eur_val * w, "Metric": "ReservePower"})
    if not rows_out:
        return None
    out = pd.DataFrame(rows_out)
    out["Country"] = out["Country"].apply(standardize_country)
    out = out.dropna(subset=["Country"]).copy()
    return out[["Country", "Year", "Value", "Metric"]]


def normalize_dataset(df: pd.DataFrame, metric_name: str, source_file: str) -> Optional[pd.DataFrame]:
    df = _normalize_columns(df)
    cols = list(df.columns)
    country_col = None
    for c in [
        "Country",
        "Entity",
        "Name",
        "REF_AREA_LABEL",
        "REF_AREA",
        "Code",
        "LOCATION",
        "country",
        "entity",
        "name",
        "ref_area_label",
        "ref_area",
        "code",
        "location",
    ]:
        if c in cols:
            country_col = c
            break
    year_long = None
    for c in ["Year", "TIME_PERIOD", "Date", "year", "time", "date"]:
        if c in cols:
            year_long = c
            break
    year_wide = [c for c in cols if re.fullmatch(r"\d{4}", str(c))]

    # Competitiveness (WEF Data360 style)
    if metric_name.lower() == "competitiveness":
        if {"REF_AREA_LABEL", "TIME_PERIOD", "OBS_VALUE"}.issubset(df.columns):
            temp = df[["REF_AREA_LABEL", "TIME_PERIOD", "OBS_VALUE"]].copy()
            temp.rename(
                columns={"REF_AREA_LABEL": "Country", "TIME_PERIOD": "Year", "OBS_VALUE": "Value"},
                inplace=True,
            )
            temp["Metric"] = metric_name
            return temp[["Country", "Year", "Value", "Metric"]]

    if (country_col is None) and ("Code" in df.columns or "REF_AREA" in df.columns):
        country_col = "Code" if "Code" in df.columns else "REF_AREA"
    if country_col is None:
        return None
    if year_wide and len(year_wide) >= 3:
        tmp = df[[country_col] + year_wide].copy()
        tmp.rename(columns={country_col: "Country"}, inplace=True)
        long_df = tmp.melt(id_vars=["Country"], value_vars=year_wide, var_name="Year", value_name="Value")
        long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")
        long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
        long_df.dropna(subset=["Year"], inplace=True)
        long_df["Metric"] = metric_name
        return long_df[["Country", "Year", "Value", "Metric"]]
    if year_long and year_long in df.columns:
        value_col = None
        for c in ["Value", "OBS_VALUE", "value"]:
            if c in df.columns:
                value_col = c
                break
        if value_col is None:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            value_col = numeric_cols[-1] if numeric_cols else None
        if value_col is None:
            return None
        temp = df[[country_col, year_long, value_col]].copy()
        temp.rename(columns={country_col: "Country", year_long: "Year", value_col: "Value"}, inplace=True)
        temp["Year"] = pd.to_numeric(temp["Year"], errors="coerce")
        temp["Value"] = pd.to_numeric(temp["Value"], errors="coerce")
        temp.dropna(subset=["Year"], inplace=True)
        temp["Metric"] = metric_name
        return temp[["Country", "Year", "Value", "Metric"]]
    return None


__all__ = [
    "_safe_read",
    "metric_from_filename",
    "parse_military_strength_excel",
    "parse_global_debt_xls",
    "parse_global_debt_csv",
    "parse_financial_stockflows_csv",
    "parse_financial_marketcap_csv",
    "parse_gdp_wdi_csv",
    "parse_cofer_reserve_currency",
    "normalize_dataset",
]
