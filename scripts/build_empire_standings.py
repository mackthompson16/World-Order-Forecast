import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def main(data_dir: Optional[str] = None) -> None:
    data_root = Path(data_dir or os.getenv("DATA_DIR", "data")).resolve()
    results_dir = Path("results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Data root: {data_root}")
    panel, problems = collect_panel(data_root)
    if not panel.empty:
        try:
            print(f"[INFO] Collected panel: rows={len(panel)}, metrics={panel['Metric'].nunique()}, countries={panel['Country'].nunique()}")
        except Exception:
            pass
    if panel.empty:
        print(f"No data parsed from {data_root}. See warnings: {results_dir / 'parsing_warnings.csv'}")
        if problems:
            pd.DataFrame(problems, columns=["file", "note"]).to_csv(results_dir / "parsing_warnings.csv", index=False)
        return

    # Persist schema + country list
    summarize_schema(panel).to_csv(results_dir / "parsed_schema_summary.csv", index=False)
    (panel[["Country"]].drop_duplicates().sort_values("Country")).to_csv(results_dir / "countries_list.csv", index=False)

    # Build wide, mark missing as -1 for raw metrics
    wide = pivot_metrics(panel)
    # Coverage: unique countries with non-missing per metric per year (no double-counting subpages)
    coverage = compute_coverage_from_wide(wide)
    coverage.to_csv(results_dir / "data_coverage_by_metric_year.csv", index=False)
    # Only keep the combined coverage plot (remove per-metric plots if any exist)
    plot_coverage_combined(coverage, Path("coverage_all_metrics_counts.png"), logy=True)
    try:
        for p in results_dir.glob("coverage_*_counts.png"):
            if p.name != "coverage_all_metrics_counts.png":
                p.unlink(missing_ok=True)
    except Exception:
        pass

    # Compute normalized metrics per year (min-max) and composite
    composite = compute_composite(wide)
    composite.to_csv(results_dir / "empire_composite.csv", index=False)

    # Rank Top 5 by area with dynamic start year (>=3 metrics and GDP present)
    top_series, areas = select_top5_by_area(composite)
    areas.to_csv(results_dir / "country_area_ranking.csv", index=False)
    plot_top(top_series, areas, Path("empire_standings_top5.png"))

    # Winner summaries
    winners_by_metric = compute_winners_by_metric_year(composite)
    winners_by_metric.to_csv(results_dir / "winners_by_metric_year.csv", index=False)
    top5_by_year = compute_top5_by_year(composite)
    top5_by_year.to_csv(results_dir / "top5_by_year.csv", index=False)

    if problems:
        pd.DataFrame(problems, columns=["file", "note"]).to_csv(results_dir / "parsing_warnings.csv", index=False)
        # Also echo warnings to console
        print("[WARN] Parsing issues:")
        for f, n in problems:
            print(f"  - {f}: {n}")

    print("[INFO] Done. CSVs in results/, images at repo root.")


########################################
# Parsing helpers
########################################

SUPPORTED_EXT = {".csv", ".xls", ".xlsx", ".dta"}


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


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


COUNTRY_ALIASES = {
    "united states of america": "United States",
    "united states": "United States",
    "usa": "United States",
    "us": "United States",
    "u.s.": "United States",
    "uk": "United Kingdom",
    "united kingdom of great britain and northern ireland": "United Kingdom",
    "russian federation": "Russia",
    "korea, rep.": "South Korea",
    "korea, republic of": "South Korea",
    "korea, dpr": "North Korea",
    "korea, dem. people's rep.": "North Korea",
    "czech republic": "Czechia",
    "iran, islamic rep.": "Iran",
    "egypt, arab rep.": "Egypt",
    "viet nam": "Vietnam",
    "hong kong sar, china": "Hong Kong",
    "macao sar, china": "Macao",
    "china, mainland": "China",
    "taiwan, china": "Taiwan",
    "kyrgyz republic": "Kyrgyzstan",
    "lao pdr": "Laos",
    "congo, dem. rep.": "DR Congo",
    "congo, rep.": "Congo",
    "gambia, the": "Gambia",
    "yemen, rep.": "Yemen",
    "syrian arab republic": "Syria",
    "bahamas, the": "Bahamas",
    "brunei darussalam": "Brunei",
    "cabo verde": "Cape Verde",
    "swaziland": "Eswatini",
    "north macedonia": "North Macedonia",
}

AGGREGATE_PREFIXES = [
    "world", "europe", "asia", "africa", "oceania", "latin america",
    "europe & central asia", "middle east", "north america", "euro area",
    "oecd", "g7", "g20", "high income", "upper middle income", "lower middle income",
    "low income", "arab world", "caribbean", "sub-saharan africa", "south asia",
    "east asia & pacific", "european union", "commonwealth", "former ussr",
]


def standardize_country(name: str) -> Optional[str]:
    if not isinstance(name, str):
        return None
    n = name.strip()
    if not n:
        return None
    key = n.lower()
    if key in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[key]
    for pref in AGGREGATE_PREFIXES:
        if key.startswith(pref):
            return None
    n = re.sub(r"\s*,\s*total$", "", n, flags=re.I)
    n = re.sub(r"\s*\(.*?\)\s*", "", n).strip()
    return n


def metric_from_filename(path: Path) -> str:
    mapping = {
        "MillitaryStrength": "MilitaryStrength",
        "globalDebt1950": "GlobalDebt",
        "globalDebt": "GlobalDebt",
        "GDP": "GDP",
        "ReserveCurrency": "ReserveCurrency",
        "Innovation": "Innovation",
        "Education": "Education",
        "Competitiveness": "Competitiveness",
    }
    return mapping.get(path.stem, path.stem)


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
    """Parse Global Debt from sheet GG_DEBT_GDP in an .xls workbook.

    Heuristic similar to the military parser: detect a header row that contains
    multiple 4-digit years, first column is country names, then melt.
    """
    # Try with explicit assumption: row 0 is header (years), col 0 is Country
    try:
        df_hdr0 = pd.read_excel(path, sheet_name="GG_DEBT_GDP", header=0)
        # Ensure first column is Country
        first_col = df_hdr0.columns[0]
        df_hdr0 = df_hdr0.rename(columns={first_col: "Country"})
        df_hdr0 = df_hdr0[~df_hdr0["Country"].isna()].copy()
        # Select year columns (4-digit or numeric-like)
        year_cols = []
        for c in df_hdr0.columns[1:]:
            sc = str(c).strip()
            if re.fullmatch(r"\d{4}", sc):
                year_cols.append(c)
            else:
                # Sometimes headers are numeric types already
                try:
                    cv = int(float(sc))
                    if 1000 <= cv <= 3000:
                        year_cols.append(c)
                except Exception:
                    pass
        if not year_cols:
            print(f"[INFO] GG_DEBT_GDP in {path.name}: header row present but no year-like columns detected")
            return None
        long_df = df_hdr0.melt(id_vars=["Country"], value_vars=year_cols, var_name="Year", value_name="Value")
        long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")
        long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
        long_df.dropna(subset=["Year"], inplace=True)
        long_df["Metric"] = "GlobalDebt"
        return long_df[["Country", "Year", "Value", "Metric"]]
    except Exception as e:
        print(f"[WARN] Failed to read {path.name} (GG_DEBT_GDP with header=0): {e}")
        return None


def parse_global_debt_csv(path: Path) -> Optional[pd.DataFrame]:
    """Parse a wide CSV where row 0 has years and column 0 has countries.

    Assumes first row contains header with many 4-digit years and the first
    column header is a descriptive label (replaced by 'Country').
    """
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
    # First column becomes Country
    first_col = data.columns[0]
    data.rename(columns={first_col: "Country"}, inplace=True)
    data = data[~data["Country"].isna()].copy()
    # Determine year columns from headers
    year_cols = []
    col_map = {}
    for c in data.columns[1:]:
        s = str(c).strip()
        if re.fullmatch(r"\d{4}", s):
            year_cols.append(c)
            col_map[c] = int(s)
        else:
            # try to coerce numeric or extract 4-digit substring
            m = re.search(r"(\d{4})", s)
            if m:
                y = int(m.group(1))
                if 1000 <= y <= 3000:
                    year_cols.append(c)
                    col_map[c] = y
            else:
                try:
                    y = int(float(s))
                    if 1000 <= y <= 3000:
                        year_cols.append(c)
                        col_map[c] = y
                except Exception:
                    pass
    if not year_cols:
        print(f"[INFO] {path.name}: no year-like columns detected after header row")
        return None
    # Rename selected year columns to numeric year labels
    data = data.rename(columns=col_map)
    use_cols = ["Country"] + sorted(set(col_map.values()))
    data = data[use_cols]
    long_df = data.melt(id_vars=["Country"], var_name="Year", value_name="Value")
    long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")
    long_df["Value"] = pd.to_numeric(long_df["Value"].replace({"no data": np.nan}), errors="coerce")
    long_df.dropna(subset=["Year"], inplace=True)
    long_df["Metric"] = "GlobalDebt"
    return long_df[["Country", "Year", "Value", "Metric"]]


def parse_cofer_reserve_currency(path: Path, panel_current: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Parse IMF COFER CSV and compute a country-level ReservePower metric.

    Steps:
    - Identify rows for currency share series (e.g., CI_USD, CI_EUR, ... SHRO_PT)
    - Extract yearly values (prefer column 'YYYY'; fallback to avg of 'YYYY-Q*')
    - For each year compute leader share; normalize shares by leader (leader=1.0)
    - Map currencies to issuing countries; EUR is GDP-weighted across core members
    """
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"[WARN] Failed to read {path.name}: {e}")
        return None

    # Find columns that are years or quarters
    cols = list(df.columns)
    year_cols = [c for c in cols if re.fullmatch(r"\d{4}", str(c))]
    q_cols = [c for c in cols if re.fullmatch(r"\d{4}-Q[1-4]", str(c))]
    if not year_cols and not q_cols:
        print(f"[INFO] COFER: no year/quarter columns detected in {path.name}")
        return None

    # Filter to share series rows; rely on SERIES_CODE containing currency code and 'SHRO_PT'
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
        "CI_EUR": ("EUR", None),  # special case
    }

    rows = {}
    for key in wanted.keys():
        mask = df["SERIES_CODE"].astype(str).str.contains(key) & df["SERIES_CODE"].astype(str).str.contains("SHRO_PT")
        sub = df[mask]
        if sub.empty:
            continue
        # Take the first matching row
        rows[key] = sub.iloc[0]

    if not rows:
        print("[INFO] COFER: No currency share rows found")
        return None

    # Build year -> currency -> value dict
    years = sorted({int(c) for c in year_cols}) if year_cols else []
    # Also include years inferred from quarters
    years_from_q = sorted({int(c.split('-')[0]) for c in q_cols}) if q_cols else []
    years = sorted(set(years) | set(years_from_q))
    if not years:
        return None

    share_by_year = {y: {} for y in years}
    for key, row in rows.items():
        # Gather per-year value
        for y in years:
            val = np.nan
            # Prefer exact yearly column if present
            if str(y) in df.columns:
                try:
                    v = pd.to_numeric(row[str(y)], errors="coerce")
                    if pd.notna(v):
                        val = float(v)
                except Exception:
                    pass
            # If missing, try average of quarters; prefer Q4 if available
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
                    # As last resort, mean of available quarters
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

    # Normalize by leading currency each year
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

    # GDP-weight EUR across core members
    eur_members = ["Germany", "France", "Italy", "Spain", "Netherlands"]
    gdp = panel_current[panel_current["Metric"] == "GDP"][['Country','Year','Value']].copy() if not panel_current.empty else pd.DataFrame(columns=['Country','Year','Value'])
    gdp_pivot = None
    if not gdp.empty:
        gdp_pivot = gdp.pivot_table(index='Year', columns='Country', values='Value', aggfunc='mean')

    rows_out = []
    for y, shares in norm_yearly.items():
        # Non-EUR currencies map 1:1 to issuing country with their normalized share
        for key, (code, country) in wanted.items():
            if key == 'CI_EUR':
                continue
            if key not in shares or country is None:
                continue
            rows_out.append({"Country": country, "Year": y, "Value": shares[key], "Metric": "ReservePower"})
        # EUR allocation
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
                    # Equal split if GDP missing
                    w = 1.0 / len(eur_members)
                    for ctry in eur_members:
                        rows_out.append({"Country": ctry, "Year": y, "Value": eur_val * w, "Metric": "ReservePower"})
            else:
                # Equal split if no GDP pivot
                w = 1.0 / len(eur_members)
                for ctry in eur_members:
                    rows_out.append({"Country": ctry, "Year": y, "Value": eur_val * w, "Metric": "ReservePower"})

    if not rows_out:
        return None
    out = pd.DataFrame(rows_out)
    # Standardize names
    out["Country"] = out["Country"].apply(standardize_country)
    out = out.dropna(subset=["Country"]).copy()
    return out[["Country", "Year", "Value", "Metric"]]


def normalize_dataset(df: pd.DataFrame, metric_name: str, source_file: str) -> Optional[pd.DataFrame]:
    df = _normalize_columns(df)
    cols = list(df.columns)

    # Competitiveness: WEF Data360-like
    if metric_name.lower() == "competitiveness":
        if {"REF_AREA_LABEL", "TIME_PERIOD", "OBS_VALUE"}.issubset(df.columns):
            temp = df[["REF_AREA_LABEL", "TIME_PERIOD", "OBS_VALUE"]].copy()
            temp.rename(columns={"REF_AREA_LABEL": "Country", "TIME_PERIOD": "Year", "OBS_VALUE": "Value"}, inplace=True)
            temp["Metric"] = metric_name
            return temp[["Country", "Year", "Value", "Metric"]]

    # Generic: detect country/year/value
    country_col = None
    for c in [
        "Country", "Entity", "Name", "REF_AREA_LABEL", "REF_AREA", "Code", "LOCATION",
        "country", "entity", "name", "ref_area_label", "ref_area", "code", "location",
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

    if country_col is None:
        return None

    # Wide to long
    if len(year_wide) >= 3:
        tmp = df[[country_col] + year_wide].copy()
        tmp.rename(columns={country_col: "Country"}, inplace=True)
        long_df = tmp.melt(id_vars=["Country"], value_vars=year_wide, var_name="Year", value_name="Value")
        long_df["Year"] = pd.to_numeric(long_df["Year"], errors="coerce")
        long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
        long_df.dropna(subset=["Year"], inplace=True)
        long_df["Metric"] = metric_name
        return long_df[["Country", "Year", "Value", "Metric"]]

    # Long format
    if year_long and year_long in df.columns:
        # Pick last numeric as value
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        value_col = None
        for c in ["Value", "OBS_VALUE", "value"]:
            if c in df.columns:
                value_col = c
                break
        if value_col is None:
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


def collect_panel(data_root: Path) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    frames: List[pd.DataFrame] = []
    problems: List[Tuple[str, str]] = []
    reserve_path: Optional[Path] = None

    if not data_root.exists():
        return pd.DataFrame(columns=["Country", "Year", "Metric", "Value"]), [(str(data_root), "Data directory does not exist")] 

    # Recurse into subfolders to support data/<Metric>/* layouts
    for path in sorted(p for p in data_root.rglob("*") if p.is_file()):
        if path.name.startswith("~$"):
            continue
        if path.suffix.lower() not in SUPPORTED_EXT:
            continue

        # Allow directory-driven metric naming: use parent directory as a hint
        parent_name = path.parent.name
        metric = metric_from_filename(path)
        # Prefer the parent directory if it looks like a known metric label
        if parent_name in {"GDP","GlobalDebt","MilitaryStrength","Innovation","Education","Competitiveness","ReserveCurrency","ReservePower"}:
            metric = parent_name
        print(f"[PARSE] {path} -> metric={metric}")
        if metric == "ReserveCurrency":
            # Defer parsing until after other data so we can GDP-weight EUR
            reserve_path = path
            print(f"[PARSE] Deferring COFER (ReserveCurrency) for later: {path.name}")
            continue

        # special-case military sheet
        if (metric == "MilitaryStrength" and path.suffix.lower() in {".xlsx", ".xls"}) or (path.suffix.lower() == ".xlsx" and path.stem == "MillitaryStrength"):
            ms = parse_military_strength_excel(path)
            if ms is None:
                problems.append((str(path), "Military sheet not parsed"))
                print(f"[WARN] MilitaryStrength not parsed: {path}")
                continue
            ms["Country"] = ms["Country"].apply(standardize_country)
            ms = ms.dropna(subset=["Country"]).copy()
            frames.append(ms)
            try:
                yrs = pd.to_numeric(ms["Year"], errors="coerce").dropna()
                print(f"[OK] MilitaryStrength parsed: rows={len(ms)} years=[{int(yrs.min())}-{int(yrs.max())}] countries={ms['Country'].nunique()}")
            except Exception:
                print(f"[OK] MilitaryStrength parsed: rows={len(ms)}")
            continue

        # special-case global debt xls sheet
        if path.suffix.lower() == ".xls" and (path.stem == "globalDebt1950" or metric == "GlobalDebt"):
            gd = parse_global_debt_xls(path)
            if gd is None:
                problems.append((str(path), "GG_DEBT_GDP sheet not parsed"))
                print(f"[WARN] GlobalDebt XLS not parsed: {path}")
            else:
                gd["Country"] = gd["Country"].apply(standardize_country)
                gd = gd.dropna(subset=["Country"]).copy()
                frames.append(gd)
                try:
                    yrs = pd.to_numeric(gd["Year"], errors="coerce").dropna()
                    print(f"[OK] GlobalDebt (xls) parsed: rows={len(gd)} years=[{int(yrs.min())}-{int(yrs.max())}] countries={gd['Country'].nunique()}")
                except Exception:
                    print(f"[OK] GlobalDebt (xls) parsed: rows={len(gd)}")
                continue

        # special-case global debt csv
        if path.suffix.lower() == ".csv" and (path.stem == "globalDebt" or metric == "GlobalDebt"):
            gd = parse_global_debt_csv(path)
            if gd is None:
                problems.append((str(path), "globalDebt.csv not parsed"))
                print(f"[WARN] GlobalDebt CSV not parsed: {path}")
                continue
            gd["Country"] = gd["Country"].apply(standardize_country)
            gd = gd.dropna(subset=["Country"]).copy()
            frames.append(gd)
            try:
                yrs = pd.to_numeric(gd["Year"], errors="coerce").dropna()
                print(f"[OK] GlobalDebt (csv) parsed: rows={len(gd)} years=[{int(yrs.min())}-{int(yrs.max())}] countries={gd['Country'].nunique()}")
            except Exception:
                print(f"[OK] GlobalDebt (csv) parsed: rows={len(gd)}")
            continue

        df = _safe_read(path)
        if df is None:
            problems.append((str(path), "Failed to read"))
            print(f"[WARN] Failed to read: {path}")
            continue
        norm = normalize_dataset(df, metric, path.name)
        if norm is None:
            problems.append((str(path), "Unrecognized schema; skipped"))
            print(f"[WARN] Unrecognized schema; skipped: {path} (metric={metric})")
            continue
        norm["Country"] = norm["Country"].apply(standardize_country)
        norm = norm.dropna(subset=["Country"]).copy()
        frames.append(norm)
        try:
            yrs = pd.to_numeric(norm["Year"], errors="coerce").dropna()
            print(f"[OK] {metric} parsed: file={path.name} rows={len(norm)} years=[{int(yrs.min())}-{int(yrs.max())}] countries={norm['Country'].nunique()}")
        except Exception:
            print(f"[OK] {metric} parsed: file={path.name} rows={len(norm)}")

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
                try:
                    yrs = pd.to_numeric(reserves["Year"], errors="coerce").dropna()
                    print(f"[OK] ReservePower parsed: rows={len(reserves)} years=[{int(yrs.min())}-{int(yrs.max())}] countries={reserves['Country'].nunique()}")
                except Exception:
                    print(f"[OK] ReservePower parsed: rows={len(reserves)}")
            else:
                problems.append((str(reserve_path), "COFER parsed empty or failed"))
                print(f"[WARN] COFER parsed empty or failed: {reserve_path}")
        except Exception as e:
            problems.append((str(reserve_path), f"COFER parsing error: {e}"))
            print(f"[ERROR] COFER parsing error: {e}")
    panel = panel.dropna(subset=["Year"]).reset_index(drop=True)
    try:
        print("[INFO] Rows per metric after collection:")
        for m, n in panel.groupby("Metric").size().sort_values(ascending=False).items():
            print(f"  - {m}: {n}")
    except Exception:
        pass
    return panel, problems


########################################
# Aggregation & normalization
########################################

def pivot_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    # Build wide matrix Country, Year, metric columns. Missing -> -1 sentinel per requirements.
    pivot = panel.pivot_table(index=["Country", "Year"], columns="Metric", values="Value", aggfunc="mean").reset_index()
    metric_cols = [c for c in pivot.columns if c not in {"Country", "Year"}]
    for m in metric_cols:
        pivot[m] = pivot[m].fillna(-1)
    try:
        print("[INFO] Wide table non-missing counts per metric (value != -1):")
        for m in metric_cols:
            print(f"  - {m}: {(pivot[m] != -1).sum()}")
    except Exception:
        pass
    return pivot


def compute_composite(wide: pd.DataFrame) -> pd.DataFrame:
    df = wide.copy()
    metric_cols = [c for c in df.columns if c not in {"Country", "Year"}]

    # All-time min-max normalization per metric across all years, skipping -1 values.
    lower_is_better = {"GlobalDebt"}
    print(f"[INFO] Normalizing metrics all-time (min-max across all years). Metrics: {', '.join(metric_cols)}")
    for m in metric_cols:
        norm_col = f"{m}_norm"
        df[norm_col] = np.nan
        s_all = pd.to_numeric(df[m], errors="coerce")
        valid_mask_all = s_all.ge(0) & s_all.notna()
        if not valid_mask_all.any():
            print(f"[WARN] No valid values for metric {m}; normalized column will remain NaN")
            continue
        v = s_all.where(valid_mask_all)
        vmin, vmax = v.min(), v.max()
        if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
            # no variation -> neutral 0.5 where valid
            df.loc[valid_mask_all, norm_col] = 0.5
        else:
            scaled_all = (v - vmin) / (vmax - vmin)
            if m in lower_is_better:
                scaled_all = 1.0 - scaled_all
            df.loc[valid_mask_all, norm_col] = scaled_all
    try:
        norm_cols = [c for c in df.columns if c.endswith('_norm')]
        print("[INFO] Normalized coverage (rows with non-NaN) and min/max per metric:")
        for nc in norm_cols:
            base = nc[:-5]
            cnt = df[nc].notna().sum()
            mn = pd.to_numeric(df[nc], errors='coerce').min()
            mx = pd.to_numeric(df[nc], errors='coerce').max()
            print(f"  - {nc}: count={cnt} min={mn} max={mx}")
    except Exception:
        pass

    norm_cols = [c for c in df.columns if c.endswith("_norm")]
    # Composite: average only available (non-negative raw -> valid normalized). No penalty; strictly exclude missing (-1).
    df["AvailableCount"] = df[[c.replace("_norm", "") for c in norm_cols]].ge(0).sum(axis=1)
    df["CompositeStanding"] = df[norm_cols].mean(axis=1, skipna=True)

    keep_cols = ["Country", "Year", "CompositeStanding", "AvailableCount"] + metric_cols + norm_cols
    return df[keep_cols]


def select_top5_by_area(composite: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Dynamic start year: require GDP present and at least 3 metrics available
    csel = composite.copy()
    csel["Year"] = pd.to_numeric(csel["Year"], errors="coerce")
    csel = csel.dropna(subset=["Year"])  # numeric years only
    if "GDP_norm" in csel.columns:
        csel = csel[csel["GDP_norm"].notna()].copy()
    if "AvailableCount" in csel.columns:
        csel = csel[csel["AvailableCount"] >= 3].copy()
    # Compute areas over the dynamically filtered window
    # Rank by average CompositeStanding over the filtered window
    avgs = csel.groupby("Country")["CompositeStanding"].mean().sort_values(ascending=False)
    top_countries = avgs.head(5).index.tolist()
    series = csel[csel["Country"].isin(top_countries)].copy()
    return series, avgs.reset_index(name="Average")


def plot_top(series: pd.DataFrame, areas: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    rank_col = "Average" if "Average" in areas.columns else ("Area" if "Area" in areas.columns else areas.columns[-1])
    order = areas.sort_values(rank_col, ascending=False)["Country"].head(5).tolist()
    plt.figure(figsize=(12, 7))
    for c in order:
        s = series[series["Country"] == c].sort_values("Year")
        # Ensure plotting respects the 3-metric minimum rule
        if "AvailableCount" in s.columns:
            s = s[s["AvailableCount"] >= 3]
        if s.empty:
            continue
        plt.plot(s["Year"], s["CompositeStanding"], label=c)
    plt.title("Empire Composite Standing — Top 5 by Average (dynamic start)")
    plt.xlabel("Year")
    plt.ylabel("Normalized Composite Standing (0–1)")
    # Dynamic x-axis: start at earliest year present in filtered series
    years = pd.to_numeric(series["Year"], errors="coerce").dropna()
    if not years.empty:
        xmin, xmax = int(years.min()), int(years.max())
        plt.xlim(xmin, xmax)
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _metric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in {"Country", "Year", "CompositeStanding", "AvailableCount"} and not c.endswith("_norm")]


def _norm_columns_for(df: pd.DataFrame, metric: str) -> str:
    nc = f"{metric}_norm"
    return nc if nc in df.columns else ""


def compute_winners_by_metric_year(composite: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = _metric_columns(composite)
    for yr, grp in composite.groupby("Year"):
        for m in metrics:
            s = grp[m]
            valid = grp[s.ge(0).fillna(False)]
            if valid.empty:
                continue
            idx = valid[m].astype(float).idxmax()
            row = composite.loc[idx]
            rows.append({
                "Year": int(yr) if pd.notna(yr) else yr,
                "Metric": m,
                "Country": row["Country"],
                "Value": row[m],
            })
    return pd.DataFrame(rows).sort_values(["Year", "Metric"]).reset_index(drop=True)


def compute_top5_by_year(composite: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = _metric_columns(composite)
    norm_map = {m: _norm_columns_for(composite, m) for m in metrics}
    for yr, grp in composite.groupby("Year"):
        top = grp.sort_values("CompositeStanding", ascending=False).head(5).copy()
        # Determine which metrics each top country wins (normalized = per-year max of that norm column)
        wins_per_metric = {}
        for m in metrics:
            nc = norm_map[m]
            if not nc:
                continue
            maxv = grp[nc].max(skipna=True)
            wins_per_metric[m] = maxv
        for rank, (_, r) in enumerate(top.iterrows(), start=1):
            won = []
            for m in metrics:
                nc = norm_map[m]
                if not nc or pd.isna(r[nc]) or pd.isna(wins_per_metric[m]):
                    continue
                # Allow floating tolerance
                if abs(float(r[nc]) - float(wins_per_metric[m])) < 1e-12:
                    won.append(m)
            rows.append({
                "Year": int(yr) if pd.notna(yr) else yr,
                "Rank": rank,
                "Country": r["Country"],
                "CompositeStanding": r["CompositeStanding"],
                "AvailableCount": int(r["AvailableCount"]) if pd.notna(r["AvailableCount"]) else 0,
                "MetricsWon": ";".join(sorted(won)) if won else "",
            })
    return pd.DataFrame(rows).sort_values(["Year", "Rank"]).reset_index(drop=True)


def compute_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    # Deprecated in favor of compute_coverage_from_wide; kept for reference.
    if panel.empty:
        return pd.DataFrame(columns=["Metric", "Year", "Count"])
    tmp = panel.dropna(subset=["Year"]).copy()
    tmp["Year"] = pd.to_numeric(tmp["Year"], errors="coerce")
    tmp = tmp.dropna(subset=["Year"])  # keep numeric years
    cov = (
        tmp.dropna(subset=["Value"]) 
           .groupby(["Metric", "Year"]) ["Country"].nunique()
           .reset_index(name="Count")
    )
    return cov.sort_values(["Metric", "Year"]).reset_index(drop=True)


def compute_coverage_from_wide(wide: pd.DataFrame) -> pd.DataFrame:
    if wide.empty:
        return pd.DataFrame(columns=["Metric", "Year", "Count"])
    df = wide.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"]).reset_index(drop=True)
    metric_cols = [c for c in df.columns if c not in {"Country", "Year"}]
    rows = []
    for m in metric_cols:
        # Presence if value != -1 and not NaN
        pres = df[["Year", m]].copy()
        pres["present"] = pres[m].apply(lambda v: (pd.notna(v)) and (float(v) != -1.0))
        cnt = pres.groupby("Year")["present"].sum().reset_index()
        for _, r in cnt.iterrows():
            rows.append({"Metric": m, "Year": int(r["Year"]), "Count": int(r["present"])})
    cov = pd.DataFrame(rows)
    return cov.sort_values(["Metric", "Year"]).reset_index(drop=True)


def _sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def plot_coverage_per_metric(coverage: pd.DataFrame, out_dir: Path) -> None:
    if coverage.empty:
        return
    import matplotlib.pyplot as plt
    for metric, grp in coverage.groupby("Metric"):
        g = grp.sort_values("Year")
        plt.figure(figsize=(10, 4))
        plt.plot(g["Year"], g["Count"], marker="o", linewidth=1.5)
        plt.title(f"Data Points per Year — {metric}")
        plt.xlabel("Year")
        plt.ylabel("Count of Data Points")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = out_dir / f"coverage_{_sanitize_filename(str(metric))}_counts.png"
        plt.savefig(fname, dpi=150)
        plt.close()


def plot_coverage_combined(coverage: pd.DataFrame, out_path: Path, logy: bool = True) -> None:
    if coverage.empty:
        return
    import matplotlib.pyplot as plt
    cov = coverage.copy()
    cov["Year"] = pd.to_numeric(cov["Year"], errors="coerce")
    cov = cov.dropna(subset=["Year"])  # numeric years only
    pivot = cov.pivot_table(index="Year", columns="Metric", values="Count", aggfunc="sum").sort_index()
    # Do not draw lines down to zero; mask zeros so series start/end cleanly
    pivot = pivot.where(pivot > 0)
    # Dynamic start: first year where at least 3 metrics have data (>0 count)
    presence = (pivot.fillna(0) > 0).sum(axis=1)
    valid_years = presence[presence >= 3].index
    xmin = int(valid_years.min()) if len(valid_years) else int(pivot.index.min())
    xmax = int(pivot.index.max())
    plt.figure(figsize=(12, 6))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], label=str(col))
    plt.title("Data Points per Year by Metric")
    plt.xlabel("Year")
    plt.ylabel("Count of Data Points")
    if logy:
        plt.yscale("log")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Add small horizontal padding so first/last points don’t touch the axes
    try:
        plt.xlim(xmin - 1, xmax + 1)
        plt.margins(x=0.02)
    except Exception:
        plt.xlim(xmin, xmax)
    plt.savefig(out_path, dpi=150)
    plt.close()


########################################
# Reporting helpers
########################################

def summarize_schema(panel: pd.DataFrame) -> pd.DataFrame:
    return (
        panel.groupby("Metric").agg(
            n_rows=("Value", "size"),
            min_year=("Year", "min"),
            max_year=("Year", "max"),
            n_countries=("Country", pd.Series.nunique),
        ).reset_index()
    )


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
