import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

SUPPORTED_EXT = {".csv", ".xls", ".xlsx", ".dta"}


MEASUREMENT_LABEL: Dict[str, str] = {
    "GlobalDebt": "GeneralGovernmentDebt_pctGDP",
    "GDP": "GDP_Total",
    "MilitaryStrength": "MilitaryExpenditure_USD_const2023",
    "Innovation": "PatentApplications_perMillion",
    "Education": "AverageYearsSchooling",
    "Competitiveness": "GCI",
    "ReserveCurrency": "ReserveCurrencyShares",
}


def detect_metric_from_filename(path: Path) -> Optional[str]:
    base = path.stem.lower()
    if "debt" in base:
        return "GlobalDebt"
    if base == "gdp" or "gdp" in base:
        return "GDP"
    if "military" in base or "millitary" in base:
        return "MilitaryStrength"
    if "innovation" in base:
        return "Innovation"
    if "education" in base:
        return "Education"
    if "competitiveness" in base or "wef" in base:
        return "Competitiveness"
    if "cofer" in base or "reserve" in base:
        return "ReserveCurrency"
    return None


def detect_year_range_csv_head(path: Path) -> Tuple[Optional[int], Optional[int]]:
    try:
        df = pd.read_csv(path, nrows=25, header=None)
    except Exception:
        return None, None
    if df.empty:
        return None, None
    header = df.iloc[0].astype(str)
    years = []
    for v in header:
        m = re.search(r"(\d{4})", str(v))
        if m:
            y = int(m.group(1))
            if 1000 <= y <= 3000:
                years.append(y)
    if years:
        return min(years), max(years)
    # Try long format with a Year column
    try:
        df2 = pd.read_csv(path, nrows=500)
        if "Year" in df2.columns:
            y = pd.to_numeric(df2["Year"], errors="coerce").dropna()
            if not y.empty:
                return int(y.min()), int(y.max())
    except Exception:
        pass
    return None, None


def detect_year_range_generic(path: Path) -> Tuple[Optional[int], Optional[int]]:
    if path.suffix.lower() == ".csv":
        return detect_year_range_csv_head(path)
    try:
        df = None
        if path.suffix.lower() in {".xls", ".xlsx"}:
            df = pd.read_excel(path, nrows=50)
        elif path.suffix.lower() == ".dta":
            df = pd.read_stata(path)
        if df is not None:
            if "Year" in df.columns:
                y = pd.to_numeric(df["Year"], errors="coerce").dropna()
                if not y.empty:
                    return int(y.min()), int(y.max())
            # scan headers for 4-digit years
            years = []
            for c in df.columns:
                m = re.fullmatch(r"\d{4}", str(c))
                if m:
                    years.append(int(c))
            if years:
                return min(years), max(years)
    except Exception:
        pass
    return None, None


def propose_move(path: Path) -> Optional[Tuple[Path, Path]]:
    metric = detect_metric_from_filename(path)
    if not metric:
        return None
    start, end = detect_year_range_generic(path)
    meas = MEASUREMENT_LABEL.get(metric, metric)
    name = f"{meas}"
    if start and end:
        name = f"{meas}_{start}_{end}"
    target_dir = path.parents[1] / "data" / metric if path.parent.name != metric else path.parent
    target = target_dir / f"{name}{path.suffix.lower()}"
    return target_dir, target


def main():
    ap = argparse.ArgumentParser(description="Organize data/ into metric subfolders with measurement_start_end filenames.")
    ap.add_argument("--data", default="data", help="Data root directory (default: data)")
    ap.add_argument("--apply", action="store_true", help="Actually move/rename files (default: dry-run)")
    args = ap.parse_args()

    root = Path(args.data).resolve()
    if not root.exists():
        print(f"Data directory not found: {root}")
        return

    moves: List[Tuple[Path, Path]] = []
    for p in sorted(pr for pr in root.rglob("*") if pr.is_file() and pr.suffix.lower() in SUPPORTED_EXT):
        prop = propose_move(p)
        if not prop:
            continue
        target_dir, target = prop
        if target == p:
            continue
        moves.append((p, target))

    if not moves:
        print("No moves proposed — files may already be organized.")
        return

    print("Proposed moves/renames:")
    for src, dst in moves:
        print(f" - {src} -> {dst}")

    if args.apply:
        for src, dst in moves:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dst)
        print("Done. Files moved.")
    else:
        print("Dry-run. Re-run with --apply to perform moves.")


if __name__ == "__main__":
    main()

