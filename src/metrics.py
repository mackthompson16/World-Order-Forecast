from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
import json

from .utils import yearwise_min_max_norm, forward_fill_limited, interpolate_panel, moving_average
from .composite import compute_composite


METRIC_COLS = ["EDU", "MIL", "ECON", "TRAD", "RESV", "FIN", "INV", "CMPT"]


# Alias matching requested helper name
per_year_minmax = yearwise_min_max_norm


def _compute_innovation(chat: pd.DataFrame) -> pd.DataFrame:
    df = chat.copy()
    feature_cols: List[str] = [c for c in df.columns if c not in ("country", "year")]
    # Coerce features to numeric to avoid string arithmetic
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Normalize each feature per year, then sum and normalize the sum per year
    for c in feature_cols:
        df[c] = per_year_minmax(df[c], df["year"])  # type: ignore[arg-type]
    df["_sum_norm"] = df[feature_cols].sum(axis=1, skipna=True)
    df["innovation"] = per_year_minmax(df["_sum_norm"], df["year"])  # type: ignore[arg-type]
    return df[["country", "year", "innovation"]]



def compute_metrics(clean_path: Path, chat_path: Path) -> pd.DataFrame:
    clean = pd.read_csv(clean_path)
    chat = pd.read_csv(chat_path)

    # Prepare base index
    base = clean[["ISO3", "country_name", "year"]].dropna(subset=["ISO3", "year"]).drop_duplicates().copy()

    def _unique_mean(df: pd.DataFrame, key_cols: List[str], val_col: str) -> pd.DataFrame:
        out = (
            df[key_cols + [val_col]]
            .groupby(key_cols, as_index=False)[val_col]
            .mean()
        )
        return out

    # Normalized simple metrics
    def add_norm_metric(src: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
        df = src[["ISO3", "year", value_col]].copy()
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df = _unique_mean(df, ["ISO3", "year"], value_col)
        df[out_col] = yearwise_min_max_norm(df[value_col], df["year"])
        return df[["ISO3", "year", out_col]]

    parts = []
    if "education" in clean.columns:
        parts.append(add_norm_metric(clean, "education", "EDU"))
    if "CINC" in clean.columns:
        parts.append(add_norm_metric(clean, "CINC", "MIL"))
    if "rGDP_USD" in clean.columns:
        parts.append(add_norm_metric(clean, "rGDP_USD", "ECON"))

    # Trade: average of normalized exports and imports
    trade_components = []
    if "exports_USD" in clean.columns:
        trade_components.append(add_norm_metric(clean, "exports_USD", "_NEXP"))
    if "imports_USD" in clean.columns:
        trade_components.append(add_norm_metric(clean, "imports_USD", "_NIMP"))
    if trade_components:
        tdf = base[["ISO3", "year"]].drop_duplicates().copy()
        for t in trade_components:
            tdf = tdf.merge(t, on=["ISO3", "year"], how="left")
        tcols = [c for c in tdf.columns if c.startswith("_N")]
        tdf["TRAD"] = tdf[tcols].mean(axis=1)
        parts.append(tdf[["ISO3", "year", "TRAD"]])

    # Reserve currency: CA_USD
    if "CA_USD" in clean.columns:
        parts.append(add_norm_metric(clean, "CA_USD", "RESV"))

    # Financial center: 0.5*norm(M0)+0.3*norm(finv_GDP)+0.2*(1-norm(cgovdebt_GDP))
    fin_parts = []
    if "M0" in clean.columns:
        fin_parts.append(add_norm_metric(clean, "M0", "_NM0"))
    if "finv_GDP" in clean.columns:
        fin_parts.append(add_norm_metric(clean, "finv_GDP", "_NFINV"))
    if "cgovdebt_GDP" in clean.columns:
        fin_parts.append(add_norm_metric(clean, "cgovdebt_GDP", "_NDEBT"))
    if fin_parts:
        fdf = base[["ISO3", "year"]].drop_duplicates().copy()
        for f in fin_parts:
            fdf = fdf.merge(f, on=["ISO3", "year"], how="left")
        fdf["FIN"] = 0.5 * fdf.get("_NM0") + 0.3 * fdf.get("_NFINV") + 0.2 * (1 - fdf.get("_NDEBT"))
        parts.append(fdf[["ISO3", "year", "FIN"]])

    # Innovation (Technology) - simplified per request
    if not chat.empty:
        chat_use = chat.copy()
        chat_use = chat_use.rename(columns={"ISO3": "country"})
        inv_df = _compute_innovation(chat_use)
        inv_df = inv_df.rename(columns={"country": "ISO3", "innovation": "INV"})
        parts.append(inv_df[["ISO3", "year", "INV"]])

    # Competitiveness: average of normalized xconst and parcomp
    comp_parts = []
    if "xconst" in clean.columns:
        comp_parts.append(add_norm_metric(clean, "xconst", "_NXCONST"))
    if "parcomp" in clean.columns:
        comp_parts.append(add_norm_metric(clean, "parcomp", "_NPARCOMP"))
    if comp_parts:
        cdf = base[["ISO3", "year"]].drop_duplicates().copy()
        for c in comp_parts:
            cdf = cdf.merge(c, on=["ISO3", "year"], how="left")
        cdf["CMPT"] = cdf[[col for col in cdf.columns if col.startswith("_N")]].mean(axis=1)
        parts.append(cdf[["ISO3", "year", "CMPT"]])

    # Merge all parts
    metrics = base.drop_duplicates(["ISO3", "year"]).copy()
    for p in parts:
        # ensure part uniqueness
        p = p.drop_duplicates(["ISO3", "year"]).copy()
        metrics = metrics.merge(p, on=["ISO3", "year"], how="left")

    # Drop rows where no metric is available (sparsity)
    if not metrics.empty:
        has_any = metrics[METRIC_COLS].notna().any(axis=1)
        metrics = metrics.loc[has_any].copy()

    # Forward-fill tail to 2024, at most 10y, without adding empty early years
    if not metrics.empty:
        # Create tail rows per country up to min(last+10, 2024)
        tails = []
        for iso3, g in metrics.groupby("ISO3"):
            g = g.sort_values("year")
            # last year with any metric value
            g_has = g[METRIC_COLS].notna().any(axis=1)
            if not g_has.any():
                continue
            last_year = int(g.loc[g_has, "year"].max())
            limit = min(2024, last_year + 10)
            add_years = [y for y in range(last_year + 1, limit + 1)]
            if add_years:
                cname = g["country_name"].dropna().iloc[0] if g["country_name"].notna().any() else None
                tails.append(pd.DataFrame({"ISO3": iso3, "country_name": cname, "year": add_years}))
        if tails:
            tail_df = pd.concat(tails, ignore_index=True)
            metrics = pd.concat([metrics, tail_df], ignore_index=True)
        metrics = metrics.drop_duplicates(["ISO3", "year"]).sort_values(["ISO3", "year"]).reset_index(drop=True)
        metrics = forward_fill_limited(metrics, ["ISO3"], METRIC_COLS, max_years=10, year_col="year")

    # Linear interpolate inside gaps for all metric columns (and INDEX later)
    if not metrics.empty:
        interp_cols = [c for c in METRIC_COLS if c in metrics.columns]
        metrics = interpolate_panel(metrics, ["ISO3"], interp_cols)

    return metrics


def _smooth_components(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    if df.empty:
        return df
    present = [c for c in METRIC_COLS if c in df.columns]
    groups = []
    for iso3, g in df.groupby("ISO3"):
        g = g.sort_values("year").copy()
        for col in present:
            g[col] = moving_average(g[col], window)
        if "INDEX" in g.columns:
            g["INDEX"] = moving_average(g["INDEX"], window)
        groups.append(g)
    return pd.concat(groups, ignore_index=True).sort_values(["ISO3", "year"]) if groups else df


def write_metrics(clean_path: Path, chat_path: Path, out_path: Path, smooth_window: int = 5) -> Path:
    metrics = compute_metrics(clean_path, chat_path)
    # Add composite index into the metrics as INDEX
    comp = compute_composite(metrics)
    if "WorldOrderIndex" in comp.columns:
        metrics = metrics.merge(
            comp[["ISO3", "year", "WorldOrderIndex"]],
            on=["ISO3", "year"],
            how="left",
        )
        metrics = metrics.rename(columns={"WorldOrderIndex": "INDEX"})
        # Interpolate INDEX inside gaps too (after adding it)
        metrics = interpolate_panel(metrics, ["ISO3"], ["INDEX"]) if "INDEX" in metrics.columns else metrics

    # Smooth calculated metrics and INDEX so metrics.csv is always smoothed
    metrics = _smooth_components(metrics, window=smooth_window)

    # Merge geography index if available (do not change row coverage)
    try:
        geo_path = Path(clean_path).parent / "clean_geography.csv"
        if geo_path.exists():
            geo = pd.read_csv(geo_path)
            # Expect columns: Country, abv, year, ..., index
            if set(["abv", "year"]).issubset(geo.columns) and "index" in geo.columns:
                geo_use = geo[["abv", "year", "index"]].rename(columns={"abv": "ISO3", "index": "geography_index"})
                metrics = metrics.merge(geo_use, on=["ISO3", "year"], how="left")
    except Exception:
        # Non-fatal if geography data missing or malformed
        pass
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Order columns
    cols = ["country_name", "ISO3", "year"] + METRIC_COLS + ["INDEX", "geography_index"]
    cols = [c for c in cols if c in metrics.columns]
    metrics[cols].to_csv(out_path, index=False)
    return out_path
