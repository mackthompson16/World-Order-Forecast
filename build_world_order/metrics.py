from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
import json

from .utils import yearwise_min_max_norm, forward_fill_limited, interpolate_panel
from .composite import compute_composite


METRIC_COLS = ["EDU", "MIL", "ECON", "TRAD", "RESV", "FIN", "INV", "CMPT"]


def _load_innovation_config() -> Dict[str, Any]:
    """Load configs/innovation.json if present; else return sensible defaults."""
    defaults = {
        "include_columns": [
            "internetuser", "computer", "cellphone", "creditdebit", "eft",
            "telephone", "tv", "med_mriunit", "med_catscanner",
            "vehicle_car", "vehicle_com"
        ],
        "treat_as_rate": ["internetuser"],
        "min_coverage": 15,
        "use_per_capita": True,
        "log1p_on_counts": True,
        "weights": {},
        "method": "sum_then_norm"  # or "avg_norm"
    }
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "innovation.json"
    try:
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            # update defaults shallowly
            for k, v in user_cfg.items():
                defaults[k] = v
    except Exception:
        pass
    return defaults


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

    # Innovation (Technology)
    cfg = _load_innovation_config()
    chat_cols_all = [c for c in chat.columns if c not in ("ISO3", "country_name", "year")]
    choose = [c for c in cfg.get("include_columns", []) if c in chat_cols_all]
    chat_cols = choose if choose else chat_cols_all
    if chat_cols:
        chat_num = chat[["ISO3", "year"] + chat_cols].copy()
        for col in chat_cols:
            chat_num[col] = pd.to_numeric(chat_num[col], errors="coerce")
        chat_num = chat_num.groupby(["ISO3", "year"], as_index=False).mean(numeric_only=True)
        # Drop rows where no tech data exists
        has_any = chat_num[chat_cols].notna().any(axis=1)
        chat_num = chat_num.loc[has_any].copy()
        method = str(cfg.get("method", "sum_then_norm")).lower()
        if method == "sum_then_norm":
            # Sum available tech values per country-year, then min-max normalize the sums by year.
            sums = chat_num[["ISO3", "year"]].copy()
            sums["tech_sum"] = chat_num[chat_cols].sum(axis=1, skipna=True)
            # If all techs were NaN for a row, drop it
            sums = sums[~sums["tech_sum"].isna()].copy()
            inv = []
            for y, g in sums.groupby("year"):
                vals = g["tech_sum"].dropna()
                if len(vals) < 4:
                    # too few countries to normalize reliably
                    part = pd.DataFrame({"ISO3": g["ISO3"], "year": g["year"], "INV": np.nan})
                else:
                    vmin, vmax = vals.min(), vals.max()
                    if vmin == vmax:
                        part = pd.DataFrame({"ISO3": g["ISO3"], "year": g["year"], "INV": np.nan})
                    else:
                        part = pd.DataFrame({
                            "ISO3": g["ISO3"],
                            "year": g["year"],
                            "INV": (g["tech_sum"] - vmin) / (vmax - vmin)
                        })
                inv.append(part)
            inv_df = pd.concat(inv, ignore_index=True)
            parts.append(inv_df[["ISO3", "year", "INV"]])
        else:
            # Fallback to per-tech min-max then mean
            normed = chat_num[["ISO3", "year"]].copy()
            min_cov = 10
            for col in chat_cols:
                out = pd.Series(np.nan, index=chat_num.index, dtype=float)
                for y, g in chat_num[["year", col]].groupby("year"):
                    vals = g[col].dropna()
                    if len(vals) < max(2, min_cov):
                        continue
                    vmin, vmax = vals.min(), vals.max()
                    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
                        continue
                    out.loc[g.index] = (g[col] - vmin) / (vmax - vmin)
                normed[col] = out.values
            normed["INV"] = normed[chat_cols].mean(axis=1)
            parts.append(normed[["ISO3", "year", "INV"]])

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


def write_metrics(clean_path: Path, chat_path: Path, out_path: Path) -> Path:
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
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Order columns
    cols = ["country_name", "ISO3", "year"] + METRIC_COLS + ["INDEX"]
    cols = [c for c in cols if c in metrics.columns]
    metrics[cols].to_csv(out_path, index=False)
    return out_path
