from typing import List

import numpy as np
import pandas as pd

from .utils import min_max_norm_by_year, share_by_year, robust_min_max_norm_by_year, group_rolling_mean, group_rolling_std


METRIC_COLUMNS = [
    "Education",
    "Military",
    "EconomicIndex",
    "TradeShare",
    "ReserveCurrency",
    "FinancialCenter",
    "Innovation",
    "Competitiveness",
]


def compute_metrics(edu: pd.DataFrame, mil: pd.DataFrame, gmd: pd.DataFrame, chat: pd.DataFrame, polity: pd.DataFrame) -> pd.DataFrame:
    """Compute six metrics per country-year and return a combined DataFrame.

    Returns columns: country, year, <metrics...>
    """

    # 1) Education = norm(education)
    edu_df = edu[["country", "year", "education"]].dropna(subset=["year"]).copy()
    edu_df = min_max_norm_by_year(edu_df, "year", "education", "Education")
    edu_df = edu_df[["country", "year", "Education"]]

    # 2) Military = norm(share of CINC) (fallback: norm(share of milex))
    if "cinc" in mil.columns:
        mil_df = mil[["country", "year", "cinc"]].copy()
        mil_df = share_by_year(mil_df, "year", "cinc", "cinc_share")
        mil_df = min_max_norm_by_year(mil_df, "year", "cinc_share", "Military")
    else:
        mil_df = mil[["country", "year", "milex"]].copy()
        mil_df = share_by_year(mil_df, "year", "milex", "milex_share")
        mil_df = min_max_norm_by_year(mil_df, "year", "milex_share", "Military")
    mil_df = mil_df[["country", "year", "Military"]]

    # 3) EconomicIndex = norm(share of rGDP_USD)
    gmd_econ = gmd[["country", "year", "rGDP_USD"]].copy()
    gmd_econ = share_by_year(gmd_econ, "year", "rGDP_USD", "rgdp_share")
    gmd_econ = min_max_norm_by_year(gmd_econ, "year", "rgdp_share", "EconomicIndex")
    gmd_econ = gmd_econ[["country", "year", "EconomicIndex"]]

    # 4) TradeShare = avg(norm(share of exports), norm(share of imports))
    gmd_trade = gmd[["country", "year", "exports_USD", "imports_USD"]].copy()
    gmd_trade = share_by_year(gmd_trade, "year", "exports_USD", "exports_share")
    gmd_trade = share_by_year(gmd_trade, "year", "imports_USD", "imports_share")
    # Revert to simple min-max normalization within year
    gmd_trade = min_max_norm_by_year(gmd_trade, "year", "exports_share", "exports_norm")
    gmd_trade = min_max_norm_by_year(gmd_trade, "year", "imports_share", "imports_norm")
    gmd_trade["TradeShare"] = gmd_trade[["exports_norm", "imports_norm"]].mean(axis=1, skipna=True)
    gmd_trade = gmd_trade[["country", "year", "TradeShare"]]

    # 5) ReserveCurrency = 1 - norm(USDfx)
    # ReserveCurrency = 1 - norm(USDfx)
    gmd_res = gmd[["country", "year", "USDfx"]].copy()
    gmd_res = min_max_norm_by_year(gmd_res, "year", "USDfx", "usd_norm")
    gmd_res["ReserveCurrency"] = 1 - gmd_res["usd_norm"]
    gmd_res = gmd_res[["country", "year", "ReserveCurrency"]]

    # 6) FinancialCenter = 1 - norm(share of cgovdebt)
    # FinancialCenter = 1 - norm(cgovdebt_GDP)
    gmd_fin = gmd[["country", "year", "cgovdebt_GDP"]].copy()
    gmd_fin = min_max_norm_by_year(gmd_fin, "year", "cgovdebt_GDP", "debt_norm")
    gmd_fin["FinancialCenter"] = 1 - gmd_fin["debt_norm"]
    gmd_fin = gmd_fin[["country", "year", "FinancialCenter"]]

    # 7) Innovation = average of all available normalized CHAT columns
    chat_df = chat.copy()
    feature_cols = [c for c in chat_df.columns if c not in ("country", "year")]
    norm_cols = []
    for c in feature_cols:
        out_c = f"{c}__norm"
        chat_df = min_max_norm_by_year(chat_df, "year", c, out_c)
        norm_cols.append(out_c)
    chat_df["Innovation"] = chat_df[norm_cols].mean(axis=1, skipna=True)
    chat_df = chat_df[["country", "year", "Innovation"]]

    # 8) Competitiveness = average of all normalized non-null metrics per country-year (no forward fill)
    # We'll compute it after merging core metrics and Innovation.

    # Merge all metrics outer on (country, year)
    out = edu_df.merge(mil_df, on=["country", "year"], how="outer") \
                .merge(gmd_econ, on=["country", "year"], how="outer") \
                .merge(gmd_trade, on=["country", "year"], how="outer") \
                .merge(gmd_res, on=["country", "year"], how="outer") \
                .merge(gmd_fin, on=["country", "year"], how="outer") \
                .merge(chat_df, on=["country", "year"], how="outer")

    # Compute Competitiveness as mean of normalized metrics available (exclude Competitiveness itself)
    base_norm_cols = [
        c for c in [
            "Education", "Military", "EconomicIndex", "TradeShare",
            "ReserveCurrency", "FinancialCenter", "Innovation"
        ] if c in out.columns
    ]
    out["Competitiveness"] = out[base_norm_cols].mean(axis=1, skipna=True)
    
    # Also compute a weighted composite that reweights remaining factors when some are missing
    # Aliases per requested schema
    if "Innovation" in out.columns:
        out["Technology"] = out["Innovation"]
    if "EconomicIndex" in out.columns:
        out["EconomicOutput"] = out["EconomicIndex"]

    weights = {
        "Education": 0.15,
        "Competitiveness": 0.15,
        "Technology": 0.15,
        "EconomicOutput": 0.15,
        "TradeShare": 0.10,
        "Military": 0.10,
        "FinancialCenter": 0.10,
        "ReserveCurrency": 0.10,
    }
    available_cols = [k for k in weights if k in out.columns]
    if available_cols:
        value_mat = out[available_cols]
        weight_vec = pd.Series({k: weights[k] for k in available_cols})
        weighted_sum = (value_mat * weight_vec).sum(axis=1, skipna=True)
        present_weights = value_mat.notna().astype(float) * weight_vec
        present_weights = present_weights.sum(axis=1)
        out["WorldOrderIndex"] = weighted_sum.divide(present_weights).where(present_weights > 0)
    
    # Keep numeric years only
    out = out.dropna(subset=["year"]).sort_values(["country", "year"]).reset_index(drop=True)
    return out
