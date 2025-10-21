from typing import List

import numpy as np
import pandas as pd

from .utils import min_max_norm_by_year, share_by_year


METRIC_COLUMNS = [
    "Education",
    "Military",
    "EconomicIndex",
    "TradeShare",
    "ReserveCurrency",
    "FinancialCenter",
]


def compute_metrics(edu: pd.DataFrame, mil: pd.DataFrame, gmd: pd.DataFrame) -> pd.DataFrame:
    """Compute six metrics per country-year and return a combined DataFrame.

    Returns columns: country, year, <metrics...>
    """

    # 1) Education = norm(education)
    edu_df = edu[["country", "year", "education"]].dropna(subset=["year"]).copy()
    edu_df = min_max_norm_by_year(edu_df, "year", "education", "Education")
    edu_df = edu_df[["country", "year", "Education"]]

    # 2) Military = (norm(milex/sum(milex)) + norm(milper/sum(milper))) / 2
    mil_df = mil[["country", "year", "milex", "milper"]].copy()
    mil_df = share_by_year(mil_df, "year", "milex", "milex_share")
    mil_df = share_by_year(mil_df, "year", "milper", "milper_share")
    mil_df = min_max_norm_by_year(mil_df, "year", "milex_share", "milex_norm")
    mil_df = min_max_norm_by_year(mil_df, "year", "milper_share", "milper_norm")
    mil_df["Military"] = (
        mil_df[["milex_norm", "milper_norm"]].mean(axis=1, skipna=True)
    )
    mil_df = mil_df[["country", "year", "Military"]]

    # 3) EconomicIndex = norm(rGDP_USD / sum(rGDP_USD))
    gmd_econ = gmd[["country", "year", "rGDP_USD"]].copy()
    gmd_econ = share_by_year(gmd_econ, "year", "rGDP_USD", "rgdp_share")
    gmd_econ = min_max_norm_by_year(gmd_econ, "year", "rgdp_share", "EconomicIndex")
    gmd_econ = gmd_econ[["country", "year", "EconomicIndex"]]

    # 4) TradeShare = norm((exports_USD + imports_USD) / sum(...))
    gmd_trade = gmd[["country", "year", "exports_USD", "imports_USD"]].copy()
    gmd_trade["trade_val"] = gmd_trade[["exports_USD", "imports_USD"]].sum(axis=1, skipna=True)
    gmd_trade = share_by_year(gmd_trade, "year", "trade_val", "trade_share")
    gmd_trade = min_max_norm_by_year(gmd_trade, "year", "trade_share", "TradeShare")
    gmd_trade = gmd_trade[["country", "year", "TradeShare"]]

    # 5) ReserveCurrency = 1 - (norm(USDfx) + norm(infl) + norm(CA_GDP)) / 3
    gmd_res = gmd[["country", "year", "USDfx", "infl", "CA_GDP"]].copy()
    gmd_res = min_max_norm_by_year(gmd_res, "year", "USDfx", "USDfx_norm")
    gmd_res = min_max_norm_by_year(gmd_res, "year", "infl", "infl_norm")
    gmd_res = min_max_norm_by_year(gmd_res, "year", "CA_GDP", "ca_norm")
    gmd_res["ReserveCurrency"] = 1 - gmd_res[["USDfx_norm", "infl_norm", "ca_norm"]].mean(axis=1, skipna=True)
    gmd_res = gmd_res[["country", "year", "ReserveCurrency"]]

    # 6) FinancialCenter = avg(
    #     norm(M2 / sum(M2)),
    #     1 - avg(norm(infl), norm(abs(govdef_GDP))),
    #     norm(max(CA_GDP, 0))
    # )
    gmd_fin = gmd[["country", "year", "M2", "infl", "govdef_GDP", "CA_GDP"]].copy()
    gmd_fin = share_by_year(gmd_fin, "year", "M2", "M2_share")
    gmd_fin = min_max_norm_by_year(gmd_fin, "year", "M2_share", "m2_norm")
    # stability term
    gmd_fin = min_max_norm_by_year(gmd_fin, "year", "infl", "infl_norm")
    gmd_fin["abs_def"] = gmd_fin["govdef_GDP"].abs()
    gmd_fin = min_max_norm_by_year(gmd_fin, "year", "abs_def", "def_norm")
    gmd_fin["stability"] = 1 - gmd_fin[["infl_norm", "def_norm"]].mean(axis=1, skipna=True)
    # positive CA
    gmd_fin["capos"] = gmd_fin["CA_GDP"].clip(lower=0)
    gmd_fin = min_max_norm_by_year(gmd_fin, "year", "capos", "capos_norm")
    gmd_fin["FinancialCenter"] = (
        gmd_fin[["m2_norm", "stability", "capos_norm"]].mean(axis=1, skipna=True)
    )
    gmd_fin = gmd_fin[["country", "year", "FinancialCenter"]]

    # Merge all metrics outer on (country, year)
    out = edu_df.merge(mil_df, on=["country", "year"], how="outer") \
                .merge(gmd_econ, on=["country", "year"], how="outer") \
                .merge(gmd_trade, on=["country", "year"], how="outer") \
                .merge(gmd_res, on=["country", "year"], how="outer") \
                .merge(gmd_fin, on=["country", "year"], how="outer")

    # Keep numeric years only
    out = out.dropna(subset=["year"]).sort_values(["country", "year"]).reset_index(drop=True)
    return out

