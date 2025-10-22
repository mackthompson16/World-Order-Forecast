from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .normalize import per_year_minmax, fill_gaps


def _compute_military_share(mil: pd.DataFrame) -> pd.DataFrame:
    # Sum CINC per year, compute share
    mil = mil.copy()
    mil["cinc"] = pd.to_numeric(mil["cinc"], errors="coerce")
    totals = mil.groupby("year")["cinc"].transform("sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        mil["military_share"] = mil["cinc"] / totals
    return mil[["country", "year", "military_share"]]


def _compute_economic_share(gmd: pd.DataFrame) -> pd.DataFrame:
    g = gmd.copy()
    g["rGDP_USD"] = pd.to_numeric(g["rGDP_USD"], errors="coerce")
    totals = g.groupby("year")["rGDP_USD"].transform("sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        g["economic_share"] = g["rGDP_USD"] / totals
    return g[["country", "year", "economic_share"]]


def _compute_trade_share(gmd: pd.DataFrame) -> pd.DataFrame:
    g = gmd.copy()
    for c in ["exports_USD", "imports_USD"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    # Shares per year
    exp_total = g.groupby("year")["exports_USD"].transform("sum")
    imp_total = g.groupby("year")["imports_USD"].transform("sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        g["exp_share"] = g["exports_USD"] / exp_total
        g["imp_share"] = g["imports_USD"] / imp_total
    # Normalize each share per year and average
    g["exp_norm"] = per_year_minmax(g["exp_share"], g["year"])  # type: ignore[arg-type]
    g["imp_norm"] = per_year_minmax(g["imp_share"], g["year"])  # type: ignore[arg-type]
    g["trade_share"] = g[["exp_norm", "imp_norm"]].mean(axis=1, skipna=True)
    return g[["country", "year", "trade_share"]]


def _compute_reserve_and_finance(gmd: pd.DataFrame) -> pd.DataFrame:
    g = gmd.copy()
    for c in ["USDfx", "cgovdebt_GDP"]:
        if c not in g.columns:
            g[c] = np.nan
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g["USDfx_norm"] = per_year_minmax(g["USDfx"], g["year"])  # type: ignore[arg-type]
    g["debt_norm"] = per_year_minmax(g["cgovdebt_GDP"], g["year"])  # type: ignore[arg-type]
    g["reserve_currency"] = 1.0 - g["USDfx_norm"]
    g["financial_center"] = 1.0 - g["debt_norm"]
    return g[["country", "year", "reserve_currency", "financial_center"]]


def _compute_innovation(chat: pd.DataFrame) -> pd.DataFrame:
    df = chat.copy()
    feature_cols: List[str] = [c for c in df.columns if c not in ("country", "year")]
    # Normalize each feature per year and average across available features
    for c in feature_cols:
        df[c] = per_year_minmax(df[c], df["year"])  # type: ignore[arg-type]
    df["innovation"] = df[feature_cols].mean(axis=1, skipna=True)
    return df[["country", "year", "innovation"]]


def _compute_competitiveness(polity: pd.DataFrame) -> pd.DataFrame:
    p = polity.copy()
    p["competitiveness"] = per_year_minmax(p["polity"], p["year"])  # type: ignore[arg-type]
    return p[["country", "year", "competitiveness"]]


def build_metrics(dfs: Dict[str, pd.DataFrame], grid: pd.DataFrame) -> pd.DataFrame:
    """Compute all metrics and return a country-year panel.

    Applies per-country supplementation rules (ffill small gaps, interpolate longer)
    after composing raw metric series.
    """
    edu = dfs["education"][ ["country", "year", "education"] ].copy()
    edu["education"] = per_year_minmax(edu["education"], edu["year"])  # type: ignore[arg-type]

    mil_share = _compute_military_share(dfs["military"])  # share
    mil_share["military"] = per_year_minmax(mil_share["military_share"], mil_share["year"])  # type: ignore[arg-type]

    econ_share = _compute_economic_share(dfs["gmd"])  # share
    econ_share["economic_index"] = per_year_minmax(econ_share["economic_share"], econ_share["year"])  # type: ignore[arg-type]

    trade_share = _compute_trade_share(dfs["gmd"])  # already normalized average per-year

    res_fin = _compute_reserve_and_finance(dfs["gmd"])  # inverse norms

    innovation = _compute_innovation(dfs["chat"])  # per-year norm avg

    comp = _compute_competitiveness(dfs["polity"])  # per-year norm

    # Merge onto full grid and apply supplementation rules
    out = grid.merge(edu, on=["country", "year"], how="left") \
              .merge(mil_share[["country", "year", "military"]], on=["country", "year"], how="left") \
              .merge(econ_share[["country", "year", "economic_index"]], on=["country", "year"], how="left") \
              .merge(trade_share, on=["country", "year"], how="left") \
              .merge(res_fin, on=["country", "year"], how="left") \
              .merge(innovation, on=["country", "year"], how="left") \
              .merge(comp, on=["country", "year"], how="left")

    metric_cols = [
        "education",
        "military",
        "economic_index",
        "trade_share",
        "reserve_currency",
        "financial_center",
        "innovation",
        "competitiveness",
    ]

    # Supplement missing data per country
    def _fill_group(g: pd.DataFrame) -> pd.DataFrame:
        for c in metric_cols:
            g[c] = fill_gaps(g[c])
        return g

    out = out.sort_values(["country", "year"]).groupby("country", group_keys=False).apply(_fill_group)
    return out


def compute_composite(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    # Map aliases from README: Technology == Innovation; EconomicOutput == EconomicIndex
    df["technology"] = df["innovation"]
    df["economic_output"] = df["economic_index"]

    weights = {
        "education": 0.15,
        "competitiveness": 0.15,
        "technology": 0.15,
        "economic_output": 0.15,
        "trade_share": 0.10,
        "military": 0.10,
        "financial_center": 0.10,
        "reserve_currency": 0.10,
    }

    cols = list(weights.keys())
    # Row-wise renormalization of weights over available components
    present = df[cols].notna()
    present_weight = present.mul(pd.Series(weights))
    weight_sum = present_weight.sum(axis=1)
    # Avoid division by zero: if nothing present, keep NaN
    norm_w = present_weight.div(weight_sum, axis=0).fillna(0.0)
    df["WorldOrderIndex"] = (df[cols] * norm_w).sum(axis=1)
    return df

