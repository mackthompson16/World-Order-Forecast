import re
import pandas as pd


def select_top5_by_area(composite: pd.DataFrame) -> pd.DataFrame:
    # Dynamic start: require GDP present and at least 3 metrics available
    csel = composite.copy()
    csel["Year"] = pd.to_numeric(csel["Year"], errors="coerce")
    csel = csel.dropna(subset=["Year"])  # numeric years only
    if "GDP_norm" in csel.columns:
        csel = csel[csel["GDP_norm"].notna()].copy()
    if "AvailableCount" in csel.columns:
        csel = csel[csel["AvailableCount"] >= 3].copy()

    coverage = csel.groupby("Country")["CompositeStanding"].size().rename("Coverage")
    area = csel.groupby("Country")["CompositeStanding"].sum().rename("Area")
    rank_df = pd.concat([coverage, area], axis=1).sort_values(["Coverage", "Area"], ascending=[False, False])
    top_countries = rank_df.head(5).index.tolist()
    series = csel[csel["Country"].isin(top_countries)].copy()
    return series, rank_df.reset_index()


def _metric_columns(df: pd.DataFrame):
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


__all__ = ["select_top5_by_area", "compute_winners_by_metric_year", "compute_top5_by_year"]
