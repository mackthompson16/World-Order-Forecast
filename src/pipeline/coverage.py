from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def compute_coverage_from_wide(wide: pd.DataFrame) -> pd.DataFrame:
    if wide.empty:
        return pd.DataFrame(columns=["Metric", "Year", "Count"])
    df = wide.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"]).reset_index(drop=True)
    metric_cols = [c for c in df.columns if c not in {"Country", "Year"}]

    def canonical(name: str) -> str:
        # Collapse Competitiveness variants into one
        return "Competitiveness" if name.startswith("Competitiveness") else name

    rows = []
    for yr, grp in df.groupby("Year"):
        by_canon = {}
        for col in metric_cols:
            by_canon.setdefault(canonical(col), []).append(col)
        for canon_metric, cols in by_canon.items():
            present_any = (grp[cols] != -1).any(axis=1)
            rows.append({"Metric": canon_metric, "Year": int(yr), "Count": int(present_any.sum())})
    cov = pd.DataFrame(rows)
    return cov.sort_values(["Metric", "Year"]).reset_index(drop=True)


def plot_coverage_combined(coverage: pd.DataFrame, out_path: Path, logy: bool = True) -> None:
    if coverage.empty:
        return
    cov = coverage.copy()
    # Collapse Competitiveness variants into a single metric for display
    if "Metric" in cov.columns:
        cov["Metric"] = cov["Metric"].replace(
            {"Competitiveness_GCI": "Competitiveness", "Competitiveness_component": "Competitiveness"}
        )
    cov["Year"] = pd.to_numeric(cov["Year"], errors="coerce")
    cov = cov.dropna(subset=["Year"])  # numeric years only
    pivot = cov.pivot_table(index="Year", columns="Metric", values="Count", aggfunc="sum").sort_index()
    # Do not draw lines down to zero; mask zeros so series start/end cleanly
    pivot = pivot.where(pivot > 0)
    # Dynamic start: first year where at least 3 metrics have data
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
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.xlim(xmin - 1, xmax + 1)
        plt.margins(x=0.02)
    except Exception:
        plt.xlim(xmin, xmax)
    plt.savefig(out_path, dpi=150)
    plt.close()


__all__ = ["compute_coverage_from_wide", "plot_coverage_combined"]
