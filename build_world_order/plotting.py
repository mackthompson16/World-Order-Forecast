from pathlib import Path
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt

from .utils import moving_average


HIGHLIGHT = {
    "USA": {"color": "#1f77b4", "lw": 2.8, "alpha": 1.0, "zorder": 3},
    "CHN": {"color": "#d62728", "lw": 2.8, "alpha": 1.0, "zorder": 3},
    "GBR": {"color": "#2ca02c", "lw": 2.8, "alpha": 1.0, "zorder": 3},
}


def _shade_wars(ax):
    # WWI: 1914-1918, WWII: 1939-1945
    ax.axvspan(1914, 1918, color="grey", alpha=0.15, lw=0)
    ax.axvspan(1939, 1945, color="grey", alpha=0.15, lw=0)


def plot_composite(
    comp: pd.DataFrame,
    out_dir: Path,
    smooth: int = 5,
    selected_countries: List[str] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if selected_countries is None:
        selected_countries = ["CHINA", "USA", "FRANCE", "GERMANY", "UNITED KINGDOM", "RUSSIA"]

    # Map names -> ISO3 using what's available in comp
    name_to_iso3 = (
        comp[["country_name", "ISO3"]].dropna().drop_duplicates().assign(country_name=lambda d: d["country_name"].str.upper())
    )
    wanted_iso3 = name_to_iso3[name_to_iso3["country_name"].isin([n.upper() for n in selected_countries])]["ISO3"].tolist()

    # Accept either WorldOrderIndex or INDEX; normalize to WorldOrderIndex
    if "WorldOrderIndex" not in comp.columns and "INDEX" in comp.columns:
        comp = comp.rename(columns={"INDEX": "WorldOrderIndex"})
    years = comp["year"].astype(int)
    fig, ax = plt.subplots(figsize=(10, 6))
    _shade_wars(ax)

    # Determine y-limits based on available composite values
    ymin, ymax = comp["WorldOrderIndex"].min(skipna=True), comp["WorldOrderIndex"].max(skipna=True)
    if pd.notna(ymin) and pd.notna(ymax):
        ax.set_ylim(max(-0.05, ymin), min(1.05, ymax))

    for iso3, g in comp.groupby("ISO3"):
        s = g.sort_values("year")["WorldOrderIndex"].reset_index(drop=True)
        y = moving_average(s, smooth)
        x = g.sort_values("year")["year"].values

        # Require at least 4 component metrics present originally to start the line
        # We approximate by requiring at least 4 non-null components in any row used for composite
        # If we saved components separately, we could count; here, skip if too sparse overall
        if g["WorldOrderIndex"].notna().sum() < 3:
            continue

        style = {"color": "#888888", "lw": 1.2, "alpha": 0.6}
        if iso3 in HIGHLIGHT:
            style.update(HIGHLIGHT[iso3])
        elif iso3 in wanted_iso3:
            style.update({"color": "#777777", "lw": 1.5, "alpha": 0.8})
        else:
            style.update({"linestyle": ":", "alpha": 0.35})

        ax.plot(x, y, label=iso3, **style)

    ax.set_xlim(1800, 2024)
    ax.set_title("World Order Index (Smoothed)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Composite (0-1)")
    # Legend: show only highlighted and selected
    handles, labels = ax.get_legend_handles_labels()
    show = [i for i, lab in enumerate(labels) if lab in set(list(HIGHLIGHT.keys()) + wanted_iso3)]
    if show:
        ax.legend([handles[i] for i in show], [labels[i] for i in show], loc="best")
    fig.tight_layout()
    out_path = out_dir / "composite.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_metrics_grid(metrics: pd.DataFrame, out_dir: Path, smooth: int = 5, top_n: int = 25) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["EDU", "MIL", "ECON", "TRAD", "RESV", "FIN", "INV", "CMPT"] if c in metrics.columns]
    if not cols:
        return out_dir / "metrics_grid.png"

    # Pick top N countries by count of non-null metrics
    metrics["non_null"] = metrics[cols].notna().sum(axis=1)
    counts = metrics.groupby(["ISO3", "country_name"])['non_null'].sum().sort_values(ascending=False)
    top = counts.head(top_n).reset_index()[["ISO3", "country_name"]]
    top_set = set(top["ISO3"].tolist())

    # Grid 4x2
    nrows, ncols = 4, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    years = metrics["year"].astype(int)
    for i, col in enumerate(cols[: nrows * ncols]):
        ax = axes[i]
        _shade_wars(ax)
        for iso3, g in metrics.groupby("ISO3"):
            if iso3 not in top_set:
                continue
            s = g.sort_values("year")[col].reset_index(drop=True)
            y = moving_average(s, smooth)
            x = g.sort_values("year")["year"].values
            style = {"color": "#777777", "lw": 1.0, "alpha": 0.7}
            if iso3 in HIGHLIGHT:
                style.update(HIGHLIGHT[iso3])
            else:
                style.update({"linestyle": ":", "alpha": 0.5})
            ax.plot(x, y, **style)
        ax.set_title(col)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(1800, 2024)
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j])
    fig.suptitle("Raw Metrics (Smoothed)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_path = out_dir / "metrics_grid.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_top25_country_grids(metrics: pd.DataFrame, out_dir: Path, smooth: int = 5, top_n: int = 25) -> Path:
    out_dir = Path(out_dir) / "top25"
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["EDU", "MIL", "ECON", "TRAD", "RESV", "FIN", "INV", "CMPT"] if c in metrics.columns]
    if not cols:
        return out_dir

    # Rank countries by total count of available metric entries (across all 8 metrics and years)
    metrics["non_null"] = metrics[cols].notna().sum(axis=1)
    cover = (
        metrics.groupby(["ISO3", "country_name"])['non_null']
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()[["ISO3", "country_name"]]
    )

    for _, row in cover.iterrows():
        iso3 = row["ISO3"]
        cname = row["country_name"]
        g = metrics[metrics["ISO3"] == iso3].sort_values("year")
        if g.empty:
            continue
        nrows, ncols = 4, 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        for i, col in enumerate(cols[: nrows * ncols]):
            ax = axes[i]
            _shade_wars(ax)
            s = g[col].reset_index(drop=True)
            y = moving_average(s, smooth)
            x = g["year"].values
            style = {"color": HIGHLIGHT.get(iso3, {}).get("color", "#1f77b4"), "lw": 2.0, "alpha": 0.9}
            ax.plot(x, y, **style)
            ax.set_title(col)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlim(1800, 2024)
        # Remove any unused axes
        last_i = len(cols[: nrows * ncols]) - 1
        for j in range(last_i + 1, nrows * ncols):
            fig.delaxes(axes[j])
        fig.suptitle(f"{cname} ({iso3}) — Metrics (Smoothed)")
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        # Safe filename
        safe_name = f"{iso3}.png" if isinstance(iso3, str) else f"{str(iso3)}.png"
        fig.savefig(out_dir / safe_name, dpi=200)
        plt.close(fig)

    return out_dir
