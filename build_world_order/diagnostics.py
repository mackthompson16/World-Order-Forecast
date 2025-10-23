from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def innovation_breakdown(chat_path: Path, out_dir: Path, year: int) -> Path:
    chat = pd.read_csv(chat_path)
    id_cols = ["ISO3", "country_name", "year"]
    tech_cols = [c for c in chat.columns if c not in id_cols]
    for c in tech_cols:
        chat[c] = pd.to_numeric(chat[c], errors="coerce")
    # Aggregate to unique ISO3-year
    grp = chat.groupby(["ISO3", "country_name", "year"], as_index=False).mean(numeric_only=True)
    ydf = grp[grp["year"] == year].copy()
    if ydf.empty:
        raise SystemExit(f"No CHAT rows for year {year}")
    # Yearwise min-max per tech
    normed = ydf[["ISO3", "country_name", "year"]].copy()
    cover = {}
    for col in tech_cols:
        vals = ydf[col].dropna()
        cover[col] = int(vals.shape[0])
        if len(vals) < 2:
            normed[col] = np.nan
            continue
        vmin, vmax = vals.min(), vals.max()
        if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
            normed[col] = np.nan
        else:
            normed[col] = (ydf[col] - vmin) / (vmax - vmin)
    norm_cols = [c for c in tech_cols if c in normed.columns]
    normed["INV"] = normed[norm_cols].mean(axis=1)
    normed = normed.sort_values("INV", ascending=False)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"INV_breakdown_{year}.csv"
    normed.to_csv(out_path, index=False)

    cover_path = out_dir / f"INV_coverage_{year}.csv"
    pd.DataFrame({"tech": list(cover.keys()), "n_countries": list(cover.values())}).sort_values("n_countries", ascending=False).to_csv(cover_path, index=False)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Diagnostics for Innovation calculation")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--out-dir", default="build_world_order/results/diagnostics")
    args = p.parse_args()
    chat_path = Path(args.data_dir) / "CHAT.csv"
    out = innovation_breakdown(chat_path, Path(args.out_dir), args.year)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

