from pathlib import Path
import argparse
import pandas as pd

from .data_loading import (
    build_country_id,
    load_country_reference,
    load_gmd,
    load_education,
    load_military,
    load_polity,
    load_chat,
)
from .utils import interpolate_panel


def build_clean_data(data_dir: Path, overwrite: bool = False) -> Path:
    data_dir = Path(data_dir)
    out_path = data_dir / "clean_data.csv"
    if out_path.exists() and not overwrite:
        return out_path

    # Ensure country_id.csv exists
    build_country_id(data_dir)
    ref, name_to_iso3 = load_country_reference(data_dir)
    gmd = load_gmd(data_dir)
    edu = load_education(data_dir, ref)
    mil = load_military(data_dir, ref)
    pol = load_polity(data_dir, ref)

    # Merge outer on ISO3, year; preserve country_name from first available
    def merge_and_unify(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        m = pd.merge(left, right, on=["ISO3", "year"], how="outer", suffixes=("_l", "_r"))
        # Consolidate country_name columns immediately to avoid cascading suffix issues
        name_cols = [c for c in m.columns if c.startswith("country_name")]
        if name_cols:
            m["country_name"] = m[name_cols].bfill(axis=1).ffill(axis=1).iloc[:, 0]
            for c in name_cols:
                if c != "country_name":
                    m = m.drop(columns=c)
        return m

    base = merge_and_unify(gmd, edu)
    base = merge_and_unify(base, mil)
    base = merge_and_unify(base, pol)

    # Ensure a country_name for all rows
    if "country_name" not in base.columns or base["country_name"].isna().any():
        iso_to_name = dict(ref[["ISO3", "country_name"]].values)
        base["country_name"] = base["ISO3"].map(iso_to_name).fillna(base.get("country_name"))

    # Interpolate inside each country between first and last available
    value_cols = [
        c
        for c in [
            "rGDP_USD",
            "USDfx",
            "cgovdebt_GDP",
            "exports_USD",
            "imports_USD",
            "M0",
            "finv_GDP",
            "CA_USD",
            "pop",
            "education",
            "CINC",
            "xconst",
            "parcomp",
        ]
        if c in base.columns
    ]
    base = interpolate_panel(base, ["ISO3"], value_cols)

    # Keep only rows that have at least one datapoint among the value_cols
    has_any = base[value_cols].notna().any(axis=1)
    base = base.loc[has_any].copy()

    # Reorder columns
    ordered = [
        "country_name",
        "ISO3",
        "year",
        "rGDP_USD",
        "USDfx",
        "cgovdebt_GDP",
        "exports_USD",
        "imports_USD",
        "M0",
        "finv_GDP",
        "CA_USD",
        "pop",
        "education",
        "CINC",
        "xconst",
        "parcomp",
    ]
    cols = [c for c in ordered if c in base.columns]
    base[cols].to_csv(out_path, index=False)
    return out_path


def interpolate_chat_inplace(data_dir: Path) -> Path:
    data_dir = Path(data_dir)
    chat_path = data_dir / "CHAT.csv"
    ref, _ = load_country_reference(data_dir)
    chat = load_chat(data_dir, ref)

    # Accept files with or without country_name; always require ISO3 and year
    present_id = [c for c in ["ISO3", "country_name", "year"] if c in chat.columns]
    if not ("ISO3" in present_id and "year" in present_id):
        raise ValueError("CHAT.csv must contain at least ISO3 and year columns after mapping")
    id_cols = present_id
    value_cols = [c for c in chat.columns if c not in id_cols]
    # Ensure numeric for metrics prior to interpolation
    for c in value_cols:
        chat[c] = pd.to_numeric(chat[c], errors="coerce")
    chat_interp = interpolate_panel(chat, ["ISO3"], value_cols)

    # Drop rows where no metric is available
    has_any = chat_interp[value_cols].notna().any(axis=1)
    chat_interp = chat_interp.loc[has_any, id_cols + value_cols].copy()
    # Overwrite raw CHAT.csv in place
    chat_interp.to_csv(chat_path, index=False)
    return chat_path


# Removed CHAT_MAPPED helper; we now write interpolated/mapped data back to CHAT.csv


def main():
    parser = argparse.ArgumentParser(description="Build clean_data.csv and interpolate CHAT.csv")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing clean_data.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    clean_path = build_clean_data(data_dir, overwrite=args.overwrite)
    chat_path = interpolate_chat_inplace(data_dir)
    print(f"Wrote: {clean_path}")
    print(f"Interpolated CHAT in place: {chat_path}")


if __name__ == "__main__":
    main()
