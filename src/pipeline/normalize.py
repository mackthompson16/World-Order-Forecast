from typing import List
import numpy as np
import pandas as pd


def compute_composite(wide: pd.DataFrame) -> pd.DataFrame:
    df = wide.copy()
    metric_cols = [c for c in df.columns if c not in {"Country", "Year"}]

    lower_is_better = {"GlobalDebt"}
    print(
        "[INFO] Normalizing metrics; all-time min-max by default. Competitiveness uses per-year normalization."
    )
    for m in metric_cols:
        norm_col = f"{m}_norm"
        df[norm_col] = np.nan
        # Special-case Competitiveness sources to reduce scale discontinuity
        if m in ("Competitiveness_GCI", "Competitiveness_component"):
            for yr, idx in df.groupby("Year").groups.items():
                s = pd.to_numeric(df.loc[idx, m], errors="coerce")
                valid = s.ge(0) & s.notna()
                if not valid.any():
                    continue
                v = s.where(valid)
                vmin, vmax = v.min(), v.max()
                if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
                    df.loc[idx, norm_col] = np.where(valid, 0.5, np.nan)
                else:
                    scaled = (v - vmin) / (vmax - vmin)
                    df.loc[idx, norm_col] = scaled
            continue

        # Default: all-time min-max across all years
        s_all = pd.to_numeric(df[m], errors="coerce")
        valid_mask_all = s_all.ge(0) & s_all.notna()
        if not valid_mask_all.any():
            print(f"[WARN] No valid values for metric {m}; normalized column will remain NaN")
            continue
        v = s_all.where(valid_mask_all)
        vmin, vmax = v.min(), v.max()
        if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
            df.loc[valid_mask_all, norm_col] = 0.5
        else:
            scaled_all = (v - vmin) / (vmax - vmin)
            if m in lower_is_better:
                scaled_all = 1.0 - scaled_all
            df.loc[valid_mask_all, norm_col] = scaled_all

    try:
        norm_cols = [c for c in df.columns if c.endswith("_norm")]
        print("[INFO] Normalized coverage (rows with non-NaN) and min/max per metric:")
        for nc in norm_cols:
            cnt = df[nc].notna().sum()
            mn = pd.to_numeric(df[nc], errors="coerce").min()
            mx = pd.to_numeric(df[nc], errors="coerce").max()
            print(f"  - {nc}: count={cnt} min={mn} max={mx}")
    except Exception:
        pass

    # Fuse Competitiveness: prefer GCI when available, else average components (but GCI excluded upstream)
    gci_col = "Competitiveness_GCI_norm"
    comp_col = "Competitiveness_component_norm"
    if gci_col in df.columns or comp_col in df.columns:
        df["Competitiveness_norm"] = np.nan
        if gci_col in df.columns:
            df.loc[df[gci_col].notna(), "Competitiveness_norm"] = df.loc[df[gci_col].notna(), gci_col]
        if comp_col in df.columns:
            mask = df["Competitiveness_norm"].isna() & df[comp_col].notna()
            df.loc[mask, "Competitiveness_norm"] = df.loc[mask, comp_col]
        # Remove underlying columns from consideration
        drop_these: List[str] = []
        if gci_col in df.columns:
            drop_these.append(gci_col)
        if comp_col in df.columns:
            drop_these.append(comp_col)
        df.drop(columns=drop_these, inplace=True, errors="ignore")

    # Rebuild normalized list and compute composite
    norm_cols = [c for c in df.columns if c.endswith("_norm")]
    # AvailableCount based on actual (non-filled) availability
    df["AvailableCount"] = df[norm_cols].notna().sum(axis=1)

    # To reduce jumps when a metric first appears, extend the first available
    # normalized value backward in time for that country/metric. We use these
    # filled values for averaging, but do NOT change AvailableCount or plotting gates.
    df_sorted = df.sort_values(["Country", "Year"]).copy()
    for col in norm_cols:
        # Fill both ends: first extend backward (pre-appearance), then forward (post-last observation)
        df_sorted[col + "__filled"] = df_sorted.groupby("Country")[col].transform(lambda s: s.bfill().ffill())
    filled_cols = [c + "__filled" for c in norm_cols]
    # Compute composite mean across filled values
    df_sorted["CompositeStanding"] = df_sorted[filled_cols].mean(axis=1, skipna=True)
    # Move CompositeStanding back to original df order
    df = df_sorted.sort_index(axis=0)

    keep_cols = ["Country", "Year", "CompositeStanding", "AvailableCount"] + metric_cols + norm_cols
    return df[keep_cols]


__all__ = ["compute_composite"]
