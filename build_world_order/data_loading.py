import pandas as pd
from pathlib import Path
from typing import Dict, Tuple


def _canonicalize_country_names(df: pd.DataFrame, name_col: str) -> pd.DataFrame:
    df = df.copy()
    df[name_col] = (
        df[name_col]
        .astype(str)
        .str.replace("\u00A0", " ", regex=False)
        .str.strip()
    )
    return df


def build_country_id(data_dir: Path) -> Path:
    """Builds data/country_id.csv with columns [name, abv, code, COW].

    - country_names.csv provides [name, alpha-3 -> abv, country-code -> code]
    - cow2iso.csv provides [cow_id -> COW, iso_id -> code]
    - When reading ids, convert to int and trim leading zeros
    """
    data_dir = Path(data_dir)
    cn = pd.read_csv(data_dir / "country_names.csv")
    cn = cn.rename(columns={"name": "name", "alpha-3": "abv", "country-code": "code"})
    cn["abv"] = cn["abv"].astype(str).str.upper()
    # Ensure numeric code with trimmed leading zeros
    cn["code"] = pd.to_numeric(cn["code"], errors="coerce").astype("Int64")
    cn = _canonicalize_country_names(cn, "name")

    cow = pd.read_csv(data_dir / "cow2iso.csv")
    cow = cow.rename(columns={"cow_id": "COW", "iso3": "iso3"})
    cow["COW"] = pd.to_numeric(cow["COW"], errors="coerce").astype("Int64")
    # Normalize iso3 for join
    cow["iso3"] = cow["iso3"].astype(str).str.upper()

    # Prefer join on ISO3 for robust mapping; keep numeric ISO code from country_names
    id_df = cn.merge(cow[["iso3", "COW"]], left_on="abv", right_on="iso3", how="left")
    id_df = id_df[["name", "abv", "code", "COW"]].drop_duplicates()

    out_path = data_dir / "country_id.csv"
    id_df.to_csv(out_path, index=False)
    return out_path


def load_country_reference(data_dir: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Loads country_id.csv if present; otherwise builds it first.

    Returns (country_id_df, name_to_iso3 map) for convenience.
    """
    data_dir = Path(data_dir)
    path = data_dir / "country_id.csv"
    if not path.exists():
        build_country_id(data_dir)
    ref = pd.read_csv(path)
    ref["abv"] = ref["abv"].astype(str).str.upper()
    # name->ISO3 mapping
    name_to_iso3 = {}
    for _, row in ref.iterrows():
        if pd.notna(row.get("name")) and pd.notna(row.get("abv")):
            name_to_iso3[str(row["name"]).upper()] = row["abv"]
    return ref, name_to_iso3


def load_gmd(data_dir: Path) -> pd.DataFrame:
    gmd = pd.read_csv(data_dir / "GMD.csv")
    keep = [
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
        "countryname",
    ]
    cols = [c for c in keep if c in gmd.columns]
    gmd = gmd[cols].copy()
    gmd = gmd.rename(columns={"countryname": "country_name"})
    gmd["ISO3"] = gmd["ISO3"].astype(str).str.upper()
    gmd["year"] = pd.to_numeric(gmd["year"], errors="coerce")
    return gmd


def load_education(data_dir: Path, country_id: pd.DataFrame) -> pd.DataFrame:
    # Education.csv: [CCODE = CODE : year]
    edu = pd.read_csv(data_dir / "Education.csv")
    edu = edu.rename(columns={"country name": "country_name", "ccode": "CODE"})
    # Melt wide years
    year_cols = [c for c in edu.columns if c not in ("CODE", "country_name")]
    edu_long = edu.melt(id_vars=["CODE", "country_name"], value_vars=year_cols, var_name="year", value_name="education")
    edu_long["year"] = pd.to_numeric(edu_long["year"], errors="coerce")
    edu_long["CODE"] = pd.to_numeric(edu_long["CODE"], errors="coerce").astype("Int64")
    # Map CODE->abv via merge to avoid dtype/key issues
    ref = country_id[["code", "abv"]].dropna().copy()
    ref["code"] = pd.to_numeric(ref["code"], errors="coerce").astype("Int64")
    edu_long = edu_long.merge(ref, left_on="CODE", right_on="code", how="left")
    edu_long = edu_long.rename(columns={"abv": "ISO3"})
    edu_long = edu_long.dropna(subset=["ISO3", "year"]).copy()
    return edu_long[["ISO3", "country_name", "year", "education"]]


def load_military(data_dir: Path, country_id: pd.DataFrame) -> pd.DataFrame:
    # military.csv: [CCODE = COW : CINC]
    mil = pd.read_csv(data_dir / "military.csv")
    mil = mil.rename(columns={"country": "country_name", "cinc": "CINC", "year": "year", "ccode": "COW"})
    mil["COW"] = pd.to_numeric(mil["COW"], errors="coerce").astype("Int64")
    ref = country_id[["COW", "abv"]].dropna().copy()
    ref["COW"] = pd.to_numeric(ref["COW"], errors="coerce").astype("Int64")
    mil = mil.merge(ref, on="COW", how="left").rename(columns={"abv": "ISO3"})
    mil = mil.dropna(subset=["ISO3", "year"]).copy()
    return mil[["ISO3", "country_name", "year", "CINC"]]


def load_polity(data_dir: Path, country_id: pd.DataFrame) -> pd.DataFrame:
    # polity.csv: [CCODE = COW : xconst, parcomp]
    pol = pd.read_csv(data_dir / "polity.csv")
    pol = pol.rename(columns={"country": "country_name", "ccode": "COW"})
    pol["COW"] = pd.to_numeric(pol["COW"], errors="coerce").astype("Int64")
    ref = country_id[["COW", "abv"]].dropna().copy()
    ref["COW"] = pd.to_numeric(ref["COW"], errors="coerce").astype("Int64")
    pol = pol.merge(ref, on="COW", how="left").rename(columns={"abv": "ISO3"})
    use_cols = [c for c in ["ISO3", "country_name", "year", "xconst", "parcomp"] if c in pol.columns]
    pol = pol[use_cols].copy()
    pol["year"] = pd.to_numeric(pol["year"], errors="coerce")
    pol = pol.dropna(subset=["ISO3", "year"]).copy()
    return pol


def load_chat(data_dir: Path, country_id: pd.DataFrame) -> pd.DataFrame:
    # CHAT.csv contains country_name; map to ISO3 via canonical names from country_id
    chat = pd.read_csv(data_dir / "CHAT.csv")
    # If file already contains ISO3 column, trust it; otherwise map via name
    if "ISO3" not in chat.columns and "country_name" in chat.columns:
        chat = _canonicalize_country_names(chat, "country_name")
        name_to_abv = {str(n).upper(): a for n, a in country_id[["name", "abv"]].dropna().values}
        chat["ISO3"] = chat["country_name"].map(name_to_abv)
        # Fallback: prefix-based name matching when exact match not found
        missing_mask = chat["ISO3"].isna()
        if missing_mask.any():
            keys = list(name_to_abv.keys())
            def try_prefix(nm: str):
                if not isinstance(nm, str) or len(nm) < 3:
                    return None
                # Candidates where either direction is a prefix
                cands = [k for k in keys if k.startswith(nm) or nm.startswith(k)]
                cands_iso = list({name_to_abv[k] for k in cands})
                return cands_iso[0] if len(cands_iso) == 1 else None
            chat.loc[missing_mask, "ISO3"] = chat.loc[missing_mask, "country_name"].apply(try_prefix)
    elif "ISO3" in chat.columns:
        chat["ISO3"] = chat["ISO3"].astype(str).str.upper()
    else:
        # No way to map — return empty frame with expected columns
        return pd.DataFrame(columns=["ISO3", "country_name", "year"])

    chat["year"] = pd.to_numeric(chat.get("year"), errors="coerce")
    chat = chat.dropna(subset=["ISO3", "year"]).copy()
    return chat


def load_chat_strict(data_dir: Path, country_id: pd.DataFrame) -> pd.DataFrame:
    """Strict CHAT loader:
    - If ISO3 present, use it (uppercased).
    - Else map country_name to ISO3 via country_id.csv with case-insensitive exact matching.
    - No hardcoded alias tables or fuzzy prefix matching.
    """
    chat = pd.read_csv(data_dir / "CHAT.csv")
    if "ISO3" in chat.columns:
        chat["ISO3"] = chat["ISO3"].astype(str).str.upper()
    elif "country_name" in chat.columns:
        # Normalize whitespace only
        chat = _canonicalize_country_names(chat, "country_name")
        ref = _canonicalize_country_names(country_id.copy(), "name")
        key_chat = chat["country_name"].astype(str).str.upper()
        key_ref = ref["name"].astype(str).str.upper()
        name_to_abv = {n: a for n, a in zip(key_ref, ref["abv"])}
        chat["ISO3"] = key_chat.map(name_to_abv)
    else:
        return pd.DataFrame(columns=["ISO3", "country_name", "year"])

    chat["year"] = pd.to_numeric(chat.get("year"), errors="coerce")
    chat = chat.dropna(subset=["ISO3", "year"]).copy()
    return chat
