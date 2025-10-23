import pandas as pd
from pathlib import Path
from typing import Dict, Tuple


def _canonicalize_country_names(df: pd.DataFrame, name_col: str) -> pd.DataFrame:
    df = df.copy()
    df[name_col] = (
        df[name_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace("\u00A0", " ", regex=False)
    )
    # Common aliases
    replacements = {
        "UNITED STATES": "UNITED STATES OF AMERICA",
        "USA": "UNITED STATES OF AMERICA",
        "US": "UNITED STATES OF AMERICA",
        "U.S.": "UNITED STATES OF AMERICA",
        "U.S.A.": "UNITED STATES OF AMERICA",
        "UK": "UNITED KINGDOM",
        "RUSSIA": "RUSSIAN FEDERATION",
        "IRAN": "IRAN (ISLAMIC REPUBLIC OF)",
        "SOUTH KOREA": "KOREA, REPUBLIC OF",
        "NORTH KOREA": "KOREA, DEMOCRATIC PEOPLE'S REPUBLIC OF",
        "CZECHIA": "CZECH REPUBLIC",
        "BOLIVIA": "BOLIVIA, PLURINATIONAL STATE OF",
        "VENEZUELA": "VENEZUELA, BOLIVARIAN REPUBLIC OF",
        "SYRIA": "SYRIAN ARAB REPUBLIC",
        "LAOS": "LAO PEOPLE'S DEMOCRATIC REPUBLIC",
        "MOLDOVA": "MOLDOVA, REPUBLIC OF",
        "CONGO": "CONGO",
    }
    df[name_col] = df[name_col].replace(replacements)
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
    cow = cow.rename(columns={"cow_id": "COW", "iso_id": "code_iso"})
    cow["COW"] = pd.to_numeric(cow["COW"], errors="coerce").astype("Int64")
    cow["code_iso"] = pd.to_numeric(cow["code_iso"], errors="coerce").astype("Int64")

    # Join on numeric ISO numeric code
    id_df = cn.merge(cow[["COW", "code_iso"]], left_on="code", right_on="code_iso", how="left")
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
    # Map CODE->abv via country_id
    code_to_abv = dict(country_id[["code", "abv"]].dropna().values)
    edu_long["ISO3"] = edu_long["CODE"].map(code_to_abv)
    edu_long = edu_long.dropna(subset=["ISO3", "year"]).copy()
    return edu_long[["ISO3", "country_name", "year", "education"]]


def load_military(data_dir: Path, country_id: pd.DataFrame) -> pd.DataFrame:
    # military.csv: [CCODE = COW : CINC]
    mil = pd.read_csv(data_dir / "military.csv")
    mil = mil.rename(columns={"country": "country_name", "cinc": "CINC", "year": "year", "ccode": "COW"})
    mil["COW"] = pd.to_numeric(mil["COW"], errors="coerce").astype("Int64")
    cow_to_abv = dict(country_id[["COW", "abv"]].dropna().values)
    mil["ISO3"] = mil["COW"].map(cow_to_abv)
    mil = mil.dropna(subset=["ISO3", "year"]).copy()
    return mil[["ISO3", "country_name", "year", "CINC"]]


def load_polity(data_dir: Path, country_id: pd.DataFrame) -> pd.DataFrame:
    # polity.csv: [CCODE = COW : xconst, parcomp]
    pol = pd.read_csv(data_dir / "polity.csv")
    pol = pol.rename(columns={"country": "country_name", "ccode": "COW"})
    pol["COW"] = pd.to_numeric(pol["COW"], errors="coerce").astype("Int64")
    cow_to_abv = dict(country_id[["COW", "abv"]].dropna().values)
    pol["ISO3"] = pol["COW"].map(cow_to_abv)
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
