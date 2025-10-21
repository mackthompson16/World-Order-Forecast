import re
from typing import Optional


COUNTRY_ALIASES = {
    "united states of america": "United States",
    "united states": "United States",
    "usa": "United States",
    "us": "United States",
    "u.s.": "United States",
    "uk": "United Kingdom",
    "united kingdom of great britain and northern ireland": "United Kingdom",
    "russian federation": "Russia",
    "korea, rep.": "South Korea",
    "korea, republic of": "South Korea",
    "korea, dpr": "North Korea",
    "korea, dem. people's rep.": "North Korea",
    "czech republic": "Czechia",
    "iran, islamic rep.": "Iran",
    "egypt, arab rep.": "Egypt",
    "viet nam": "Vietnam",
    "hong kong sar, china": "Hong Kong",
    "macao sar, china": "Macao",
    "china, mainland": "China",
    "taiwan, china": "Taiwan",
    "kyrgyz republic": "Kyrgyzstan",
    "lao pdr": "Laos",
    "congo, dem. rep.": "DR Congo",
    "congo, rep.": "Congo",
    "gambia, the": "Gambia",
    "bahamas, the": "Bahamas",
    "swaziland": "Eswatini",
    "north macedonia": "North Macedonia",
}

AGGREGATE_PREFIXES = [
    "world",
    "europe",
    "asia",
    "africa",
    "oceania",
    "latin america",
    "europe & central asia",
    "middle east",
    "north america",
    "euro area",
    "oecd",
    "g7",
    "g20",
    "high income",
    "upper middle income",
    "lower middle income",
    "low income",
    "arab world",
    "caribbean",
    "sub-saharan africa",
    "south asia",
    "east asia & pacific",
    "european union",
    "commonwealth",
    "former ussr",
]


def standardize_country(name: str) -> Optional[str]:
    if not isinstance(name, str):
        return None
    n = name.strip()
    if not n:
        return None
    key = n.lower()
    if key in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[key]
    for pref in AGGREGATE_PREFIXES:
        if key.startswith(pref):
            return None
    n = re.sub(r"\s*,\s*total$", "", n, flags=re.I)
    n = re.sub(r"\s*\(.*?\)\s*", "", n).strip()
    return n


__all__ = ["standardize_country", "COUNTRY_ALIASES", "AGGREGATE_PREFIXES"]
