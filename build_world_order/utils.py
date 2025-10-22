from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


# Year bounds for analysis
YEAR_MIN = 1800
YEAR_MAX = 2024


# Minimal alias map to canonical names used in outputs
_ALIASES: Dict[str, str] = {
    # United States
    "UNITED STATES": "USA",
    "US": "USA",
    "U.S.": "USA",
    "U.S.A.": "USA",
    "USA": "USA",
    "UNITED STATES OF AMERICA": "USA",
    # China
    "CHINA": "CHINA",
    "PEOPLE'S REPUBLIC OF CHINA": "CHINA",
    "PRC": "CHINA",
    "CHN": "CHINA",
    # United Kingdom
    "UNITED KINGDOM": "UNITED KINGDOM",
    "UK": "UNITED KINGDOM",
    "UKG": "UNITED KINGDOM",
    "GREAT BRITAIN": "UNITED KINGDOM",
    "BRITAIN": "UNITED KINGDOM",
    "GBR": "UNITED KINGDOM",
    # France
    "FRANCE": "FRANCE",
    "FRN": "FRANCE",
    "FRA": "FRANCE",
    # Germany
    "GERMANY": "GERMANY",
    "GMY": "GERMANY",
    "DEU": "GERMANY",
    "FEDERAL REPUBLIC OF GERMANY": "GERMANY",
    # Russia / USSR
    "RUSSIA": "RUSSIA",
    "RUSSIAN FEDERATION": "RUSSIA",
    "RUS": "RUSSIA",
    "SOVIET UNION": "RUSSIA",
    "USSR": "RUSSIA",
}


def canonical_country(name: str | None) -> str | None:
    if name is None:
        return None
    key = name.strip().upper()
    # If name already looks like ISO3 for key countries, map via aliases
    if key in _ALIASES:
        return _ALIASES[key]
    return key


SELECTED_COUNTRIES = [
    "CHINA",
    "USA",
    "FRANCE",
    "GERMANY",
    "UNITED KINGDOM",
    "RUSSIA",
]


@dataclass(frozen=True)
class WarPeriod:
    label: str
    start: int
    end: int


WW1 = WarPeriod("WWI", 1914, 1918)
WW2 = WarPeriod("WWII", 1939, 1945)
