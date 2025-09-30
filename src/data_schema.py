"""
Data schema definitions for country standing forecast project.

Defines the 8 macro factors inspired by Ray Dalio's approach to measuring
country strength, along with their units, orientation, and transformations.
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field
import numpy as np


class FactorDefinition(BaseModel):
    """Definition of a single macro factor."""
    
    name: str = Field(..., description="Factor name")
    unit: str = Field(..., description="Unit of measurement")
    description: str = Field(..., description="Human-readable description")
    higher_is_better: bool = Field(..., description="Whether higher values indicate better standing")
    transform: Literal["level", "yoy", "log", "log_yoy"] = Field(
        default="level", 
        description="Transformation to apply"
    )
    min_value: Optional[float] = Field(None, description="Minimum plausible value")
    max_value: Optional[float] = Field(None, description="Maximum plausible value")


# Define the 8 core macro factors
FACTORS = {
    "education": FactorDefinition(
        name="education",
        unit="years",
        description="Average years of schooling (population 25+)",
        higher_is_better=True,
        transform="level",
        min_value=0.0,
        max_value=20.0
    ),
    "innovation": FactorDefinition(
        name="innovation", 
        unit="patents_per_million",
        description="Patent applications per million population",
        higher_is_better=True,
        transform="log",
        min_value=0.1,
        max_value=10000.0
    ),
    "competitiveness": FactorDefinition(
        name="competitiveness",
        unit="index",
        description="Global Competitiveness Index (0-100)",
        higher_is_better=True,
        transform="level",
        min_value=0.0,
        max_value=100.0
    ),
    "military": FactorDefinition(
        name="military",
        unit="expenditure_pct_gdp",
        description="Military expenditure as % of GDP",
        higher_is_better=True,
        transform="level",
        min_value=0.0,
        max_value=20.0
    ),
    "trade_share": FactorDefinition(
        name="trade_share",
        unit="pct_gdp",
        description="Trade (exports + imports) as % of GDP",
        higher_is_better=True,
        transform="level",
        min_value=0.0,
        max_value=200.0
    ),
    "reserve_currency_proxy": FactorDefinition(
        name="reserve_currency_proxy",
        unit="pct_global_reserves",
        description="Currency share in global foreign exchange reserves",
        higher_is_better=True,
        transform="level",
        min_value=0.0,
        max_value=100.0
    ),
    "financial_center_proxy": FactorDefinition(
        name="financial_center_proxy",
        unit="index",
        description="Financial center development index",
        higher_is_better=True,
        transform="level",
        min_value=0.0,
        max_value=100.0
    ),
    "debt": FactorDefinition(
        name="debt",
        unit="pct_gdp",
        description="Government debt as % of GDP (inverted - lower is better)",
        higher_is_better=False,
        transform="level",
        min_value=0.0,
        max_value=300.0
    )
}

# Factor names in order
FACTOR_NAMES = list(FACTORS.keys())

# Forecast horizons
FORECAST_HORIZONS = [1, 5, 10]  # years

# Window parameters
WINDOW_LENGTH = 20  # years
MIN_COUNTRIES = 5   # minimum countries needed for training
MIN_YEARS = 25      # minimum years of data needed per country


class CountryData(BaseModel):
    """Schema for country-level data."""
    
    country: str = Field(..., description="Country name")
    year: int = Field(..., description="Year")
    
    # Factor values
    education: Optional[float] = None
    innovation: Optional[float] = None
    competitiveness: Optional[float] = None
    military: Optional[float] = None
    trade_share: Optional[float] = None
    reserve_currency_proxy: Optional[float] = None
    financial_center_proxy: Optional[float] = None
    debt: Optional[float] = None
    
    # Optional exogenous variables
    gdp_per_capita: Optional[float] = None
    population: Optional[float] = None
    
    # Data quality flags
    mask_education: bool = Field(default=False, description="Missing data flag")
    mask_innovation: bool = Field(default=False, description="Missing data flag")
    mask_competitiveness: bool = Field(default=False, description="Missing data flag")
    mask_military: bool = Field(default=False, description="Missing data flag")
    mask_trade_share: bool = Field(default=False, description="Missing data flag")
    mask_reserve_currency_proxy: bool = Field(default=False, description="Missing data flag")
    mask_financial_center_proxy: bool = Field(default=False, description="Missing data flag")
    mask_debt: bool = Field(default=False, description="Missing data flag")


class ForecastTarget(BaseModel):
    """Schema for forecast targets."""
    
    country: str
    year: int
    horizon: int  # years ahead
    
    # Composite standing score (0-100)
    standing_score: float = Field(..., ge=0.0, le=100.0)
    
    # Optional: individual factor forecasts
    factor_forecasts: Optional[Dict[str, float]] = None


def get_factor_values(data: CountryData) -> np.ndarray:
    """Extract factor values as numpy array."""
    return np.array([
        data.education or 0.0,
        data.innovation or 0.0,
        data.competitiveness or 0.0,
        data.military or 0.0,
        data.trade_share or 0.0,
        data.reserve_currency_proxy or 0.0,
        data.financial_center_proxy or 0.0,
        data.debt or 0.0
    ])


def get_mask_values(data: CountryData) -> np.ndarray:
    """Extract mask values as numpy array."""
    return np.array([
        data.mask_education,
        data.mask_innovation,
        data.mask_competitiveness,
        data.mask_military,
        data.mask_trade_share,
        data.mask_reserve_currency_proxy,
        data.mask_financial_center_proxy,
        data.mask_debt
    ])


def compute_composite_standing(factor_values: np.ndarray) -> float:
    """
    Compute composite standing score from factor values.
    
    Uses weighted average with factor-specific weights and handles
    the debt factor (which is inverted - lower is better).
    """
    weights = np.array([0.15, 0.15, 0.15, 0.10, 0.15, 0.10, 0.10, 0.10])
    
    # Normalize factors to 0-100 scale
    normalized = np.zeros_like(factor_values)
    
    for i, (factor_name, factor_def) in enumerate(FACTORS.items()):
        if factor_name == "debt":
            # Invert debt (lower is better)
            normalized[i] = 100.0 - min(100.0, factor_values[i] * 100.0 / 300.0)
        else:
            # Normalize to 0-100 based on min/max values
            min_val = factor_def.min_value or 0.0
            max_val = factor_def.max_value or 100.0
            normalized[i] = min(100.0, max(0.0, 
                (factor_values[i] - min_val) * 100.0 / (max_val - min_val)
            ))
    
    return float(np.sum(weights * normalized))


# Data source information
DATA_SOURCES = {
    "education": {
        "primary": "World Bank WDI",
        "indicator": "SE.SEC.DURS.AG",
        "description": "Average years of schooling"
    },
    "innovation": {
        "primary": "WIPO Patent Statistics",
        "indicator": "patents_per_million",
        "description": "Patent applications per million population"
    },
    "competitiveness": {
        "primary": "WEF Global Competitiveness Report",
        "indicator": "gci_score",
        "description": "Global Competitiveness Index"
    },
    "military": {
        "primary": "SIPRI Military Expenditure Database",
        "indicator": "military_expenditure_pct_gdp",
        "description": "Military expenditure as % of GDP"
    },
    "trade_share": {
        "primary": "World Bank WDI",
        "indicator": "NE.TRD.GNFS.ZS",
        "description": "Trade as % of GDP"
    },
    "reserve_currency_proxy": {
        "primary": "IMF COFER",
        "indicator": "currency_reserves_pct",
        "description": "Currency share in global reserves"
    },
    "financial_center_proxy": {
        "primary": "GFCI (Global Financial Centres Index)",
        "indicator": "gfci_score",
        "description": "Financial center development"
    },
    "debt": {
        "primary": "World Bank WDI",
        "indicator": "GC.DOD.TOTL.GD.ZS",
        "description": "Government debt as % of GDP"
    }
}
