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


# Define corruption and geography constants
CORRUPTION_CONSTANTS = {
    "very_low": {"score": 0.95, "description": "Highly transparent institutions"},
    "low": {"score": 0.85, "description": "Good governance, minor issues"},
    "moderate": {"score": 0.70, "description": "Some corruption, data reliability concerns"},
    "high": {"score": 0.50, "description": "Significant corruption, unreliable data"},
    "very_high": {"score": 0.25, "description": "Severe corruption, highly unreliable data"}
}

GEOGRAPHY_CONSTANTS = {
    "island_advantage": {"multiplier": 1.15, "description": "Natural defense, trade advantages"},
    "coastal_access": {"multiplier": 1.10, "description": "Maritime trade, resource access"},
    "strategic_location": {"multiplier": 1.05, "description": "Geographic strategic importance"},
    "landlocked": {"multiplier": 0.90, "description": "Limited trade routes, dependency"},
    "resource_rich": {"multiplier": 1.08, "description": "Natural resource abundance"},
    "arctic_challenges": {"multiplier": 0.85, "description": "Harsh climate, infrastructure costs"},
    "desert_limitations": {"multiplier": 0.88, "description": "Water scarcity, agricultural challenges"},
    "mountain_barriers": {"multiplier": 0.92, "description": "Transportation difficulties"}
}

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
    ),
    "corruption_index": FactorDefinition(
        name="corruption_index",
        unit="trust_score",
        description="Data trustworthiness based on corruption levels (0-1 scale)",
        higher_is_better=True,
        transform="level",
        min_value=0.0,
        max_value=1.0
    ),
    "geography_advantage": FactorDefinition(
        name="geography_advantage",
        unit="multiplier",
        description="Geographic advantage multiplier for optimistic curves",
        higher_is_better=True,
        transform="level",
        min_value=0.5,
        max_value=1.5
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
    corruption_index: Optional[float] = None
    geography_advantage: Optional[float] = None
    
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
    mask_corruption_index: bool = Field(default=False, description="Missing data flag")
    mask_geography_advantage: bool = Field(default=False, description="Missing data flag")


# Define industry/company factors
COMPANY_FACTORS = {
    "market_cap_billions": FactorDefinition(
        name="market_cap_billions",
        unit="billions_usd",
        description="Total market capitalization by industry sector",
        higher_is_better=True,
        transform="log",
        min_value=1.0,
        max_value=50000.0
    ),
    "rd_spending_percent": FactorDefinition(
        name="rd_spending_percent",
        unit="pct_revenue",
        description="R&D expenditure as % of revenue",
        higher_is_better=True,
        transform="level",
        min_value=0.0,
        max_value=30.0
    ),
    "employment_share_percent": FactorDefinition(
        name="employment_share_percent",
        unit="pct_total_employment",
        description="Employment as % of total workforce",
        higher_is_better=True,
        transform="level",
        min_value=0.1,
        max_value=50.0
    ),
    "productivity_index": FactorDefinition(
        name="productivity_index",
        unit="index_2000_100",
        description="Labor productivity index (base year 2000 = 100)",
        higher_is_better=True,
        transform="level",
        min_value=50.0,
        max_value=500.0
    )
}

COMPANY_FACTOR_NAMES = list(COMPANY_FACTORS.keys())


class CompanyData(BaseModel):
    """Schema for company/industry-level data."""
    
    industry: str = Field(..., description="Industry sector name")
    year: int = Field(..., description="Year")
    
    # Factor values
    market_cap_billions: Optional[float] = None
    rd_spending_percent: Optional[float] = None
    employment_share_percent: Optional[float] = None
    productivity_index: Optional[float] = None
    
    # Optional additional metrics
    revenue_growth_rate: Optional[float] = None
    patent_filings: Optional[int] = None
    energy_consumption: Optional[float] = None
    
    # Data quality flags
    mask_market_cap: bool = Field(default=False, description="Missing data flag")
    mask_rd_spending: bool = Field(default=False, description="Missing data flag")
    mask_employment_share: bool = Field(default=False, description="Missing data flag")
    mask_productivity: bool = Field(default=False, description="Missing data flag")


class ForecastTarget(BaseModel):
    """Schema for forecast targets."""
    
    country: str
    year: int
    horizon: int  # years ahead
    
    # Composite standing score (0-100)
    standing_score: float = Field(..., ge=0.0, le=100.0)
    
    # Optional: individual factor forecasts
    factor_forecasts: Optional[Dict[str, float]] = None


class CompanyForecastTarget(BaseModel):
    """Schema for company/industry forecast targets."""
    
    industry: str
    year: int
    horizon: int  # years ahead
    
    # Composite market dominance score (0-100)
    dominance_score: float = Field(..., ge=0.0, le=100.0)
    
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
        data.debt or 0.0,
        data.corruption_index or 0.5,  # Default to moderate corruption
        data.geography_advantage or 1.0  # Default to neutral geography
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
        data.mask_debt,
        data.mask_corruption_index,
        data.mask_geography_advantage
    ])


def get_company_factor_values(data: CompanyData) -> np.ndarray:
    """Extract company factor values as numpy array."""
    return np.array([
        data.market_cap_billions or 0.0,
        data.rd_spending_percent or 0.0,
        data.employment_share_percent or 0.0,
        data.productivity_index or 100.0  # Default to base year value
    ])


def get_company_mask_values(data: CompanyData) -> np.ndarray:
    """Extract company mask values as numpy array."""
    return np.array([
        data.mask_market_cap,
        data.mask_rd_spending,
        data.mask_employment_share,
        data.mask_productivity
    ])


def compute_composite_standing(factor_values: np.ndarray) -> float:
    """
    Compute base composite standing score from factor values (without corruption/geography effects).
    
    Uses weighted average with factor-specific weights and handles
    the debt factor (which is inverted - lower is better).
    """
    # Weights for 8 core factors (excluding corruption_index and geography_advantage)
    weights = np.array([0.15, 0.15, 0.15, 0.10, 0.15, 0.10, 0.10, 0.10])
    
    # Use only the first 8 factors for base score calculation
    core_factors = factor_values[:8]
    
    # Normalize factors to 0-100 scale
    normalized = np.zeros_like(core_factors)
    
    core_factor_names = list(FACTORS.keys())[:8]  # First 8 factors
    
    for i, (factor_name, factor_def) in enumerate(zip(core_factor_names, [FACTORS[name] for name in core_factor_names])):
        if factor_name == "debt":
            # Invert debt (lower is better)
            normalized[i] = 100.0 - min(100.0, core_factors[i] * 100.0 / 300.0)
        else:
            # Normalize to 0-100 based on min/max values
            min_val = factor_def.min_value or 0.0
            max_val = factor_def.max_value or 100.0
            if max_val > min_val:
                normalized[i] = min(100.0, max(0.0, 
                    (core_factors[i] - min_val) * 100.0 / (max_val - min_val)
                ))
            else:
                normalized[i] = 50.0  # Default neutral value
    
    # Calculate base composite score
    base_score = float(np.sum(weights * normalized))
    
    return float(np.clip(base_score, 0.0, 100.0))


def compute_data_confidence(factor_values: np.ndarray) -> float:
    """
    Compute data confidence/trust score based on corruption levels.
    
    This affects learning rate and gradient trust, not the final score directly.
    
    Args:
        factor_values: Array including corruption_index at index 8
        
    Returns:
        Confidence score (0.0 to 1.0) - higher means more trustworthy data
    """
    if len(factor_values) > 8:
        corruption_index = factor_values[8]  # corruption_index is 9th factor (index 8)
        return float(corruption_index)  # Direct mapping: corruption_index = confidence
    else:
        return 0.5  # Default moderate confidence


def compute_geography_growth_multiplier(factor_values: np.ndarray) -> float:
    """
    Compute geography-based growth multiplier for derivatives/trends.
    
    This affects the rate of change over time, not the current score.
    
    Args:
        factor_values: Array including geography_advantage at index 9
        
    Returns:
        Growth multiplier (0.5 to 1.5) - higher means more optimistic growth trends
    """
    if len(factor_values) > 9:
        geography_advantage = factor_values[9]  # geography_advantage is 10th factor (index 9)
        return float(geography_advantage)  # Direct mapping: geography_advantage = growth multiplier
    else:
        return 1.0  # Default neutral growth


def compute_learning_rate_adjustment(data_confidence: float, base_learning_rate: float = 0.01) -> float:
    """
    Adjust learning rate based on data confidence.
    
    Higher confidence = higher learning rate (trust gradients more)
    Lower confidence = lower learning rate (be more conservative)
    
    Args:
        data_confidence: Confidence score (0.0 to 1.0)
        base_learning_rate: Base learning rate
        
    Returns:
        Adjusted learning rate
    """
    # Scale learning rate by confidence
    # High confidence (0.95) -> 1.0x learning rate
    # Low confidence (0.25) -> 0.25x learning rate
    adjusted_lr = base_learning_rate * data_confidence
    return float(np.clip(adjusted_lr, base_learning_rate * 0.1, base_learning_rate * 1.5))


def compute_trend_adjustment(geography_multiplier: float, base_trend: float) -> float:
    """
    Adjust growth trend based on geography advantage.
    
    Higher geography multiplier = more optimistic trend (higher derivative)
    Lower geography multiplier = more pessimistic trend (lower derivative)
    
    Args:
        geography_multiplier: Geography growth multiplier (0.5 to 1.5)
        base_trend: Base trend/growth rate
        
    Returns:
        Adjusted trend/growth rate
    """
    # Multiply trend by geography advantage
    adjusted_trend = base_trend * geography_multiplier
    return float(adjusted_trend)


def get_country_corruption_level(country_code: str) -> str:
    """
    Get corruption level for a country based on Transparency International data and governance indicators.
    
    Returns one of: "very_low", "low", "moderate", "high", "very_high"
    """
    # Based on Transparency International Corruption Perceptions Index and governance data
    corruption_levels = {
        # Very low corruption (CPI > 80)
        "DNK": "very_low", "FIN": "very_low", "NZL": "very_low", "NOR": "very_low", 
        "SWE": "very_low", "CHE": "very_low", "SGP": "very_low", "NLD": "very_low",
        
        # Low corruption (CPI 70-80)
        "AUS": "low", "CAN": "low", "DEU": "low", "GBR": "low", "AUT": "low",
        "BEL": "low", "IRL": "low", "JPN": "low", "EST": "low", "ISL": "low",
        
        # Moderate corruption (CPI 50-70)
        "USA": "moderate", "FRA": "moderate", "ITA": "moderate", "ESP": "moderate",
        "KOR": "moderate", "PRT": "moderate", "POL": "moderate", "CZE": "moderate",
        "CHL": "moderate", "ISR": "moderate", "LTU": "moderate", "LVA": "moderate",
        
        # High corruption (CPI 30-50)
        "IND": "high", "BRA": "high", "CHN": "high", "MEX": "high", "TUR": "high",
        "RUS": "high", "IDN": "high", "THA": "high", "SAU": "high", "ARE": "high",
        
        # Very high corruption (CPI < 30)
        "VEN": "very_high", "MMR": "very_high", "PRK": "very_high", "IRN": "very_high",
        "AFG": "very_high", "SYR": "very_high", "YEM": "very_high", "SDN": "very_high"
    }
    
    return corruption_levels.get(country_code, "moderate")  # Default to moderate


def get_country_geography_advantages(country_code: str) -> List[str]:
    """
    Get geographic advantages for a country.
    
    Returns list of applicable geography constants.
    """
    # Based on geographic characteristics and strategic importance
    geography_map = {
        "USA": ["coastal_access", "resource_rich", "strategic_location"],
        "GBR": ["island_advantage", "coastal_access", "strategic_location"],
        "JPN": ["island_advantage", "coastal_access", "strategic_location"],
        "AUS": ["island_advantage", "coastal_access", "resource_rich"],
        "NZL": ["island_advantage", "coastal_access"],
        "ITA": ["coastal_access", "strategic_location"],
        "ESP": ["coastal_access", "strategic_location"],
        "FRA": ["coastal_access", "strategic_location"],
        "DEU": ["strategic_location"],
        "NLD": ["coastal_access", "strategic_location"],
        "BEL": ["coastal_access", "strategic_location"],
        "CHE": ["mountain_barriers", "strategic_location"],
        "AUT": ["mountain_barriers"],
        "NOR": ["arctic_challenges", "coastal_access", "resource_rich"],
        "SWE": ["arctic_challenges", "coastal_access"],
        "FIN": ["arctic_challenges"],
        "DNK": ["coastal_access", "strategic_location"],
        "CAN": ["arctic_challenges", "coastal_access", "resource_rich"],
        "CHN": ["coastal_access", "strategic_location", "resource_rich"],
        "IND": ["coastal_access", "strategic_location"],
        "BRA": ["coastal_access", "resource_rich"],
        "RUS": ["arctic_challenges", "coastal_access", "resource_rich"],
        "KOR": ["coastal_access", "strategic_location"],
        "MEX": ["coastal_access", "resource_rich"],
        "IDN": ["island_advantage", "coastal_access", "resource_rich"],
        "SAU": ["desert_limitations", "resource_rich"],
        "ARE": ["desert_limitations", "coastal_access", "resource_rich"],
        "TUR": ["coastal_access", "strategic_location"],
        "EGY": ["desert_limitations", "coastal_access", "strategic_location"],
        "IRN": ["desert_limitations", "coastal_access", "resource_rich"],
        "IRQ": ["desert_limitations", "resource_rich"],
        "AFG": ["mountain_barriers", "desert_limitations"],
        "PAK": ["mountain_barriers", "coastal_access"],
        "BGD": ["coastal_access"],
        "VNM": ["coastal_access", "strategic_location"],
        "THA": ["coastal_access", "strategic_location"],
        "MYS": ["coastal_access", "resource_rich"],
        "SGP": ["island_advantage", "coastal_access", "strategic_location"],
        "PHL": ["island_advantage", "coastal_access"],
        "CHL": ["coastal_access", "resource_rich"],
        "ARG": ["coastal_access", "resource_rich"],
        "ZAF": ["coastal_access", "resource_rich"],
        "NGA": ["coastal_access", "resource_rich"],
        "EGY": ["desert_limitations", "coastal_access", "strategic_location"],
        "ETH": ["landlocked"],
        "KEN": ["coastal_access"],
        "MAR": ["coastal_access"],
        "TUN": ["coastal_access"],
        "DZA": ["coastal_access", "desert_limitations", "resource_rich"],
        "LBY": ["desert_limitations", "coastal_access", "resource_rich"],
        "UKR": ["coastal_access", "strategic_location"],
        "POL": ["coastal_access", "strategic_location"],
        "CZE": ["landlocked"],
        "HUN": ["landlocked"],
        "ROU": ["coastal_access"],
        "BGR": ["coastal_access"],
        "HRV": ["coastal_access"],
        "SRB": ["landlocked"],
        "BIH": ["landlocked"],
        "MKD": ["landlocked"],
        "ALB": ["coastal_access"],
        "MNE": ["coastal_access"],
        "GRC": ["island_advantage", "coastal_access", "strategic_location"],
        "CYP": ["island_advantage", "coastal_access", "strategic_location"],
        "MLT": ["island_advantage", "coastal_access", "strategic_location"],
        "ISL": ["island_advantage", "arctic_challenges", "resource_rich"],
        "LUX": ["landlocked"],
        "LIE": ["landlocked"],
        "AND": ["mountain_barriers"],
        "MCO": ["coastal_access"],
        "SMR": ["landlocked"],
        "VAT": ["landlocked"]
    }
    
    return geography_map.get(country_code, ["strategic_location"])  # Default to strategic location


def calculate_corruption_index(country_code: str) -> float:
    """Calculate corruption index value (0-1) for a country."""
    corruption_level = get_country_corruption_level(country_code)
    return CORRUPTION_CONSTANTS[corruption_level]["score"]


def calculate_geography_advantage(country_code: str) -> float:
    """Calculate geography advantage multiplier for a country."""
    geography_advantages = get_country_geography_advantages(country_code)
    
    # Combine multiple geography advantages (multiplicative)
    total_multiplier = 1.0
    for advantage in geography_advantages:
        total_multiplier *= GEOGRAPHY_CONSTANTS[advantage]["multiplier"]
    
    # Cap the multiplier to reasonable bounds
    return float(np.clip(total_multiplier, 0.5, 1.5))


def compute_company_dominance_score(factor_values: np.ndarray) -> float:
    """
    Compute composite market dominance score from company factor values.
    
    Uses weighted average with equal weights for all factors.
    """
    weights = np.array([0.3, 0.2, 0.2, 0.3])  # market_cap, rd_spending, employment, productivity
    
    # Normalize factors to 0-100 scale
    normalized = np.zeros_like(factor_values)
    
    for i, (factor_name, factor_def) in enumerate(COMPANY_FACTORS.items()):
        min_val = factor_def.min_value or 0.0
        max_val = factor_def.max_value or 100.0
        
        if factor_def.transform == "log":
            # Apply log transformation first
            log_val = np.log(max(factor_values[i], min_val))
            log_min = np.log(min_val)
            log_max = np.log(max_val)
            normalized[i] = min(100.0, max(0.0, 
                (log_val - log_min) * 100.0 / (log_max - log_min)
            ))
        else:
            # Linear normalization
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
