"""
Company/Industry data loaders for market analysis.

TODO: Implement actual data source integrations.
Currently provides synthetic data generation for development.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Industry sectors based on GICS classification
INDUSTRY_SECTORS = [
    "Technology", "Healthcare", "Financials", "Consumer_Discretionary",
    "Communication_Services", "Industrials", "Consumer_Staples", 
    "Energy", "Utilities", "Real_Estate", "Materials"
]


def load_market_cap_data(
    data_dir: Path,
    industries: Optional[List[str]] = None,
    years: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Load market capitalization data by industry sector.
    
    Args:
        data_dir: Directory containing market data files
        industries: List of industry sectors to load (None = all)
        years: List of years to load (None = all)
        
    Returns:
        DataFrame with columns: industry, year, market_cap_billions
    """
    # TODO: Implement actual market cap data loading from Yahoo Finance/Bloomberg
    logger.warning("Market cap data loader not implemented - using synthetic data")
    
    if industries is None:
        industries = INDUSTRY_SECTORS
    
    if years is None:
        years = list(range(2000, 2024))
    
    data = []
    for industry in industries:
        for year in years:
            # Generate synthetic market cap data
            if industry == "Technology":
                base_cap = np.random.lognormal(12.0, 0.5)  # Tech has highest valuations
            elif industry == "Healthcare":
                base_cap = np.random.lognormal(11.5, 0.4)
            elif industry == "Financials":
                base_cap = np.random.lognormal(11.0, 0.6)
            elif industry == "Energy":
                base_cap = np.random.lognormal(10.5, 0.8)  # More volatile
            elif industry == "Utilities":
                base_cap = np.random.lognormal(9.5, 0.3)   # Most stable
            else:
                base_cap = np.random.lognormal(10.0, 0.5)
            
            # Add time trend (general market growth)
            growth_factor = np.exp((year - 2000) * 0.06)
            
            data.append({
                "industry": industry,
                "year": year,
                "market_cap_billions": base_cap * growth_factor
            })
    
    return pd.DataFrame(data)


def load_rd_spending_data(
    data_dir: Path,
    industries: Optional[List[str]] = None,
    years: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Load R&D spending data by industry sector.
    
    Args:
        data_dir: Directory containing R&D data files
        industries: List of industry sectors to load (None = all)
        years: List of years to load (None = all)
        
    Returns:
        DataFrame with columns: industry, year, rd_spending_percent
    """
    # TODO: Implement actual R&D data loading from OECD STAN
    logger.warning("R&D spending data loader not implemented - using synthetic data")
    
    if industries is None:
        industries = INDUSTRY_SECTORS
    
    if years is None:
        years = list(range(2000, 2024))
    
    data = []
    for industry in industries:
        for year in years:
            # Generate synthetic R&D spending as % of revenue
            if industry == "Technology":
                base_rd = np.random.normal(15.0, 3.0)  # Tech spends most on R&D
            elif industry == "Healthcare":
                base_rd = np.random.normal(12.0, 2.5)
            elif industry == "Industrials":
                base_rd = np.random.normal(3.5, 1.0)
            elif industry == "Energy":
                base_rd = np.random.normal(2.0, 0.8)
            elif industry == "Utilities":
                base_rd = np.random.normal(0.5, 0.2)   # Lowest R&D spending
            else:
                base_rd = np.random.normal(2.5, 1.0)
            
            # Add slight upward trend
            time_trend = (year - 2000) * 0.1
            
            data.append({
                "industry": industry,
                "year": year,
                "rd_spending_percent": max(0, base_rd + time_trend + np.random.normal(0, 0.5))
            })
    
    return pd.DataFrame(data)


def load_employment_share_data(
    data_dir: Path,
    industries: Optional[List[str]] = None,
    years: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Load employment share data by industry sector.
    
    Args:
        data_dir: Directory containing employment data files
        industries: List of industry sectors to load (None = all)
        years: List of years to load (None = all)
        
    Returns:
        DataFrame with columns: industry, year, employment_share_percent
    """
    # TODO: Implement actual employment data loading from BLS
    logger.warning("Employment share data loader not implemented - using synthetic data")
    
    if industries is None:
        industries = INDUSTRY_SECTORS
    
    if years is None:
        years = list(range(2000, 2024))
    
    data = []
    for industry in industries:
        for year in years:
            # Generate synthetic employment share data
            if industry == "Technology":
                base_employment = np.random.normal(8.0, 1.0)  # Growing sector
                time_trend = (year - 2000) * 0.2  # Strong growth
            elif industry == "Healthcare":
                base_employment = np.random.normal(12.0, 1.5)
                time_trend = (year - 2000) * 0.15
            elif industry == "Industrials":
                base_employment = np.random.normal(15.0, 2.0)
                time_trend = -(year - 2000) * 0.1  # Declining
            elif industry == "Energy":
                base_employment = np.random.normal(3.0, 0.5)
                time_trend = -(year - 2000) * 0.05
            elif industry == "Utilities":
                base_employment = np.random.normal(2.0, 0.3)
                time_trend = -(year - 2000) * 0.02
            else:
                base_employment = np.random.normal(6.0, 1.0)
                time_trend = 0
            
            data.append({
                "industry": industry,
                "year": year,
                "employment_share_percent": max(0.1, base_employment + time_trend + np.random.normal(0, 0.3))
            })
    
    return pd.DataFrame(data)


def load_productivity_data(
    data_dir: Path,
    industries: Optional[List[str]] = None,
    years: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Load productivity index data by industry sector.
    
    Args:
        data_dir: Directory containing productivity data files
        industries: List of industry sectors to load (None = all)
        years: List of years to load (None = all)
        
    Returns:
        DataFrame with columns: industry, year, productivity_index
    """
    # TODO: Implement actual productivity data loading from OECD
    logger.warning("Productivity data loader not implemented - using synthetic data")
    
    if industries is None:
        industries = INDUSTRY_SECTORS
    
    if years is None:
        years = list(range(2000, 2024))
    
    data = []
    for industry in industries:
        base_year = 2000
        for year in years:
            # Generate synthetic productivity index (base year 2000 = 100)
            if industry == "Technology":
                annual_growth = np.random.normal(0.08, 0.02)  # High productivity growth
            elif industry == "Healthcare":
                annual_growth = np.random.normal(0.03, 0.015)
            elif industry == "Industrials":
                annual_growth = np.random.normal(0.025, 0.01)
            elif industry == "Energy":
                annual_growth = np.random.normal(0.015, 0.02)  # Volatile
            elif industry == "Utilities":
                annual_growth = np.random.normal(0.01, 0.005)  # Stable, slow growth
            else:
                annual_growth = np.random.normal(0.02, 0.01)
            
            # Compound growth from base year
            years_elapsed = year - base_year
            productivity_index = 100 * np.exp(annual_growth * years_elapsed)
            
            data.append({
                "industry": industry,
                "year": year,
                "productivity_index": productivity_index
            })
    
    return pd.DataFrame(data)


def load_all_company_data(
    data_dir: Path,
    industries: Optional[List[str]] = None,
    years: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Load and merge all company/industry data sources.
    
    Args:
        data_dir: Directory containing data files
        industries: List of industry sectors to load (None = all)
        years: List of years to load (None = all)
        
    Returns:
        Merged DataFrame with all company metrics
    """
    logger.info("Loading all company data sources...")
    
    # Load individual datasets
    market_cap = load_market_cap_data(data_dir, industries, years)
    rd_spending = load_rd_spending_data(data_dir, industries, years)
    employment = load_employment_share_data(data_dir, industries, years)
    productivity = load_productivity_data(data_dir, industries, years)
    
    # Merge all datasets
    merged = market_cap.merge(rd_spending, on=['industry', 'year'], how='outer')
    merged = merged.merge(employment, on=['industry', 'year'], how='outer')
    merged = merged.merge(productivity, on=['industry', 'year'], how='outer')
    
    logger.info(f"Loaded company data: {len(merged)} records across {len(merged['industry'].unique())} industries")
    
    return merged
