"""
Panel data merging and alignment utilities.

Combines data from multiple sources into a unified annual panel format.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from .load_wdi import load_wdi_education, load_wdi_trade, load_wdi_debt
from .load_wipo import load_wipo_data
from .load_sipri import load_sipri_data
from .load_imf_cofer import load_imf_cofer_data
from ..data_schema import FACTOR_NAMES, CountryData, get_factor_values, get_mask_values

logger = logging.getLogger(__name__)


def load_all_data(
    data_dir: Path,
    countries: Optional[List[str]] = None,
    years: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Load and merge all data sources into a unified panel.
    
    Args:
        data_dir: Directory containing data files
        countries: List of country codes to load (None = all)
        years: List of years to load (None = all)
        
    Returns:
        DataFrame with columns: country, year, f1..f8, mask_* flags
    """
    logger.info("Loading data from all sources...")
    
    # Load data from each source
    wdi_education = load_wdi_education(data_dir)
    wdi_trade = load_wdi_trade(data_dir)
    wdi_debt = load_wdi_debt(data_dir)
    wipo_data = load_wipo_data(data_dir, countries, years)
    sipri_data = load_sipri_data(data_dir, countries, years)
    imf_data = load_imf_cofer_data(data_dir, countries, years)
    
    # Create base panel with all country-year combinations
    if countries is None:
        countries = list(set(
            list(wdi_education['country'].unique()) +
            list(wipo_data['country'].unique()) +
            list(sipri_data['country'].unique()) +
            list(imf_data['country'].unique())
        ))
    
    if years is None:
        years = list(set(
            list(wdi_education['year'].unique()) +
            list(wipo_data['year'].unique()) +
            list(sipri_data['year'].unique()) +
            list(imf_data['year'].unique())
        ))
    
    # Create base panel
    panel_data = []
    for country in countries:
        for year in years:
            panel_data.append({
                'country': country,
                'year': year
            })
    
    panel_df = pd.DataFrame(panel_data)
    
    # Merge each data source
    panel_df = panel_df.merge(
        wdi_education[['country', 'year', 'education']], 
        on=['country', 'year'], 
        how='left'
    )
    panel_df = panel_df.merge(
        wdi_trade[['country', 'year', 'trade_share']], 
        on=['country', 'year'], 
        how='left'
    )
    panel_df = panel_df.merge(
        wdi_debt[['country', 'year', 'debt']], 
        on=['country', 'year'], 
        how='left'
    )
    panel_df = panel_df.merge(
        wipo_data[['country', 'year', 'innovation']], 
        on=['country', 'year'], 
        how='left'
    )
    panel_df = panel_df.merge(
        sipri_data[['country', 'year', 'military']], 
        on=['country', 'year'], 
        how='left'
    )
    panel_df = panel_df.merge(
        imf_data[['country', 'year', 'reserve_currency_proxy']], 
        on=['country', 'year'], 
        how='left'
    )
    
    # Add synthetic competitiveness and financial center data
    panel_df = _add_synthetic_competitiveness(panel_df)
    panel_df = _add_synthetic_financial_center(panel_df)
    
    # Create mask flags for missing data
    panel_df = _create_mask_flags(panel_df)
    
    logger.info(f"Created panel with {len(panel_df)} observations")
    logger.info(f"Countries: {len(panel_df['country'].unique())}")
    logger.info(f"Years: {panel_df['year'].min()}-{panel_df['year'].max()}")
    
    return panel_df


def _add_synthetic_competitiveness(df: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic competitiveness data."""
    competitiveness = []
    
    for _, row in df.iterrows():
        # Base competitiveness on other factors
        base_score = 50.0
        
        if pd.notna(row.get('education')):
            base_score += (row['education'] - 12.0) * 2.0
        if pd.notna(row.get('innovation')):
            base_score += np.log(row['innovation']) * 5.0
        if pd.notna(row.get('trade_share')):
            base_score += (row['trade_share'] - 60.0) * 0.2
        
        # Add country-specific effects
        if row['country'] == 'USA':
            base_score += 10.0
        elif row['country'] == 'CHN':
            base_score += 5.0
        elif row['country'] == 'DEU':
            base_score += 8.0
        
        # Add noise
        score = base_score + np.random.normal(0, 5.0)
        competitiveness.append(max(0, min(100, score)))
    
    df['competitiveness'] = competitiveness
    return df


def _add_synthetic_financial_center(df: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic financial center data."""
    financial_center = []
    
    for _, row in df.iterrows():
        # Base financial center score
        base_score = 30.0
        
        # Correlate with reserve currency status
        if pd.notna(row.get('reserve_currency_proxy')):
            base_score += row['reserve_currency_proxy'] * 0.5
        
        # Country-specific effects
        if row['country'] == 'USA':
            base_score += 30.0
        elif row['country'] == 'GBR':
            base_score += 25.0
        elif row['country'] == 'JPN':
            base_score += 15.0
        elif row['country'] == 'DEU':
            base_score += 10.0
        
        # Add noise
        score = base_score + np.random.normal(0, 3.0)
        financial_center.append(max(0, min(100, score)))
    
    df['financial_center_proxy'] = financial_center
    return df


def _create_mask_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create mask flags for missing data."""
    for factor in FACTOR_NAMES:
        df[f'mask_{factor}'] = df[factor].isna()
    
    return df


def filter_panel_data(
    df: pd.DataFrame,
    min_years_per_country: int = 15,
    min_countries_per_year: int = 5,
    max_missing_factor_pct: float = 0.3
) -> pd.DataFrame:
    """
    Filter panel data to ensure sufficient coverage.
    
    Args:
        df: Panel DataFrame
        min_years_per_country: Minimum years of data per country
        min_countries_per_year: Minimum countries per year
        max_missing_factor_pct: Maximum % of missing data per factor
        
    Returns:
        Filtered DataFrame
    """
    logger.info("Filtering panel data...")
    
    # Filter countries with sufficient years
    country_years = df.groupby('country')['year'].count()
    valid_countries = country_years[country_years >= min_years_per_country].index
    df = df[df['country'].isin(valid_countries)]
    
    # Filter years with sufficient countries
    year_countries = df.groupby('year')['country'].nunique()
    valid_years = year_countries[year_countries >= min_countries_per_year].index
    df = df[df['year'].isin(valid_years)]
    
    # Check missing data by factor
    for factor in FACTOR_NAMES:
        missing_pct = df[factor].isna().mean()
        if missing_pct > max_missing_factor_pct:
            logger.warning(f"Factor {factor} has {missing_pct:.1%} missing data")
    
    logger.info(f"Filtered panel: {len(df)} observations")
    logger.info(f"Countries: {len(df['country'].unique())}")
    logger.info(f"Years: {df['year'].min()}-{df['year'].max()}")
    
    return df


def create_country_data_objects(df: pd.DataFrame) -> List[CountryData]:
    """Convert DataFrame to list of CountryData objects."""
    country_data = []
    
    for _, row in df.iterrows():
        data = CountryData(
            country=row['country'],
            year=row['year'],
            education=row.get('education'),
            innovation=row.get('innovation'),
            competitiveness=row.get('competitiveness'),
            military=row.get('military'),
            trade_share=row.get('trade_share'),
            reserve_currency_proxy=row.get('reserve_currency_proxy'),
            financial_center_proxy=row.get('financial_center_proxy'),
            debt=row.get('debt'),
            mask_education=row.get('mask_education', False),
            mask_innovation=row.get('mask_innovation', False),
            mask_competitiveness=row.get('mask_competitiveness', False),
            mask_military=row.get('mask_military', False),
            mask_trade_share=row.get('mask_trade_share', False),
            mask_reserve_currency_proxy=row.get('mask_reserve_currency_proxy', False),
            mask_financial_center_proxy=row.get('mask_financial_center_proxy', False),
            mask_debt=row.get('mask_debt', False)
        )
        country_data.append(data)
    
    return country_data


def get_panel_summary(df: pd.DataFrame) -> Dict:
    """Get summary statistics for the panel data."""
    summary = {
        'n_observations': len(df),
        'n_countries': df['country'].nunique(),
        'n_years': df['year'].nunique(),
        'year_range': (df['year'].min(), df['year'].max()),
        'countries': sorted(df['country'].unique().tolist()),
        'missing_data': {}
    }
    
    for factor in FACTOR_NAMES:
        missing_pct = df[factor].isna().mean()
        summary['missing_data'][factor] = {
            'missing_pct': missing_pct,
            'n_missing': df[factor].isna().sum()
        }
    
    return summary
