"""
SIPRI military expenditure data loader.

TODO: Implement actual SIPRI data integration.
Currently provides synthetic data generation for development.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_sipri_data(
    data_dir: Path,
    countries: Optional[List[str]] = None,
    years: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Load SIPRI military expenditure data.
    
    Args:
        data_dir: Directory containing SIPRI data files
        countries: List of country codes to load (None = all)
        years: List of years to load (None = all)
        
    Returns:
        DataFrame with columns: country, year, military
    """
    # TODO: Implement actual SIPRI data loading
    logger.warning("SIPRI data loader not implemented - using synthetic data")
    
    if countries is None:
        countries = ["USA", "CHN", "DEU", "JPN", "GBR", "FRA", "IND", "BRA", "CAN", "AUS"]
    
    if years is None:
        years = list(range(2000, 2024))
    
    data = []
    for country in countries:
        for year in years:
            # Generate synthetic military expenditure data
            base_military = np.random.normal(2.0, 1.0)  # Military expenditure as % of GDP
            
            # Country-specific effects
            if country == "USA":
                base_military += 1.0
            elif country == "CHN":
                base_military += 0.5
            elif country == "RUS":
                base_military += 1.2
            
            # Add some volatility
            volatility = np.random.normal(0, 0.3)
            
            data.append({
                "country": country,
                "year": year,
                "military": max(0, base_military + volatility)
            })
    
    return pd.DataFrame(data)
