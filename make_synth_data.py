"""
Enhanced synthetic data generator for country standing forecast.

Generates plausible synthetic panel data that mimics real-world patterns
in macro factors for multiple countries over time.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

from src.data_schema import CountryData, FACTOR_NAMES, FACTORS, compute_composite_standing

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generator for synthetic country standing data."""
    
    def __init__(
        self,
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        seed: int = 42
    ):
        """
        Initialize synthetic data generator.
        
        Args:
            countries: List of country codes (None = default set)
            years: List of years (None = 2000-2023)
            seed: Random seed for reproducibility
        """
        self.countries = countries or [
            "USA", "CHN", "DEU", "JPN", "GBR", "FRA", "IND", "BRA", 
            "CAN", "AUS", "RUS", "KOR", "ITA", "ESP", "MEX", "IDN"
        ]
        self.years = years or list(range(2000, 2024))
        self.seed = seed
        
        # Set random seed
        np.random.seed(seed)
        
        # Country-specific characteristics
        self.country_profiles = self._create_country_profiles()
        
        logger.info(f"Initialized synthetic data generator for {len(self.countries)} countries, {len(self.years)} years")
    
    def _create_country_profiles(self) -> Dict[str, Dict]:
        """Create country-specific profiles with realistic characteristics."""
        profiles = {}
        
        # Define country characteristics
        country_chars = {
            "USA": {
                "development_level": "high",
                "innovation_leader": True,
                "financial_center": True,
                "reserve_currency": True,
                "military_power": True,
                "trade_openness": "moderate"
            },
            "CHN": {
                "development_level": "emerging",
                "innovation_leader": True,
                "financial_center": False,
                "reserve_currency": False,
                "military_power": True,
                "trade_openness": "high"
            },
            "DEU": {
                "development_level": "high",
                "innovation_leader": True,
                "financial_center": False,
                "reserve_currency": False,
                "military_power": False,
                "trade_openness": "high"
            },
            "JPN": {
                "development_level": "high",
                "innovation_leader": True,
                "financial_center": True,
                "reserve_currency": False,
                "military_power": False,
                "trade_openness": "moderate"
            },
            "GBR": {
                "development_level": "high",
                "innovation_leader": True,
                "financial_center": True,
                "reserve_currency": False,
                "military_power": True,
                "trade_openness": "high"
            },
            "FRA": {
                "development_level": "high",
                "innovation_leader": True,
                "financial_center": False,
                "reserve_currency": False,
                "military_power": True,
                "trade_openness": "moderate"
            },
            "IND": {
                "development_level": "emerging",
                "innovation_leader": False,
                "financial_center": False,
                "reserve_currency": False,
                "military_power": False,
                "trade_openness": "moderate"
            },
            "BRA": {
                "development_level": "emerging",
                "innovation_leader": False,
                "financial_center": False,
                "reserve_currency": False,
                "military_power": False,
                "trade_openness": "moderate"
            },
            "CAN": {
                "development_level": "high",
                "innovation_leader": True,
                "financial_center": False,
                "reserve_currency": False,
                "military_power": False,
                "trade_openness": "high"
            },
            "AUS": {
                "development_level": "high",
                "innovation_leader": True,
                "financial_center": False,
                "reserve_currency": False,
                "military_power": False,
                "trade_openness": "moderate"
            },
            "RUS": {
                "development_level": "emerging",
                "innovation_leader": False,
                "financial_center": False,
                "reserve_currency": False,
                "military_power": True,
                "trade_openness": "low"
            },
            "KOR": {
                "development_level": "high",
                "innovation_leader": True,
                "financial_center": False,
                "reserve_currency": False,
                "military_power": False,
                "trade_openness": "high"
            },
            "ITA": {
                "development_level": "high",
                "innovation_leader": True,
                "financial_center": False,
                "reserve_currency": False,
                "military_power": False,
                "trade_openness": "moderate"
            },
            "ESP": {
                "development_level": "high",
                "innovation_leader": True,
                "financial_center": False,
                "reserve_currency": False,
                "military_power": False,
                "trade_openness": "moderate"
            },
            "MEX": {
                "development_level": "emerging",
                "innovation_leader": False,
                "financial_center": False,
                "reserve_currency": False,
                "military_power": False,
                "trade_openness": "high"
            },
            "IDN": {
                "development_level": "emerging",
                "innovation_leader": False,
                "financial_center": False,
                "reserve_currency": False,
                "military_power": False,
                "trade_openness": "moderate"
            }
        }
        
        # Create profiles with factor-specific parameters
        for country, chars in country_chars.items():
            profile = {}
            
            # Education (years of schooling)
            if chars["development_level"] == "high":
                profile["education"] = {
                    "base": 13.0,
                    "trend": 0.05,
                    "volatility": 0.3,
                    "min": 10.0,
                    "max": 16.0
                }
            elif chars["development_level"] == "emerging":
                profile["education"] = {
                    "base": 8.0,
                    "trend": 0.15,
                    "volatility": 0.5,
                    "min": 5.0,
                    "max": 12.0
                }
            else:
                profile["education"] = {
                    "base": 6.0,
                    "trend": 0.2,
                    "volatility": 0.7,
                    "min": 3.0,
                    "max": 10.0
                }
            
            # Innovation (patents per million)
            if chars["innovation_leader"]:
                profile["innovation"] = {
                    "base": 100.0,
                    "trend": 0.08,
                    "volatility": 0.4,
                    "min": 20.0,
                    "max": 1000.0
                }
            else:
                profile["innovation"] = {
                    "base": 10.0,
                    "trend": 0.12,
                    "volatility": 0.6,
                    "min": 1.0,
                    "max": 200.0
                }
            
            # Competitiveness (index 0-100)
            if chars["development_level"] == "high":
                profile["competitiveness"] = {
                    "base": 75.0,
                    "trend": 0.02,
                    "volatility": 2.0,
                    "min": 60.0,
                    "max": 90.0
                }
            elif chars["development_level"] == "emerging":
                profile["competitiveness"] = {
                    "base": 50.0,
                    "trend": 0.05,
                    "volatility": 3.0,
                    "min": 30.0,
                    "max": 70.0
                }
            else:
                profile["competitiveness"] = {
                    "base": 35.0,
                    "trend": 0.08,
                    "volatility": 4.0,
                    "min": 20.0,
                    "max": 55.0
                }
            
            # Military expenditure (% of GDP)
            if chars["military_power"]:
                profile["military"] = {
                    "base": 3.5,
                    "trend": 0.01,
                    "volatility": 0.3,
                    "min": 2.0,
                    "max": 6.0
                }
            else:
                profile["military"] = {
                    "base": 1.5,
                    "trend": 0.005,
                    "volatility": 0.2,
                    "min": 0.5,
                    "max": 3.0
                }
            
            # Trade share (% of GDP)
            if chars["trade_openness"] == "high":
                profile["trade_share"] = {
                    "base": 80.0,
                    "trend": 0.01,
                    "volatility": 8.0,
                    "min": 50.0,
                    "max": 120.0
                }
            elif chars["trade_openness"] == "moderate":
                profile["trade_share"] = {
                    "base": 60.0,
                    "trend": 0.02,
                    "volatility": 6.0,
                    "min": 40.0,
                    "max": 90.0
                }
            else:  # low
                profile["trade_share"] = {
                    "base": 40.0,
                    "trend": 0.03,
                    "volatility": 5.0,
                    "min": 25.0,
                    "max": 65.0
                }
            
            # Reserve currency proxy (% of global reserves)
            if chars["reserve_currency"]:
                profile["reserve_currency_proxy"] = {
                    "base": 60.0,
                    "trend": -0.005,
                    "volatility": 3.0,
                    "min": 45.0,
                    "max": 70.0
                }
            elif chars["financial_center"]:
                profile["reserve_currency_proxy"] = {
                    "base": 5.0,
                    "trend": 0.01,
                    "volatility": 1.0,
                    "min": 2.0,
                    "max": 10.0
                }
            else:
                profile["reserve_currency_proxy"] = {
                    "base": 0.5,
                    "trend": 0.005,
                    "volatility": 0.3,
                    "min": 0.1,
                    "max": 2.0
                }
            
            # Financial center proxy (index 0-100)
            if chars["financial_center"]:
                profile["financial_center_proxy"] = {
                    "base": 80.0,
                    "trend": 0.01,
                    "volatility": 3.0,
                    "min": 65.0,
                    "max": 95.0
                }
            else:
                profile["financial_center_proxy"] = {
                    "base": 30.0,
                    "trend": 0.02,
                    "volatility": 5.0,
                    "min": 15.0,
                    "max": 60.0
                }
            
            # Debt (% of GDP) - inverted (lower is better)
            if chars["development_level"] == "high":
                profile["debt"] = {
                    "base": 60.0,
                    "trend": 0.5,
                    "volatility": 5.0,
                    "min": 30.0,
                    "max": 120.0
                }
            elif chars["development_level"] == "emerging":
                profile["debt"] = {
                    "base": 45.0,
                    "trend": 0.3,
                    "volatility": 4.0,
                    "min": 20.0,
                    "max": 80.0
                }
            else:
                profile["debt"] = {
                    "base": 35.0,
                    "trend": 0.2,
                    "volatility": 3.0,
                    "min": 15.0,
                    "max": 60.0
                }
            
            profiles[country] = profile
        
        return profiles
    
    def generate_factor_data(
        self,
        country: str,
        factor: str,
        years: List[int]
    ) -> np.ndarray:
        """
        Generate synthetic data for a specific factor and country.
        
        Args:
            country: Country code
            factor: Factor name
            years: List of years
            
        Returns:
            Array of factor values
        """
        if country not in self.country_profiles:
            raise ValueError(f"Unknown country: {country}")
        
        if factor not in self.country_profiles[country]:
            raise ValueError(f"Unknown factor: {factor}")
        
        profile = self.country_profiles[country][factor]
        n_years = len(years)
        
        # Generate base time series with trend
        base_values = np.zeros(n_years)
        for i, year in enumerate(years):
            time_factor = year - years[0]
            base_values[i] = profile["base"] + profile["trend"] * time_factor
        
        # Add volatility
        volatility = np.random.normal(0, profile["volatility"], n_years)
        
        # Add some autocorrelation
        if n_years > 1:
            for i in range(1, n_years):
                volatility[i] = 0.7 * volatility[i-1] + 0.3 * volatility[i]
        
        # Combine base trend with volatility
        values = base_values + volatility
        
        # Apply bounds
        values = np.clip(values, profile["min"], profile["max"])
        
        # Add some missing data (5-15% missing)
        missing_rate = np.random.uniform(0.05, 0.15)
        n_missing = int(n_years * missing_rate)
        missing_indices = np.random.choice(n_years, n_missing, replace=False)
        values[missing_indices] = np.nan
        
        return values
    
    def generate_country_data(self, country: str) -> List[CountryData]:
        """
        Generate synthetic data for a single country.
        
        Args:
            country: Country code
            
        Returns:
            List of CountryData objects
        """
        country_data = []
        
        for year in self.years:
            # Generate factor values
            factor_values = {}
            mask_values = {}
            
            for factor in FACTOR_NAMES:
                values = self.generate_factor_data(country, factor, [year])
                factor_values[factor] = values[0] if not np.isnan(values[0]) else None
                mask_values[f"mask_{factor}"] = np.isnan(values[0])
            
            # Create CountryData object
            data = CountryData(
                country=country,
                year=year,
                **factor_values,
                **mask_values
            )
            
            country_data.append(data)
        
        return country_data
    
    def generate_panel_data(self) -> List[CountryData]:
        """
        Generate synthetic panel data for all countries.
        
        Returns:
            List of CountryData objects
        """
        logger.info("Generating synthetic panel data...")
        
        all_data = []
        for country in self.countries:
            country_data = self.generate_country_data(country)
            all_data.extend(country_data)
        
        logger.info(f"Generated {len(all_data)} observations for {len(self.countries)} countries")
        
        return all_data
    
    def save_to_csv(self, data: List[CountryData], output_path: Path):
        """
        Save synthetic data to CSV file.
        
        Args:
            data: List of CountryData objects
            output_path: Path to save CSV file
        """
        # Convert to DataFrame
        rows = []
        for item in data:
            row = {
                "country": item.country,
                "year": item.year,
                **{factor: getattr(item, factor) for factor in FACTOR_NAMES},
                **{f"mask_{factor}": getattr(item, f"mask_{factor}") for factor in FACTOR_NAMES}
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save to CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Synthetic data saved to {output_path}")
        
        # Print summary
        print(f"\nSynthetic Data Summary:")
        print(f"Countries: {df['country'].nunique()}")
        print(f"Years: {df['year'].min()}-{df['year'].max()}")
        print(f"Total observations: {len(df)}")
        print(f"\nMissing data by factor:")
        for factor in FACTOR_NAMES:
            missing_pct = df[factor].isna().mean() * 100
            print(f"  {factor}: {missing_pct:.1f}%")
    
    def add_crisis_events(self, data: List[CountryData]) -> List[CountryData]:
        """
        Add realistic crisis events to the synthetic data.
        
        Args:
            data: List of CountryData objects
            
        Returns:
            Modified data with crisis events
        """
        logger.info("Adding crisis events to synthetic data...")
        
        # Define crisis events
        crises = {
            2008: {"name": "Global Financial Crisis", "severity": 0.8, "duration": 3},
            2011: {"name": "European Debt Crisis", "severity": 0.6, "duration": 2},
            2020: {"name": "COVID-19 Pandemic", "severity": 0.9, "duration": 2}
        }
        
        # Apply crisis effects
        for item in data:
            year = item.year
            
            if year in crises:
                crisis = crises[year]
                severity = crisis["severity"]
                
                # Crisis effects on different factors
                if item.education is not None:
                    item.education *= (1 - 0.1 * severity)  # Education slightly affected
                
                if item.innovation is not None:
                    item.innovation *= (1 - 0.2 * severity)  # Innovation more affected
                
                if item.competitiveness is not None:
                    item.competitiveness *= (1 - 0.3 * severity)  # Competitiveness significantly affected
                
                if item.military is not None:
                    item.military *= (1 + 0.1 * severity)  # Military spending may increase
                
                if item.trade_share is not None:
                    item.trade_share *= (1 - 0.4 * severity)  # Trade significantly affected
                
                if item.financial_center_proxy is not None:
                    item.financial_center_proxy *= (1 - 0.5 * severity)  # Financial centers heavily affected
                
                if item.debt is not None:
                    item.debt *= (1 + 0.3 * severity)  # Debt increases during crises
        
        return data


def main():
    """Main function to generate synthetic data."""
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Generate synthetic data
    generator = SyntheticDataGenerator()
    panel_data = generator.generate_panel_data()
    
    # Add crisis events
    panel_data = generator.add_crisis_events(panel_data)
    
    # Save to CSV
    output_path = data_dir / "synthetic_country_data.csv"
    generator.save_to_csv(panel_data, output_path)
    
    # Generate additional summary statistics
    df = pd.DataFrame([{
        "country": item.country,
        "year": item.year,
        **{factor: getattr(item, factor) for factor in FACTOR_NAMES}
    } for item in panel_data])
    
    # Save summary statistics
    summary_path = data_dir / "synthetic_data_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Synthetic Country Data Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Countries: {df['country'].nunique()}\n")
        f.write(f"Years: {df['year'].min()}-{df['year'].max()}\n")
        f.write(f"Total observations: {len(df)}\n\n")
        
        f.write("Factor Statistics:\n")
        f.write("-" * 20 + "\n")
        for factor in FACTOR_NAMES:
            factor_data = df[factor].dropna()
            f.write(f"{factor}:\n")
            f.write(f"  Mean: {factor_data.mean():.2f}\n")
            f.write(f"  Std: {factor_data.std():.2f}\n")
            f.write(f"  Min: {factor_data.min():.2f}\n")
            f.write(f"  Max: {factor_data.max():.2f}\n")
            f.write(f"  Missing: {df[factor].isna().sum()} ({df[factor].isna().mean()*100:.1f}%)\n\n")
    
    print(f"\nSynthetic data generation completed!")
    print(f"Data saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
