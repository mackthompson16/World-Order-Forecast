#!/usr/bin/env python3
"""
Demo script showing the dual Empire vs Company analysis approach.

This script demonstrates how to load and analyze both country (empire) data
and company/industry data using the updated framework.
"""

import pandas as pd
from pathlib import Path
import logging

from src.data_ingest.merge_panel import (
    load_all_data, 
    load_all_company_data_merged,
    get_panel_summary,
    get_company_panel_summary,
    filter_panel_data,
    filter_company_panel_data
)
from src.data_schema import (
    compute_composite_standing,
    compute_company_dominance_score,
    get_factor_values,
    get_company_factor_values,
    CountryData,
    CompanyData
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run the dual analysis demo."""
    data_dir = Path("data")
    
    print("=" * 60)
    print("EMPIRE vs COMPANY ANALYSIS DEMO")
    print("=" * 60)
    
    # Load country (empire) data
    print("\n1. LOADING EMPIRE (COUNTRY) DATA")
    print("-" * 40)
    
    # Focus on democratic countries with good data transparency
    democratic_countries = [
        "USA", "GBR", "DEU", "FRA", "CAN", "AUS", 
        "NLD", "CHE", "DNK", "SWE", "NOR", "FIN"
    ]
    
    country_df = load_all_data(
        data_dir=data_dir,
        countries=democratic_countries,
        years=list(range(2000, 2024))
    )
    
    # Filter for quality
    country_df = filter_panel_data(country_df)
    
    # Get summary
    country_summary = get_panel_summary(country_df)
    print(f"Country panel: {country_summary['n_observations']} observations")
    print(f"Countries: {country_summary['n_countries']} ({', '.join(country_summary['countries'][:5])}...)")
    print(f"Years: {country_summary['year_range'][0]}-{country_summary['year_range'][1]}")
    
    # Load company/industry data
    print("\n2. LOADING COMPANY (INDUSTRY) DATA")
    print("-" * 40)
    
    company_df = load_all_company_data_merged(
        data_dir=data_dir,
        years=list(range(2000, 2024))
    )
    
    # Filter for quality
    company_df = filter_company_panel_data(company_df)
    
    # Get summary
    company_summary = get_company_panel_summary(company_df)
    print(f"Company panel: {company_summary['n_observations']} observations")
    print(f"Industries: {company_summary['n_industries']} ({', '.join(company_summary['industries'][:5])}...)")
    print(f"Years: {company_summary['year_range'][0]}-{company_summary['year_range'][1]}")
    
    # Compute composite scores
    print("\n3. COMPUTING COMPOSITE SCORES")
    print("-" * 40)
    
    # Country standing scores
    country_scores = []
    for _, row in country_df.iterrows():
        country_data = CountryData(
            country=row['country'],
            year=row['year'],
            education=row.get('education'),
            innovation=row.get('innovation'),
            competitiveness=row.get('competitiveness'),
            military=row.get('military'),
            trade_share=row.get('trade_share'),
            reserve_currency_proxy=row.get('reserve_currency_proxy'),
            financial_center_proxy=row.get('financial_center_proxy'),
            debt=row.get('debt')
        )
        
        factor_values = get_factor_values(country_data)
        standing_score = compute_composite_standing(factor_values)
        
        country_scores.append({
            'country': row['country'],
            'year': row['year'],
            'standing_score': standing_score
        })
    
    country_scores_df = pd.DataFrame(country_scores)
    
    # Company dominance scores
    company_scores = []
    for _, row in company_df.iterrows():
        company_data = CompanyData(
            industry=row['industry'],
            year=row['year'],
            market_cap_billions=row.get('market_cap_billions'),
            rd_spending_percent=row.get('rd_spending_percent'),
            employment_share_percent=row.get('employment_share_percent'),
            productivity_index=row.get('productivity_index')
        )
        
        factor_values = get_company_factor_values(company_data)
        dominance_score = compute_company_dominance_score(factor_values)
        
        company_scores.append({
            'industry': row['industry'],
            'year': row['year'],
            'dominance_score': dominance_score
        })
    
    company_scores_df = pd.DataFrame(company_scores)
    
    # Show top performers
    print("\n4. TOP PERFORMERS (2023)")
    print("-" * 40)
    
    # Top countries in 2023
    latest_year = country_scores_df['year'].max()
    top_countries = (country_scores_df[country_scores_df['year'] == latest_year]
                    .nlargest(5, 'standing_score'))
    
    print("Top Countries (Empire Standing):")
    for _, row in top_countries.iterrows():
        print(f"  {row['country']}: {row['standing_score']:.1f}")
    
    # Top industries in 2023
    latest_year_company = company_scores_df['year'].max()
    top_industries = (company_scores_df[company_scores_df['year'] == latest_year_company]
                     .nlargest(5, 'dominance_score'))
    
    print("\nTop Industries (Market Dominance):")
    for _, row in top_industries.iterrows():
        print(f"  {row['industry']}: {row['dominance_score']:.1f}")
    
    # Show validation strategy
    print("\n5. VALIDATION STRATEGY")
    print("-" * 40)
    print("Leave-One-Out Validation:")
    print(f"  Country to exclude: Switzerland (CHE)")
    print(f"  Industry to exclude: Utilities")
    print(f"  Reason: Both are stable and predictable, providing good validation baselines")
    
    # Show data quality considerations
    print("\n6. DATA QUALITY FOCUS")
    print("-" * 40)
    print("Democratic Countries with Data Transparency:")
    print(f"  Included: {len(democratic_countries)} countries with strong institutions")
    print(f"  Excluded: Countries with data reliability concerns (CHN, RUS, etc.)")
    print(f"  Future: Gradual expansion with uncertainty quantification")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    # Save results
    country_scores_df.to_csv("results/empire_standing_scores.csv", index=False)
    company_scores_df.to_csv("results/company_dominance_scores.csv", index=False)
    
    print(f"\nResults saved to:")
    print(f"  - results/empire_standing_scores.csv")
    print(f"  - results/company_dominance_scores.csv")


if __name__ == "__main__":
    main()
