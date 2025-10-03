#!/usr/bin/env python3
"""
Demo script showing corruption and geography constant effects.

This script demonstrates how corruption (data trustworthiness) and geography
(optimistic curves) constants affect country standing calculations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from src.data_ingest.merge_panel import (
    load_all_data, 
    filter_panel_data,
    create_country_data_objects
)
from src.data_schema import (
    compute_composite_standing,
    get_factor_values,
    calculate_corruption_index,
    calculate_geography_advantage,
    get_country_corruption_level,
    get_country_geography_advantages,
    CORRUPTION_CONSTANTS,
    GEOGRAPHY_CONSTANTS
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Democratic countries with good data transparency
DEMOCRATIC_COUNTRIES = [
    "USA", "GBR", "DEU", "FRA", "CAN", "AUS", 
    "NLD", "CHE", "DNK", "SWE", "JPN", "KOR", 
    "ITA", "ESP", "BEL", "AUT", "NOR", "FIN", 
    "NZL", "IRL"
]


def analyze_corruption_effects():
    """Analyze how corruption affects country standing scores."""
    print("=" * 80)
    print("CORRUPTION CONSTANT ANALYSIS")
    print("=" * 80)
    
    print("Corruption Levels and Trust Scores:")
    for level, data in CORRUPTION_CONSTANTS.items():
        print(f"  {level}: {data['score']:.2f} - {data['description']}")
    
    print("\nSample Countries by Corruption Level:")
    sample_countries = ["DNK", "USA", "ITA", "CHN", "RUS", "VEN"]
    
    for country in sample_countries:
        corruption_level = get_country_corruption_level(country)
        corruption_score = calculate_corruption_index(country)
        print(f"  {country}: {corruption_level} (score: {corruption_score:.2f})")
    
    print("\n" + "=" * 80)


def analyze_geography_effects():
    """Analyze how geography affects country standing scores."""
    print("GEOGRAPHY CONSTANT ANALYSIS")
    print("=" * 80)
    
    print("Geography Advantage Types:")
    for geo_type, data in GEOGRAPHY_CONSTANTS.items():
        print(f"  {geo_type}: {data['multiplier']:.2f}x - {data['description']}")
    
    print("\nSample Countries by Geography Advantages:")
    sample_countries = ["USA", "GBR", "JPN", "AUS", "CHE", "MEX", "BRA"]
    
    for country in sample_countries:
        geo_advantages = get_country_geography_advantages(country)
        geo_multiplier = calculate_geography_advantage(country)
        print(f"  {country}: {geo_advantages} (multiplier: {geo_multiplier:.2f}x)")
    
    print("\n" + "=" * 80)


def compare_scoring_methods(data_dir: Path):
    """Compare scoring with and without corruption/geography effects."""
    print("SCORING METHOD COMPARISON")
    print("=" * 80)
    
    # Load country data
    country_df = load_all_data(
        data_dir=data_dir,
        countries=DEMOCRATIC_COUNTRIES,
        years=list(range(2020, 2024))
    )
    
    country_df = filter_panel_data(country_df)
    
    # Get latest year data for comparison
    latest_year = country_df['year'].max()
    latest_data = country_df[country_df['year'] == latest_year].copy()
    
    results = []
    
    for _, row in latest_data.iterrows():
        country = row['country']
        
        # Create CountryData object
        country_data = CountryData(
            country=country,
            year=row['year'],
            education=row.get('education'),
            innovation=row.get('innovation'),
            competitiveness=row.get('competitiveness'),
            military=row.get('military'),
            trade_share=row.get('trade_share'),
            reserve_currency_proxy=row.get('reserve_currency_proxy'),
            financial_center_proxy=row.get('financial_center_proxy'),
            debt=row.get('debt'),
            corruption_index=row.get('corruption_index'),
            geography_advantage=row.get('geography_advantage')
        )
        
        factor_values = get_factor_values(country_data)
        
        # Calculate scores with different methods
        score_basic = compute_composite_standing(factor_values, apply_corruption_penalty=False, apply_geography_boost=False)
        score_with_corruption = compute_composite_standing(factor_values, apply_corruption_penalty=True, apply_geography_boost=False)
        score_with_geography = compute_composite_standing(factor_values, apply_corruption_penalty=False, apply_geography_boost=True)
        score_full = compute_composite_standing(factor_values, apply_corruption_penalty=True, apply_geography_boost=True)
        
        results.append({
            'country': country,
            'corruption_level': get_country_corruption_level(country),
            'corruption_score': calculate_corruption_index(country),
            'geography_multiplier': calculate_geography_advantage(country),
            'basic_score': score_basic,
            'with_corruption': score_with_corruption,
            'with_geography': score_with_geography,
            'full_score': score_full,
            'corruption_impact': score_with_corruption - score_basic,
            'geography_impact': score_with_geography - score_basic,
            'combined_impact': score_full - score_basic
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('full_score', ascending=False)
    
    print(f"Country Standing Scores Comparison ({latest_year})")
    print("-" * 80)
    print(f"{'Country':<8} {'Basic':<8} {'+Corrupt':<10} {'+Geo':<8} {'Full':<8} {'Impact':<8}")
    print("-" * 80)
    
    for _, row in results_df.iterrows():
        print(f"{row['country']:<8} {row['basic_score']:<8.1f} {row['with_corruption']:<10.1f} "
              f"{row['with_geography']:<8.1f} {row['full_score']:<8.1f} {row['combined_impact']:<8.1f}")
    
    return results_df


def create_visualization(results_df: pd.DataFrame, save_dir: Path = Path("results")):
    """Create visualization of corruption and geography effects."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Corruption & Geography Constant Effects on Country Standing', fontsize=16)
    
    # Plot 1: Corruption Impact
    ax1 = axes[0, 0]
    scatter = ax1.scatter(
        results_df['corruption_score'],
        results_df['corruption_impact'],
        c=results_df['basic_score'],
        cmap='RdYlBu_r',
        s=100,
        alpha=0.7
    )
    ax1.set_xlabel('Corruption Score (Trustworthiness)')
    ax1.set_ylabel('Corruption Impact on Score')
    ax1.set_title('Corruption Penalty Effect')
    plt.colorbar(scatter, ax=ax1, label='Basic Score')
    ax1.grid(True, alpha=0.3)
    
    # Annotate some countries
    for _, row in results_df.head(5).iterrows():
        ax1.annotate(row['country'], 
                    (row['corruption_score'], row['corruption_impact']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 2: Geography Impact
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(
        results_df['geography_multiplier'],
        results_df['geography_impact'],
        c=results_df['basic_score'],
        cmap='RdYlGn',
        s=100,
        alpha=0.7
    )
    ax2.set_xlabel('Geography Multiplier')
    ax2.set_ylabel('Geography Impact on Score')
    ax2.set_title('Geography Advantage Effect')
    plt.colorbar(scatter2, ax=ax2, label='Basic Score')
    ax2.grid(True, alpha=0.3)
    
    # Annotate some countries
    for _, row in results_df.head(5).iterrows():
        ax2.annotate(row['country'], 
                    (row['geography_multiplier'], row['geography_impact']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 3: Combined Impact
    ax3 = axes[1, 0]
    colors = ['red' if x < 0 else 'green' for x in results_df['combined_impact']]
    bars = ax3.barh(results_df['country'], results_df['combined_impact'], color=colors, alpha=0.7)
    ax3.set_xlabel('Combined Impact (Corruption + Geography)')
    ax3.set_title('Net Effect of Both Constants')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 4: Score Comparison
    ax4 = axes[1, 1]
    x = np.arange(len(results_df))
    width = 0.2
    
    ax4.bar(x - 1.5*width, results_df['basic_score'], width, label='Basic', alpha=0.8)
    ax4.bar(x - 0.5*width, results_df['with_corruption'], width, label='+Corruption', alpha=0.8)
    ax4.bar(x + 0.5*width, results_df['with_geography'], width, label='+Geography', alpha=0.8)
    ax4.bar(x + 1.5*width, results_df['full_score'], width, label='Full', alpha=0.8)
    
    ax4.set_xlabel('Countries')
    ax4.set_ylabel('Standing Score')
    ax4.set_title('Score Comparison by Method')
    ax4.set_xticks(x)
    ax4.set_xticklabels(results_df['country'], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'corruption_geography_effects.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Run the corruption and geography analysis demo."""
    data_dir = Path("data")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("CORRUPTION & GEOGRAPHY CONSTANTS DEMO")
    print("=" * 80)
    print("This demo shows how corruption (data trustworthiness) and geography")
    print("(optimistic curves) constants affect country standing calculations.")
    print()
    
    # Analyze corruption effects
    analyze_corruption_effects()
    
    # Analyze geography effects  
    analyze_geography_effects()
    
    # Compare scoring methods
    results_df = compare_scoring_methods(data_dir)
    
    # Create visualization
    try:
        create_visualization(results_df, results_dir)
        print(f"\nVisualization saved to {results_dir}/corruption_geography_effects.png")
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
    # Save results
    results_df.to_csv(results_dir / 'corruption_geography_analysis.csv', index=False)
    print(f"Detailed results saved to {results_dir}/corruption_geography_analysis.csv")
    
    # Show key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    print(f"1. Corruption Impact Range: {results_df['corruption_impact'].min():.1f} to {results_df['corruption_impact'].max():.1f}")
    print(f"2. Geography Impact Range: {results_df['geography_impact'].min():.1f} to {results_df['geography_impact'].max():.1f}")
    print(f"3. Combined Impact Range: {results_df['combined_impact'].min():.1f} to {results_df['combined_impact'].max():.1f}")
    
    print("\nTop 3 Countries Benefiting Most from Geography:")
    top_geo = results_df.nlargest(3, 'geography_impact')[['country', 'geography_impact', 'geography_multiplier']]
    for _, row in top_geo.iterrows():
        print(f"  {row['country']}: +{row['geography_impact']:.1f} points ({row['geography_multiplier']:.2f}x multiplier)")
    
    print("\nTop 3 Countries Most Penalized by Corruption:")
    top_corrupt = results_df.nsmallest(3, 'corruption_impact')[['country', 'corruption_impact', 'corruption_score']]
    for _, row in top_corrupt.iterrows():
        print(f"  {row['country']}: {row['corruption_impact']:.1f} points (trust score: {row['corruption_score']:.2f})")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
