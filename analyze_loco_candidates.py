#!/usr/bin/env python3
"""
Analyze countries and industries to identify optimal LOCO candidates.

This script analyzes the volatility, predictability, and characteristics of
different countries and industries to recommend more representative
leave-one-out validation candidates.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging

from src.data_ingest.merge_panel import (
    load_all_data, 
    load_all_company_data_merged,
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

# Democratic countries with good data transparency
DEMOCRATIC_COUNTRIES = [
    "USA", "GBR", "DEU", "FRA", "CAN", "AUS", 
    "NLD", "CHE", "DNK", "SWE", "JPN", "KOR", 
    "ITA", "ESP", "BEL", "AUT", "NOR", "FIN", 
    "NZL", "IRL"
]


def calculate_volatility_metrics(df: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    """Calculate volatility metrics for each group."""
    metrics = []
    
    for group in df[group_col].unique():
        group_data = df[df[group_col] == group][value_col].dropna()
        
        if len(group_data) < 5:  # Need minimum data points
            continue
            
        # Basic volatility measures
        std_dev = group_data.std()
        cv = std_dev / group_data.mean() if group_data.mean() != 0 else np.nan
        
        # Year-over-year volatility
        yoy_changes = group_data.diff().dropna()
        yoy_volatility = yoy_changes.std()
        yoy_cv = yoy_volatility / abs(group_data.mean()) if group_data.mean() != 0 else np.nan
        
        # Trend consistency (how predictable the trend is)
        if len(group_data) > 3:
            # Calculate rolling correlation with time
            time_series = np.arange(len(group_data))
            trend_correlation = np.corrcoef(time_series, group_data)[0, 1]
        else:
            trend_correlation = np.nan
        
        # Autocorrelation (how much current value depends on past)
        if len(group_data) > 2:
            autocorr = group_data.autocorr(lag=1)
        else:
            autocorr = np.nan
        
        # Range stability (how much the range varies over time)
        rolling_range = group_data.rolling(window=3, min_periods=2).max() - group_data.rolling(window=3, min_periods=2).min()
        range_stability = 1 - (rolling_range.std() / rolling_range.mean()) if rolling_range.mean() != 0 else np.nan
        
        metrics.append({
            group_col: group,
            'mean_value': group_data.mean(),
            'std_dev': std_dev,
            'coefficient_of_variation': cv,
            'yoy_volatility': yoy_volatility,
            'yoy_cv': yoy_cv,
            'trend_correlation': trend_correlation,
            'autocorrelation': autocorr,
            'range_stability': range_stability,
            'n_observations': len(group_data)
        })
    
    return pd.DataFrame(metrics)


def calculate_predictability_score(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate a composite predictability score."""
    # Normalize metrics (higher values = less predictable)
    normalized_df = metrics_df.copy()
    
    # Higher CV = less predictable
    normalized_df['cv_score'] = normalized_df['coefficient_of_variation'] / normalized_df['coefficient_of_variation'].max()
    
    # Higher YoY volatility = less predictable  
    normalized_df['yoy_score'] = normalized_df['yoy_cv'] / normalized_df['yoy_cv'].max()
    
    # Lower trend correlation = less predictable (more erratic)
    normalized_df['trend_score'] = 1 - (normalized_df['trend_correlation'] + 1) / 2  # Convert [-1,1] to [0,1]
    
    # Lower autocorrelation = less predictable
    normalized_df['autocorr_score'] = 1 - (normalized_df['autocorrelation'] + 1) / 2  # Convert [-1,1] to [0,1]
    
    # Lower range stability = less predictable
    normalized_df['range_score'] = 1 - normalized_df['range_stability'].fillna(0.5)
    
    # Composite unpredictability score (0 = very predictable, 1 = very unpredictable)
    normalized_df['unpredictability_score'] = (
        normalized_df['cv_score'] * 0.25 +
        normalized_df['yoy_score'] * 0.25 +
        normalized_df['trend_score'] * 0.20 +
        normalized_df['autocorr_score'] * 0.15 +
        normalized_df['range_score'] * 0.15
    )
    
    # Predictability score (1 = very predictable, 0 = very unpredictable)
    normalized_df['predictability_score'] = 1 - normalized_df['unpredictability_score']
    
    return normalized_df


def analyze_countries(data_dir: Path) -> pd.DataFrame:
    """Analyze country volatility and predictability."""
    print("Analyzing country characteristics...")
    
    # Load country data
    country_df = load_all_data(
        data_dir=data_dir,
        countries=DEMOCRATIC_COUNTRIES,
        years=list(range(2000, 2024))
    )
    
    country_df = filter_panel_data(country_df)
    
    # Calculate composite standing scores
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
    
    # Calculate volatility metrics
    country_metrics = calculate_volatility_metrics(
        country_scores_df, 'standing_score', 'country'
    )
    
    # Calculate predictability scores
    country_analysis = calculate_predictability_score(country_metrics)
    
    return country_analysis


def analyze_industries(data_dir: Path) -> pd.DataFrame:
    """Analyze industry volatility and predictability."""
    print("Analyzing industry characteristics...")
    
    # Load company data
    company_df = load_all_company_data_merged(
        data_dir=data_dir,
        years=list(range(2000, 2024))
    )
    
    company_df = filter_company_panel_data(company_df)
    
    # Calculate composite dominance scores
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
    
    # Calculate volatility metrics
    industry_metrics = calculate_volatility_metrics(
        company_scores_df, 'dominance_score', 'industry'
    )
    
    # Calculate predictability scores
    industry_analysis = calculate_predictability_score(industry_metrics)
    
    return industry_analysis


def recommend_loco_candidates(
    country_analysis: pd.DataFrame,
    industry_analysis: pd.DataFrame,
    target_predictability_range: Tuple[float, float] = (0.3, 0.7)
) -> Dict:
    """Recommend LOCO candidates based on analysis."""
    
    # Filter countries in target predictability range
    country_candidates = country_analysis[
        (country_analysis['predictability_score'] >= target_predictability_range[0]) &
        (country_analysis['predictability_score'] <= target_predictability_range[1]) &
        (country_analysis['n_observations'] >= 15)
    ].copy()
    
    # Filter industries in target predictability range
    industry_candidates = industry_analysis[
        (industry_analysis['predictability_score'] >= target_predictability_range[0]) &
        (industry_analysis['predictability_score'] <= target_predictability_range[1]) &
        (industry_analysis['n_observations'] >= 15)
    ].copy()
    
    # Sort by proximity to middle of predictability range
    country_candidates['predictability_distance'] = abs(
        country_candidates['predictability_score'] - 0.5
    )
    industry_candidates['predictability_distance'] = abs(
        industry_candidates['predictability_score'] - 0.5
    )
    
    country_candidates = country_candidates.sort_values('predictability_distance')
    industry_candidates = industry_candidates.sort_values('predictability_distance')
    
    recommendations = {
        'country_candidates': country_candidates,
        'industry_candidates': industry_candidates,
        'top_country_recommendation': country_candidates.iloc[0] if len(country_candidates) > 0 else None,
        'top_industry_recommendation': industry_candidates.iloc[0] if len(industry_candidates) > 0 else None
    }
    
    return recommendations


def create_analysis_plots(
    country_analysis: pd.DataFrame,
    industry_analysis: pd.DataFrame,
    recommendations: Dict,
    save_dir: Path = Path("results")
):
    """Create visualization plots for the analysis."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LOCO Candidate Analysis: Volatility vs Predictability', fontsize=16)
    
    # Plot 1: Country Predictability vs Volatility
    ax1 = axes[0, 0]
    scatter = ax1.scatter(
        country_analysis['coefficient_of_variation'],
        country_analysis['predictability_score'],
        c=country_analysis['mean_value'],
        cmap='viridis',
        s=100,
        alpha=0.7
    )
    ax1.set_xlabel('Coefficient of Variation (Volatility)')
    ax1.set_ylabel('Predictability Score')
    ax1.set_title('Countries: Volatility vs Predictability')
    plt.colorbar(scatter, ax=ax1, label='Mean Standing Score')
    
    # Highlight recommended countries
    if recommendations['country_candidates'] is not None and len(recommendations['country_candidates']) > 0:
        candidates = recommendations['country_candidates']
        ax1.scatter(
            candidates['coefficient_of_variation'],
            candidates['predictability_score'],
            c='red', s=150, marker='s', alpha=0.8,
            label='LOCO Candidates'
        )
        
        # Annotate top recommendation
        top_country = recommendations['top_country_recommendation']
        if top_country is not None:
            ax1.annotate(
                f"{top_country.name}",
                (top_country['coefficient_of_variation'], top_country['predictability_score']),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Industry Predictability vs Volatility
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(
        industry_analysis['coefficient_of_variation'],
        industry_analysis['predictability_score'],
        c=industry_analysis['mean_value'],
        cmap='plasma',
        s=100,
        alpha=0.7
    )
    ax2.set_xlabel('Coefficient of Variation (Volatility)')
    ax2.set_ylabel('Predictability Score')
    ax2.set_title('Industries: Volatility vs Predictability')
    plt.colorbar(scatter2, ax=ax2, label='Mean Dominance Score')
    
    # Highlight recommended industries
    if recommendations['industry_candidates'] is not None and len(recommendations['industry_candidates']) > 0:
        candidates = recommendations['industry_candidates']
        ax2.scatter(
            candidates['coefficient_of_variation'],
            candidates['predictability_score'],
            c='red', s=150, marker='s', alpha=0.8,
            label='LOCO Candidates'
        )
        
        # Annotate top recommendation
        top_industry = recommendations['top_industry_recommendation']
        if top_industry is not None:
            ax2.annotate(
                f"{top_industry.name}",
                (top_industry['coefficient_of_variation'], top_industry['predictability_score']),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Country ranking by predictability
    ax3 = axes[1, 0]
    country_analysis_sorted = country_analysis.sort_values('predictability_score', ascending=True)
    bars = ax3.barh(country_analysis_sorted.index, country_analysis_sorted['predictability_score'])
    ax3.set_xlabel('Predictability Score')
    ax3.set_title('Countries Ranked by Predictability')
    ax3.grid(True, alpha=0.3)
    
    # Color bars based on whether they're candidates
    if recommendations['country_candidates'] is not None:
        candidate_countries = set(recommendations['country_candidates'].index)
        for i, (idx, bar) in enumerate(zip(country_analysis_sorted.index, bars)):
            if idx in candidate_countries:
                bar.set_color('red')
                bar.set_alpha(0.8)
    
    # Plot 4: Industry ranking by predictability
    ax4 = axes[1, 1]
    industry_analysis_sorted = industry_analysis.sort_values('predictability_score', ascending=True)
    bars2 = ax4.barh(industry_analysis_sorted.index, industry_analysis_sorted['predictability_score'])
    ax4.set_xlabel('Predictability Score')
    ax4.set_title('Industries Ranked by Predictability')
    ax4.grid(True, alpha=0.3)
    
    # Color bars based on whether they're candidates
    if recommendations['industry_candidates'] is not None:
        candidate_industries = set(recommendations['industry_candidates'].index)
        for i, (idx, bar) in enumerate(zip(industry_analysis_sorted.index, bars2)):
            if idx in candidate_industries:
                bar.set_color('red')
                bar.set_alpha(0.8)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'loco_candidate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Run the LOCO candidate analysis."""
    data_dir = Path("data")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("LOCO CANDIDATE ANALYSIS")
    print("=" * 80)
    print("Analyzing countries and industries to find optimal leave-one-out candidates")
    print("Target: Average volatility and predictability (not too stable, not too volatile)")
    print()
    
    # Analyze countries
    country_analysis = analyze_countries(data_dir)
    
    # Analyze industries  
    industry_analysis = analyze_industries(data_dir)
    
    # Get recommendations
    recommendations = recommend_loco_candidates(country_analysis, industry_analysis)
    
    # Display results
    print("COUNTRY ANALYSIS")
    print("-" * 50)
    print(f"Analyzed {len(country_analysis)} countries")
    print("\nTop 5 Most Predictable Countries (TOO STABLE - avoid):")
    most_predictable = country_analysis.nlargest(5, 'predictability_score')[['predictability_score', 'coefficient_of_variation', 'mean_value']]
    for idx, row in most_predictable.iterrows():
        print(f"  {idx}: Predictability={row['predictability_score']:.3f}, CV={row['coefficient_of_variation']:.3f}, Mean={row['mean_value']:.1f}")
    
    print("\nTop 5 Least Predictable Countries (TOO VOLATILE - avoid):")
    least_predictable = country_analysis.nsmallest(5, 'predictability_score')[['predictability_score', 'coefficient_of_variation', 'mean_value']]
    for idx, row in least_predictable.iterrows():
        print(f"  {idx}: Predictability={row['predictability_score']:.3f}, CV={row['coefficient_of_variation']:.3f}, Mean={row['mean_value']:.1f}")
    
    print("\nRECOMMENDED COUNTRY CANDIDATES:")
    if recommendations['country_candidates'] is not None and len(recommendations['country_candidates']) > 0:
        for idx, row in recommendations['country_candidates'].head(3).iterrows():
            print(f"  {idx}: Predictability={row['predictability_score']:.3f}, CV={row['coefficient_of_variation']:.3f}, Mean={row['mean_value']:.1f}")
    else:
        print("  No countries found in target predictability range")
    
    print("\n" + "=" * 80)
    print("INDUSTRY ANALYSIS")
    print("-" * 50)
    print(f"Analyzed {len(industry_analysis)} industries")
    print("\nTop 5 Most Predictable Industries (TOO STABLE - avoid):")
    most_predictable_ind = industry_analysis.nlargest(5, 'predictability_score')[['predictability_score', 'coefficient_of_variation', 'mean_value']]
    for idx, row in most_predictable_ind.iterrows():
        print(f"  {idx}: Predictability={row['predictability_score']:.3f}, CV={row['coefficient_of_variation']:.3f}, Mean={row['mean_value']:.1f}")
    
    print("\nTop 5 Least Predictable Industries (TOO VOLATILE - avoid):")
    least_predictable_ind = industry_analysis.nsmallest(5, 'predictability_score')[['predictability_score', 'coefficient_of_variation', 'mean_value']]
    for idx, row in least_predictable_ind.iterrows():
        print(f"  {idx}: Predictability={row['predictability_score']:.3f}, CV={row['coefficient_of_variation']:.3f}, Mean={row['mean_value']:.1f}")
    
    print("\nRECOMMENDED INDUSTRY CANDIDATES:")
    if recommendations['industry_candidates'] is not None and len(recommendations['industry_candidates']) > 0:
        for idx, row in recommendations['industry_candidates'].head(3).iterrows():
            print(f"  {idx}: Predictability={row['predictability_score']:.3f}, CV={row['coefficient_of_variation']:.3f}, Mean={row['mean_value']:.1f}")
    else:
        print("  No industries found in target predictability range")
    
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATIONS")
    print("-" * 50)
    
    if recommendations['top_country_recommendation'] is not None:
        top_country = recommendations['top_country_recommendation']
        print(f"RECOMMENDED COUNTRY: {top_country.name}")
        print(f"  Predictability Score: {top_country['predictability_score']:.3f} (target: 0.3-0.7)")
        print(f"  Coefficient of Variation: {top_country['coefficient_of_variation']:.3f}")
        print(f"  Mean Standing Score: {top_country['mean_value']:.1f}")
        print(f"  Observations: {top_country['n_observations']}")
        print(f"  Reasoning: Moderate volatility, average predictability - good for unbiased validation")
    else:
        print("No suitable country candidate found in target range")
    
    if recommendations['top_industry_recommendation'] is not None:
        top_industry = recommendations['top_industry_recommendation']
        print(f"\nRECOMMENDED INDUSTRY: {top_industry.name}")
        print(f"  Predictability Score: {top_industry['predictability_score']:.3f} (target: 0.3-0.7)")
        print(f"  Coefficient of Variation: {top_industry['coefficient_of_variation']:.3f}")
        print(f"  Mean Dominance Score: {top_industry['mean_value']:.1f}")
        print(f"  Observations: {top_industry['n_observations']}")
        print(f"  Reasoning: Moderate volatility, average predictability - good for unbiased validation")
    else:
        print("\nNo suitable industry candidate found in target range")
    
    # Save detailed results
    country_analysis.to_csv(results_dir / 'country_volatility_analysis.csv')
    industry_analysis.to_csv(results_dir / 'industry_volatility_analysis.csv')
    
    if recommendations['country_candidates'] is not None:
        recommendations['country_candidates'].to_csv(results_dir / 'country_loco_candidates.csv')
    if recommendations['industry_candidates'] is not None:
        recommendations['industry_candidates'].to_csv(results_dir / 'industry_loco_candidates.csv')
    
    print(f"\nDetailed results saved to {results_dir}/")
    print("Files created:")
    print("  - country_volatility_analysis.csv")
    print("  - industry_volatility_analysis.csv")
    print("  - country_loco_candidates.csv")
    print("  - industry_loco_candidates.csv")
    
    # Create visualization
    try:
        create_analysis_plots(country_analysis, industry_analysis, recommendations, results_dir)
        print("  - loco_candidate_analysis.png")
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
