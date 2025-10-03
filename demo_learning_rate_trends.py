#!/usr/bin/env python3
"""
Demo script showing corrected corruption and geography effects.

This script demonstrates how:
1. Corruption affects learning rate (gradient trust)
2. Geography affects growth trends (derivatives)
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
    compute_data_confidence,
    compute_geography_growth_multiplier,
    compute_learning_rate_adjustment,
    compute_trend_adjustment,
    get_factor_values,
    calculate_corruption_index,
    calculate_geography_advantage,
    get_country_corruption_level,
    get_country_geography_advantages
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


def analyze_learning_rates():
    """Analyze how corruption affects learning rates."""
    print("=" * 80)
    print("LEARNING RATE ANALYSIS (Corruption → Gradient Trust)")
    print("=" * 80)
    
    base_learning_rate = 0.01
    sample_countries = ["DNK", "USA", "ITA", "CHN", "RUS", "VEN"]
    
    print(f"Base Learning Rate: {base_learning_rate}")
    print("\nCountry Learning Rate Adjustments:")
    print("-" * 60)
    print(f"{'Country':<8} {'Corruption':<12} {'Trust Score':<12} {'Learning Rate':<15} {'Impact'}")
    print("-" * 60)
    
    for country in sample_countries:
        corruption_level = get_country_corruption_level(country)
        trust_score = calculate_corruption_index(country)
        adjusted_lr = compute_learning_rate_adjustment(trust_score, base_learning_rate)
        impact = f"{(adjusted_lr/base_learning_rate)*100:.0f}% of base"
        
        print(f"{country:<8} {corruption_level:<12} {trust_score:<12.3f} {adjusted_lr:<15.4f} {impact}")
    
    print("\nInterpretation:")
    print("- Higher corruption → Lower learning rate (be more conservative)")
    print("- Lower corruption → Higher learning rate (trust gradients more)")
    print("- This affects how much the model learns from each country's data")


def analyze_growth_trends():
    """Analyze how geography affects growth trends."""
    print("\n" + "=" * 80)
    print("GROWTH TREND ANALYSIS (Geography → Derivative Adjustment)")
    print("=" * 80)
    
    base_trend = 2.0  # 2% annual growth baseline
    sample_countries = ["USA", "GBR", "JPN", "AUS", "CHE", "MEX", "BRA"]
    
    print(f"Base Growth Trend: {base_trend}% annually")
    print("\nCountry Growth Trend Adjustments:")
    print("-" * 70)
    print(f"{'Country':<8} {'Geography':<15} {'Multiplier':<12} {'Adjusted Trend':<15} {'Impact'}")
    print("-" * 70)
    
    for country in sample_countries:
        geo_advantages = get_country_geography_advantages(country)
        geo_multiplier = calculate_geography_advantage(country)
        adjusted_trend = compute_trend_adjustment(geo_multiplier, base_trend)
        
        # Show geography advantages in a compact format
        geo_str = ", ".join(geo_advantages[:2])  # Show first 2 advantages
        if len(geo_advantages) > 2:
            geo_str += f" (+{len(geo_advantages)-2} more)"
        
        impact = f"{adjusted_trend-base_trend:+.1f}% vs base"
        
        print(f"{country:<8} {geo_str:<15} {geo_multiplier:<12.3f} {adjusted_trend:<15.1f} {impact}")
    
    print("\nInterpretation:")
    print("- Higher geography multiplier → More optimistic growth trends")
    print("- Lower geography multiplier → More pessimistic growth trends")
    print("- This affects the derivative (rate of change) over time, not current standing")


def simulate_training_scenario(data_dir: Path):
    """Simulate how different countries would affect model training."""
    print("\n" + "=" * 80)
    print("TRAINING SCENARIO SIMULATION")
    print("=" * 80)
    
    # Load sample data
    country_df = load_all_data(
        data_dir=data_dir,
        countries=["USA", "DNK", "CHN", "CHE", "GBR"],
        years=list(range(2020, 2024))
    )
    
    country_df = filter_panel_data(country_df)
    
    # Get latest year data
    latest_year = country_df['year'].max()
    latest_data = country_df[country_df['year'] == latest_year].copy()
    
    base_lr = 0.01
    base_trend = 1.5  # 1.5% baseline trend
    
    training_scenarios = []
    
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
        
        # Calculate metrics
        base_score = compute_composite_standing(factor_values)
        data_confidence = compute_data_confidence(factor_values)
        geography_multiplier = compute_geography_growth_multiplier(factor_values)
        
        adjusted_lr = compute_learning_rate_adjustment(data_confidence, base_lr)
        adjusted_trend = compute_trend_adjustment(geography_multiplier, base_trend)
        
        training_scenarios.append({
            'country': country,
            'base_score': base_score,
            'data_confidence': data_confidence,
            'geography_multiplier': geography_multiplier,
            'learning_rate': adjusted_lr,
            'growth_trend': adjusted_trend,
            'lr_ratio': adjusted_lr / base_lr,
            'trend_ratio': adjusted_trend / base_trend
        })
    
    scenarios_df = pd.DataFrame(training_scenarios)
    scenarios_df = scenarios_df.sort_values('data_confidence', ascending=False)
    
    print("Model Training Impact by Country:")
    print("-" * 80)
    print(f"{'Country':<8} {'Base Score':<10} {'Confidence':<10} {'LR Ratio':<8} {'Trend Ratio':<10} {'Training Strategy'}")
    print("-" * 80)
    
    for _, row in scenarios_df.iterrows():
        # Determine training strategy based on confidence and geography
        if row['data_confidence'] > 0.8 and row['geography_multiplier'] > 1.1:
            strategy = "Aggressive learning"
        elif row['data_confidence'] > 0.6 and row['geography_multiplier'] > 0.9:
            strategy = "Balanced learning"
        elif row['data_confidence'] > 0.4:
            strategy = "Conservative learning"
        else:
            strategy = "Minimal learning"
        
        print(f"{row['country']:<8} {row['base_score']:<10.1f} {row['data_confidence']:<10.3f} "
              f"{row['lr_ratio']:<8.2f} {row['trend_ratio']:<10.2f} {strategy}")
    
    return scenarios_df


def create_visualization(scenarios_df: pd.DataFrame, save_dir: Path = Path("results")):
    """Create visualization of learning rate and trend effects."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Corruption & Geography Effects on Model Training', fontsize=16)
    
    # Plot 1: Learning Rate vs Data Confidence
    ax1 = axes[0, 0]
    scatter = ax1.scatter(
        scenarios_df['data_confidence'],
        scenarios_df['lr_ratio'],
        c=scenarios_df['base_score'],
        cmap='RdYlGn',
        s=200,
        alpha=0.7
    )
    ax1.set_xlabel('Data Confidence (Corruption Trust Score)')
    ax1.set_ylabel('Learning Rate Ratio (vs Base)')
    ax1.set_title('Corruption → Learning Rate Effect')
    plt.colorbar(scatter, ax=ax1, label='Base Standing Score')
    ax1.grid(True, alpha=0.3)
    
    # Add reference line
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Base Learning Rate')
    
    # Annotate countries
    for _, row in scenarios_df.iterrows():
        ax1.annotate(row['country'], 
                    (row['data_confidence'], row['lr_ratio']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax1.legend()
    
    # Plot 2: Growth Trend vs Geography Multiplier
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(
        scenarios_df['geography_multiplier'],
        scenarios_df['trend_ratio'],
        c=scenarios_df['base_score'],
        cmap='RdYlBu',
        s=200,
        alpha=0.7
    )
    ax2.set_xlabel('Geography Multiplier')
    ax2.set_ylabel('Growth Trend Ratio (vs Base)')
    ax2.set_title('Geography → Growth Trend Effect')
    plt.colorbar(scatter2, ax=ax2, label='Base Standing Score')
    ax2.grid(True, alpha=0.3)
    
    # Add reference line
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Base Trend')
    
    # Annotate countries
    for _, row in scenarios_df.iterrows():
        ax2.annotate(row['country'], 
                    (row['geography_multiplier'], row['trend_ratio']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax2.legend()
    
    # Plot 3: Combined Learning Strategy
    ax3 = axes[1, 0]
    colors = ['green' if x > 0.7 else 'orange' if x > 0.5 else 'red' for x in scenarios_df['data_confidence']]
    bars = ax3.barh(scenarios_df['country'], scenarios_df['lr_ratio'], color=colors, alpha=0.7)
    ax3.set_xlabel('Learning Rate Ratio')
    ax3.set_title('Training Intensity by Data Confidence')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
    
    # Add color legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='High Confidence (>0.7)'),
                      Patch(facecolor='orange', alpha=0.7, label='Medium Confidence (0.5-0.7)'),
                      Patch(facecolor='red', alpha=0.7, label='Low Confidence (<0.5)')]
    ax3.legend(handles=legend_elements, loc='lower right')
    
    # Plot 4: Growth Optimism
    ax4 = axes[1, 1]
    colors2 = ['blue' if x > 1.1 else 'gray' if x > 0.9 else 'red' for x in scenarios_df['geography_multiplier']]
    bars2 = ax4.barh(scenarios_df['country'], scenarios_df['trend_ratio'], color=colors2, alpha=0.7)
    ax4.set_xlabel('Growth Trend Ratio')
    ax4.set_title('Growth Optimism by Geography')
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
    
    # Add color legend
    legend_elements2 = [Patch(facecolor='blue', alpha=0.7, label='High Advantage (>1.1)'),
                       Patch(facecolor='gray', alpha=0.7, label='Neutral (0.9-1.1)'),
                       Patch(facecolor='red', alpha=0.7, label='Low Advantage (<0.9)')]
    ax4.legend(handles=legend_elements2, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'learning_rate_trend_effects.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Run the corrected corruption and geography analysis demo."""
    data_dir = Path("data")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("CORRECTED CORRUPTION & GEOGRAPHY EFFECTS DEMO")
    print("=" * 80)
    print("This demo shows the CORRECTED understanding:")
    print("1. Corruption → Affects learning rate (gradient trust)")
    print("2. Geography → Affects growth trends (derivatives)")
    print()
    
    # Analyze learning rate effects
    analyze_learning_rates()
    
    # Analyze growth trend effects
    analyze_growth_trends()
    
    # Simulate training scenario
    scenarios_df = simulate_training_scenario(data_dir)
    
    # Create visualization
    try:
        create_visualization(scenarios_df, results_dir)
        print(f"\nVisualization saved to {results_dir}/learning_rate_trend_effects.png")
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
    # Save results
    scenarios_df.to_csv(results_dir / 'learning_rate_trend_analysis.csv', index=False)
    print(f"Detailed results saved to {results_dir}/learning_rate_trend_analysis.csv")
    
    # Show key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    print(f"1. Learning Rate Range: {scenarios_df['lr_ratio'].min():.2f}x to {scenarios_df['lr_ratio'].max():.2f}x base rate")
    print(f"2. Growth Trend Range: {scenarios_df['trend_ratio'].min():.2f}x to {scenarios_df['trend_ratio'].max():.2f}x base trend")
    
    print("\nTraining Strategy Summary:")
    high_confidence = scenarios_df[scenarios_df['data_confidence'] > 0.7]
    if len(high_confidence) > 0:
        print(f"  High Confidence Countries: {', '.join(high_confidence['country'])}")
        print(f"  → Use aggressive learning rates (trust gradients)")
    
    low_confidence = scenarios_df[scenarios_df['data_confidence'] < 0.5]
    if len(low_confidence) > 0:
        print(f"  Low Confidence Countries: {', '.join(low_confidence['country'])}")
        print(f"  → Use conservative learning rates (be cautious)")
    
    high_geography = scenarios_df[scenarios_df['geography_multiplier'] > 1.1]
    if len(high_geography) > 0:
        print(f"  High Geography Advantage: {', '.join(high_geography['country'])}")
        print(f"  → Apply optimistic growth trends")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
