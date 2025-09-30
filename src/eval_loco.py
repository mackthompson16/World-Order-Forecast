"""
Leave-One-Country-Out (LOCO) evaluation utilities.

Runs LOCO evaluation across countries, saves per-country metrics,
and generates plots for analysis.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import pickle
from tqdm import tqdm

from .data_schema import CountryData, FACTOR_NAMES, FORECAST_HORIZONS
from .features import FactorScaler, windowify_panel, create_data_loaders
from .models.forecast_net import CountryStandingForecastNet, create_model
from .metrics import ForecastMetrics
from .training import ForecastTrainer, loco_training

logger = logging.getLogger(__name__)


class LOCOEvaluator:
    """Leave-One-Country-Out evaluation class."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cpu",
        results_dir: Optional[Path] = None
    ):
        """
        Initialize LOCOEvaluator.
        
        Args:
            config: Model configuration
            device: Device to run evaluation on
            results_dir: Directory to save results
        """
        self.config = config
        self.device = device
        self.results_dir = results_dir or Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics calculator
        self.metrics_calculator = ForecastMetrics(
            horizons=config["model"]["horizons"],
            quantiles=config["model"].get("quantiles", [0.1, 0.5, 0.9])
        )
        
        # Results storage
        self.loco_results = {}
        self.country_metrics = {}
        self.aggregated_metrics = {}
    
    def run_loco_evaluation(
        self,
        panel_data: List[CountryData],
        save_models: bool = True,
        save_predictions: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete LOCO evaluation.
        
        Args:
            panel_data: List of CountryData objects
            save_models: Whether to save individual models
            save_predictions: Whether to save predictions
            
        Returns:
            Complete LOCO results
        """
        logger.info("Starting LOCO evaluation")
        
        # Run LOCO training
        model_dir = self.results_dir / "models" if save_models else None
        loco_results = loco_training(
            panel_data=panel_data,
            config=self.config,
            window_length=self.config["model"]["window_length"],
            device=self.device,
            save_dir=model_dir
        )
        
        self.loco_results = loco_results
        
        # Calculate metrics for each country
        self.country_metrics = self.metrics_calculator.calculate_country_metrics(loco_results)
        
        # Aggregate metrics across countries
        self.aggregated_metrics = self.metrics_calculator.aggregate_metrics(self.country_metrics)
        
        # Save results
        self._save_results()
        
        # Generate plots
        self._generate_plots()
        
        # Generate summary report
        self._generate_summary_report()
        
        logger.info("LOCO evaluation completed")
        
        return {
            "loco_results": loco_results,
            "country_metrics": self.country_metrics,
            "aggregated_metrics": self.aggregated_metrics
        }
    
    def _save_results(self):
        """Save LOCO results to files."""
        # Save aggregated metrics
        metrics_df = self._metrics_to_dataframe(self.aggregated_metrics)
        metrics_df.to_csv(self.results_dir / "loco_aggregated_metrics.csv", index=True)
        
        # Save per-country metrics
        country_metrics_df = self._country_metrics_to_dataframe()
        country_metrics_df.to_csv(self.results_dir / "loco_country_metrics.csv", index=True)
        
        # Save detailed results
        with open(self.results_dir / "loco_results.pkl", "wb") as f:
            pickle.dump(self.loco_results, f)
        
        logger.info(f"Results saved to {self.results_dir}")
    
    def _metrics_to_dataframe(self, metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Convert metrics dictionary to DataFrame."""
        rows = []
        for horizon_key, horizon_metrics in metrics.items():
            row = {"horizon": horizon_key}
            row.update(horizon_metrics)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _country_metrics_to_dataframe(self) -> pd.DataFrame:
        """Convert country metrics to DataFrame."""
        rows = []
        for country, country_metrics in self.country_metrics.items():
            for horizon_key, horizon_metrics in country_metrics.items():
                row = {"country": country, "horizon": horizon_key}
                row.update(horizon_metrics)
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _generate_plots(self):
        """Generate evaluation plots."""
        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")
        
        # 1. Metrics comparison across countries
        self._plot_country_metrics_comparison()
        
        # 2. Horizon performance comparison
        self._plot_horizon_performance()
        
        # 3. Error distribution plots
        self._plot_error_distributions()
        
        # 4. Country ranking plots
        self._plot_country_rankings()
        
        logger.info("Plots generated and saved")
    
    def _plot_country_metrics_comparison(self):
        """Plot metrics comparison across countries."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("LOCO Evaluation: Country Metrics Comparison", fontsize=16)
        
        # Prepare data
        country_df = self._country_metrics_to_dataframe()
        
        # MAE comparison
        mae_data = country_df.pivot(index="country", columns="horizon", values="mae")
        mae_data.plot(kind="bar", ax=axes[0, 0], title="MAE by Country and Horizon")
        axes[0, 0].set_ylabel("MAE")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        rmse_data = country_df.pivot(index="country", columns="horizon", values="rmse")
        rmse_data.plot(kind="bar", ax=axes[0, 1], title="RMSE by Country and Horizon")
        axes[0, 1].set_ylabel("RMSE")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Spearman correlation comparison
        corr_data = country_df.pivot(index="country", columns="horizon", values="spearman_corr")
        corr_data.plot(kind="bar", ax=axes[1, 0], title="Spearman Correlation by Country and Horizon")
        axes[1, 0].set_ylabel("Spearman Correlation")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # MASE comparison (if available)
        if "mase" in country_df.columns:
            mase_data = country_df.pivot(index="country", columns="horizon", values="mase")
            mase_data.plot(kind="bar", ax=axes[1, 1], title="MASE by Country and Horizon")
            axes[1, 1].set_ylabel("MASE")
        else:
            axes[1, 1].text(0.5, 0.5, "MASE not available", ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("MASE by Country and Horizon")
        
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "country_metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_horizon_performance(self):
        """Plot performance across different horizons."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("LOCO Evaluation: Horizon Performance", fontsize=16)
        
        # Prepare data
        country_df = self._country_metrics_to_dataframe()
        
        # MAE by horizon
        mae_by_horizon = country_df.groupby("horizon")["mae"].agg(['mean', 'std', 'min', 'max'])
        mae_by_horizon['mean'].plot(kind='bar', ax=axes[0, 0], title="MAE by Horizon", yerr=mae_by_horizon['std'])
        axes[0, 0].set_ylabel("MAE")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE by horizon
        rmse_by_horizon = country_df.groupby("horizon")["rmse"].agg(['mean', 'std', 'min', 'max'])
        rmse_by_horizon['mean'].plot(kind='bar', ax=axes[0, 1], title="RMSE by Horizon", yerr=rmse_by_horizon['std'])
        axes[0, 1].set_ylabel("RMSE")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Spearman correlation by horizon
        corr_by_horizon = country_df.groupby("horizon")["spearman_corr"].agg(['mean', 'std', 'min', 'max'])
        corr_by_horizon['mean'].plot(kind='bar', ax=axes[1, 0], title="Spearman Correlation by Horizon", yerr=corr_by_horizon['std'])
        axes[1, 0].set_ylabel("Spearman Correlation")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Box plot of MAE distribution by horizon
        country_df.boxplot(column="mae", by="horizon", ax=axes[1, 1])
        axes[1, 1].set_title("MAE Distribution by Horizon")
        axes[1, 1].set_xlabel("Horizon")
        axes[1, 1].set_ylabel("MAE")
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "horizon_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_distributions(self):
        """Plot error distributions."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("LOCO Evaluation: Error Distributions", fontsize=16)
        
        # Prepare data
        country_df = self._country_metrics_to_dataframe()
        
        # MAE distribution
        axes[0].hist(country_df["mae"], bins=20, alpha=0.7, edgecolor='black')
        axes[0].set_title("MAE Distribution")
        axes[0].set_xlabel("MAE")
        axes[0].set_ylabel("Frequency")
        
        # RMSE distribution
        axes[1].hist(country_df["rmse"], bins=20, alpha=0.7, edgecolor='black')
        axes[1].set_title("RMSE Distribution")
        axes[1].set_xlabel("RMSE")
        axes[1].set_ylabel("Frequency")
        
        # Spearman correlation distribution
        axes[2].hist(country_df["spearman_corr"], bins=20, alpha=0.7, edgecolor='black')
        axes[2].set_title("Spearman Correlation Distribution")
        axes[2].set_xlabel("Spearman Correlation")
        axes[2].set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "error_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_country_rankings(self):
        """Plot country rankings by performance."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("LOCO Evaluation: Country Rankings", fontsize=16)
        
        # Prepare data
        country_df = self._country_metrics_to_dataframe()
        
        # Average MAE ranking
        avg_mae = country_df.groupby("country")["mae"].mean().sort_values()
        avg_mae.plot(kind="barh", ax=axes[0, 0], title="Average MAE Ranking (Lower is Better)")
        axes[0, 0].set_xlabel("Average MAE")
        
        # Average RMSE ranking
        avg_rmse = country_df.groupby("country")["rmse"].mean().sort_values()
        avg_rmse.plot(kind="barh", ax=axes[0, 1], title="Average RMSE Ranking (Lower is Better)")
        axes[0, 1].set_xlabel("Average RMSE")
        
        # Average Spearman correlation ranking
        avg_corr = country_df.groupby("country")["spearman_corr"].mean().sort_values(ascending=False)
        avg_corr.plot(kind="barh", ax=axes[1, 0], title="Average Spearman Correlation Ranking (Higher is Better)")
        axes[1, 0].set_xlabel("Average Spearman Correlation")
        
        # Overall ranking (composite score)
        # Normalize metrics and create composite score
        mae_norm = 1 - (country_df.groupby("country")["mae"].mean() / country_df["mae"].max())
        rmse_norm = 1 - (country_df.groupby("country")["rmse"].mean() / country_df["rmse"].max())
        corr_norm = country_df.groupby("country")["spearman_corr"].mean()
        
        composite_score = (mae_norm + rmse_norm + corr_norm) / 3
        composite_score = composite_score.sort_values(ascending=False)
        
        composite_score.plot(kind="barh", ax=axes[1, 1], title="Overall Performance Ranking")
        axes[1, 1].set_xlabel("Composite Score")
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "country_rankings.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self):
        """Generate summary report."""
        report_path = self.results_dir / "loco_summary_report.txt"
        
        with open(report_path, "w") as f:
            f.write("LOCO Evaluation Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Number of countries evaluated: {len(self.country_metrics)}\n")
            f.write(f"Number of horizons: {len(self.config['model']['horizons'])}\n")
            f.write(f"Total evaluations: {len(self.country_metrics) * len(self.config['model']['horizons'])}\n\n")
            
            # Best and worst performing countries
            country_df = self._country_metrics_to_dataframe()
            avg_mae = country_df.groupby("country")["mae"].mean()
            
            f.write("Performance Rankings:\n")
            f.write("-" * 20 + "\n")
            f.write("Best performing countries (lowest MAE):\n")
            for i, (country, mae) in enumerate(avg_mae.nsmallest(5).items()):
                f.write(f"  {i+1}. {country}: {mae:.4f}\n")
            
            f.write("\nWorst performing countries (highest MAE):\n")
            for i, (country, mae) in enumerate(avg_mae.nlargest(5).items()):
                f.write(f"  {i+1}. {country}: {mae:.4f}\n")
            
            # Horizon performance
            f.write("\nHorizon Performance:\n")
            f.write("-" * 20 + "\n")
            for horizon_key, metrics in self.aggregated_metrics.items():
                f.write(f"{horizon_key}:\n")
                f.write(f"  Average MAE: {metrics.get('mae_mean', 'N/A'):.4f} ± {metrics.get('mae_std', 'N/A'):.4f}\n")
                f.write(f"  Average RMSE: {metrics.get('rmse_mean', 'N/A'):.4f} ± {metrics.get('rmse_std', 'N/A'):.4f}\n")
                f.write(f"  Average Spearman Correlation: {metrics.get('spearman_corr_mean', 'N/A'):.4f} ± {metrics.get('spearman_corr_std', 'N/A'):.4f}\n")
                f.write("\n")
        
        logger.info(f"Summary report saved to {report_path}")
    
    def load_results(self, results_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Load previously saved LOCO results."""
        if results_dir is not None:
            self.results_dir = results_dir
        
        results_path = self.results_dir / "loco_results.pkl"
        
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        with open(results_path, "rb") as f:
            self.loco_results = pickle.load(f)
        
        # Recalculate metrics
        self.country_metrics = self.metrics_calculator.calculate_country_metrics(self.loco_results)
        self.aggregated_metrics = self.metrics_calculator.aggregate_metrics(self.country_metrics)
        
        logger.info(f"Results loaded from {results_dir}")
        
        return {
            "loco_results": self.loco_results,
            "country_metrics": self.country_metrics,
            "aggregated_metrics": self.aggregated_metrics
        }


def run_loco_evaluation(
    panel_data: List[CountryData],
    config: Dict[str, Any],
    device: str = "cpu",
    results_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Convenience function to run LOCO evaluation.
    
    Args:
        panel_data: List of CountryData objects
        config: Model configuration
        device: Device to run evaluation on
        results_dir: Directory to save results
        
    Returns:
        LOCO evaluation results
    """
    evaluator = LOCOEvaluator(config, device, results_dir)
    return evaluator.run_loco_evaluation(panel_data)
