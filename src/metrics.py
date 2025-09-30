"""
Evaluation metrics for country standing forecasting.

Implements MAE, RMSE, MASE, Spearman correlation, and interval coverage
metrics for evaluating forecast performance.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mean_absolute_scaled_error(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_train: Optional[np.ndarray] = None,
    seasonal_period: int = 1
) -> float:
    """
    Calculate Mean Absolute Scaled Error (MASE).
    
    MASE = MAE / MAE_naive_forecast
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_train: Training data for naive forecast baseline
        seasonal_period: Seasonal period for naive forecast
        
    Returns:
        MASE value
    """
    mae = mean_absolute_error(y_true, y_pred)
    
    if y_train is not None:
        # Use training data for naive forecast
        naive_forecast = np.mean(np.abs(np.diff(y_train, n=seasonal_period)))
    else:
        # Use test data for naive forecast (less ideal)
        naive_forecast = np.mean(np.abs(np.diff(y_true, n=seasonal_period)))
    
    if naive_forecast == 0:
        logger.warning("Naive forecast error is zero, returning MAE instead of MASE")
        return mae
    
    return mae / naive_forecast


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Spearman rank correlation coefficient.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Tuple of (correlation, p_value)
    """
    correlation, p_value = spearmanr(y_true, y_pred)
    return correlation, p_value


def interval_coverage(
    y_true: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    target_coverage: float = 0.8
) -> float:
    """
    Calculate interval coverage rate.
    
    Args:
        y_true: True values
        lower_bound: Lower bound of prediction interval
        upper_bound: Upper bound of prediction interval
        target_coverage: Target coverage rate (e.g., 0.8 for 80% interval)
        
    Returns:
        Actual coverage rate
    """
    covered = np.logical_and(y_true >= lower_bound, y_true <= upper_bound)
    return np.mean(covered)


def pinball_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    quantile: float
) -> torch.Tensor:
    """
    Calculate pinball loss for quantile regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        quantile: Target quantile (e.g., 0.5 for median)
        
    Returns:
        Pinball loss
    """
    error = y_true - y_pred
    return torch.mean(torch.max(
        (quantile - 1) * error,
        quantile * error
    ))


def quantile_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    quantiles: List[float]
) -> torch.Tensor:
    """
    Calculate quantile loss for multiple quantiles.
    
    Args:
        y_true: True values
        y_pred: Predicted quantiles
        quantiles: List of target quantiles
        
    Returns:
        Total quantile loss
    """
    total_loss = 0.0
    n_quantiles = len(quantiles)
    
    for i, q in enumerate(quantiles):
        loss = pinball_loss(y_true, y_pred[:, i], q)
        total_loss += loss
    
    return total_loss / n_quantiles


class ForecastMetrics:
    """
    Comprehensive metrics calculator for forecasting evaluation.
    """
    
    def __init__(
        self,
        horizons: List[int] = [1, 5, 10],
        quantiles: Optional[List[float]] = None,
        target_coverage: float = 0.8
    ):
        """
        Initialize ForecastMetrics.
        
        Args:
            horizons: List of forecast horizons
            quantiles: List of quantiles for uncertainty estimation
            target_coverage: Target coverage rate for intervals
        """
        self.horizons = horizons
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.target_coverage = target_coverage
    
    def calculate_metrics(
        self,
        y_true: Dict[str, np.ndarray],
        y_pred: Dict[str, np.ndarray],
        y_train: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive metrics for all horizons.
        
        Args:
            y_true: Dictionary mapping horizon to true values
            y_pred: Dictionary mapping horizon to predicted values
            y_train: Optional training data for MASE calculation
            
        Returns:
            Dictionary of metrics for each horizon
        """
        metrics = {}
        
        for horizon in self.horizons:
            horizon_key = f"standing_{horizon}y"
            
            if horizon_key not in y_true or horizon_key not in y_pred:
                logger.warning(f"Missing data for horizon {horizon}")
                continue
            
            true_vals = y_true[horizon_key]
            pred_vals = y_pred[horizon_key]
            
            # Remove NaN values
            valid_mask = ~(np.isnan(true_vals) | np.isnan(pred_vals))
            true_vals = true_vals[valid_mask]
            pred_vals = pred_vals[valid_mask]
            
            if len(true_vals) == 0:
                logger.warning(f"No valid data for horizon {horizon}")
                continue
            
            # Calculate basic metrics
            horizon_metrics = {
                "mae": mean_absolute_error(true_vals, pred_vals),
                "rmse": root_mean_squared_error(true_vals, pred_vals),
                "spearman_corr": spearman_correlation(true_vals, pred_vals)[0],
                "spearman_pvalue": spearman_correlation(true_vals, pred_vals)[1]
            }
            
            # Calculate MASE if training data available
            if y_train is not None and horizon_key in y_train:
                train_vals = y_train[horizon_key]
                train_vals = train_vals[~np.isnan(train_vals)]
                if len(train_vals) > 0:
                    horizon_metrics["mase"] = mean_absolute_scaled_error(
                        true_vals, pred_vals, train_vals
                    )
            
            # Calculate quantile metrics if available
            quantile_key = f"quantiles_{horizon}y"
            if quantile_key in y_pred:
                quantile_preds = y_pred[quantile_key]
                
                # Interval coverage
                if len(self.quantiles) >= 2:
                    lower_idx = 0
                    upper_idx = len(self.quantiles) - 1
                    
                    coverage = interval_coverage(
                        true_vals,
                        quantile_preds[:, lower_idx],
                        quantile_preds[:, upper_idx],
                        self.target_coverage
                    )
                    horizon_metrics["interval_coverage"] = coverage
                
                # Individual quantile performance
                for i, q in enumerate(self.quantiles):
                    if i < quantile_preds.shape[1]:
                        quantile_pred = quantile_preds[:, i]
                        quantile_pred = quantile_pred[valid_mask]
                        
                        horizon_metrics[f"quantile_{q}_mae"] = mean_absolute_error(
                            true_vals, quantile_pred
                        )
            
            metrics[f"horizon_{horizon}y"] = horizon_metrics
        
        return metrics
    
    def calculate_country_metrics(
        self,
        country_results: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Calculate metrics for each country separately.
        
        Args:
            country_results: Dictionary mapping country to results
            
        Returns:
            Dictionary mapping country to metrics
        """
        country_metrics = {}
        
        for country, results in country_results.items():
            country_metrics[country] = self.calculate_metrics(results)
        
        return country_metrics
    
    def aggregate_metrics(
        self,
        country_metrics: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics across countries.
        
        Args:
            country_metrics: Dictionary mapping country to metrics
            
        Returns:
            Aggregated metrics
        """
        aggregated = {}
        
        for horizon in self.horizons:
            horizon_key = f"horizon_{horizon}y"
            
            # Collect metrics for this horizon across all countries
            horizon_metrics = {}
            for country, metrics in country_metrics.items():
                if horizon_key in metrics:
                    for metric_name, metric_value in metrics[horizon_key].items():
                        if metric_name not in horizon_metrics:
                            horizon_metrics[metric_name] = []
                        horizon_metrics[metric_name].append(metric_value)
            
            # Calculate aggregated statistics
            aggregated[horizon_key] = {}
            for metric_name, values in horizon_metrics.items():
                if values:
                    aggregated[horizon_key][f"{metric_name}_mean"] = np.mean(values)
                    aggregated[horizon_key][f"{metric_name}_std"] = np.std(values)
                    aggregated[horizon_key][f"{metric_name}_median"] = np.median(values)
                    aggregated[horizon_key][f"{metric_name}_min"] = np.min(values)
                    aggregated[horizon_key][f"{metric_name}_max"] = np.max(values)
        
        return aggregated
    
    def print_metrics_summary(
        self,
        metrics: Dict[str, Dict[str, float]],
        title: str = "Forecast Metrics"
    ):
        """
        Print formatted metrics summary.
        
        Args:
            metrics: Dictionary of metrics
            title: Title for the summary
        """
        print(f"\n{title}")
        print("=" * len(title))
        
        for horizon_key, horizon_metrics in metrics.items():
            print(f"\n{horizon_key.upper()}:")
            print("-" * len(horizon_key))
            
            for metric_name, metric_value in horizon_metrics.items():
                if isinstance(metric_value, float):
                    print(f"  {metric_name}: {metric_value:.4f}")
                else:
                    print(f"  {metric_name}: {metric_value}")


def calculate_directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prev: Optional[np.ndarray] = None
) -> float:
    """
    Calculate directional accuracy (percentage of correct direction predictions).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_prev: Previous values for calculating direction
        
    Returns:
        Directional accuracy percentage
    """
    if y_prev is None:
        # Use first-order differences
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
    else:
        # Use direction relative to previous values
        true_direction = (y_true - y_prev) > 0
        pred_direction = (y_pred - y_prev) > 0
    
    # Calculate accuracy
    correct = np.sum(true_direction == pred_direction)
    total = len(true_direction)
    
    return correct / total if total > 0 else 0.0


def calculate_forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate forecast bias (mean error).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Forecast bias
    """
    return np.mean(y_pred - y_true)


def calculate_forecast_efficiency(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_naive: np.ndarray
) -> float:
    """
    Calculate forecast efficiency relative to naive forecast.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_naive: Naive forecast values
        
    Returns:
        Forecast efficiency (1 - MSE_model / MSE_naive)
    """
    mse_model = np.mean((y_true - y_pred) ** 2)
    mse_naive = np.mean((y_true - y_naive) ** 2)
    
    if mse_naive == 0:
        return 0.0
    
    return 1 - (mse_model / mse_naive)
