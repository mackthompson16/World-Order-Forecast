"""
Feature engineering and preprocessing utilities.

Handles standardization, differencing, windowing, and tensor creation
for the country standing forecast model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
import torch
from torch.utils.data import Dataset, DataLoader
import logging

from .data_schema import FACTOR_NAMES, CountryData, get_factor_values, get_mask_values, compute_composite_standing

logger = logging.getLogger(__name__)


class FactorScaler:
    """Scaler that handles missing data and applies transformations."""
    
    def __init__(self, scaler_type: str = "robust", impute_strategy: str = "knn"):
        """
        Initialize scaler.
        
        Args:
            scaler_type: Type of scaler ("standard", "robust")
            impute_strategy: Strategy for imputing missing values ("knn", "mean", "median")
        """
        self.scaler_type = scaler_type
        self.impute_strategy = impute_strategy
        self.scalers = {}
        self.imputers = {}
        self.fitted = False
        
    def fit(self, data: np.ndarray, masks: Optional[np.ndarray] = None) -> 'FactorScaler':
        """
        Fit scalers on training data.
        
        Args:
            data: Array of shape (n_samples, n_factors)
            masks: Array of shape (n_samples, n_factors) indicating missing values
            
        Returns:
            Self for chaining
        """
        if masks is None:
            masks = np.isnan(data)
        
        # Create scaler and imputer for each factor
        for i, factor_name in enumerate(FACTOR_NAMES):
            factor_data = data[:, i]
            factor_mask = masks[:, i]
            
            # Handle missing values
            if self.impute_strategy == "knn":
                imputer = KNNImputer(n_neighbors=5)
            elif self.impute_strategy == "mean":
                imputer = None  # Will use mean imputation
            elif self.impute_strategy == "median":
                imputer = None  # Will use median imputation
            else:
                raise ValueError(f"Unknown impute_strategy: {self.impute_strategy}")
            
            # Impute missing values for fitting
            if np.any(factor_mask):
                if imputer is not None:
                    # KNN imputation
                    factor_data_2d = factor_data.reshape(-1, 1)
                    factor_data_imputed = imputer.fit_transform(factor_data_2d).flatten()
                else:
                    # Simple imputation
                    if self.impute_strategy == "mean":
                        fill_value = np.nanmean(factor_data[~factor_mask])
                    else:  # median
                        fill_value = np.nanmedian(factor_data[~factor_mask])
                    factor_data_imputed = factor_data.copy()
                    factor_data_imputed[factor_mask] = fill_value
            else:
                factor_data_imputed = factor_data.copy()
            
            # Fit scaler
            if self.scaler_type == "standard":
                scaler = StandardScaler()
            elif self.scaler_type == "robust":
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler_type: {self.scaler_type}")
            
            scaler.fit(factor_data_imputed.reshape(-1, 1))
            
            self.scalers[factor_name] = scaler
            self.imputers[factor_name] = imputer
        
        self.fitted = True
        return self
    
    def transform(self, data: np.ndarray, masks: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using fitted scalers.
        
        Args:
            data: Array of shape (n_samples, n_factors)
            masks: Array of shape (n_samples, n_factors) indicating missing values
            
        Returns:
            Tuple of (scaled_data, updated_masks)
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        if masks is None:
            masks = np.isnan(data)
        
        scaled_data = np.zeros_like(data)
        updated_masks = masks.copy()
        
        for i, factor_name in enumerate(FACTOR_NAMES):
            factor_data = data[:, i]
            factor_mask = masks[:, i]
            
            # Impute missing values
            if np.any(factor_mask):
                imputer = self.imputers[factor_name]
                if imputer is not None:
                    # KNN imputation
                    factor_data_2d = factor_data.reshape(-1, 1)
                    factor_data_imputed = imputer.transform(factor_data_2d).flatten()
                else:
                    # Simple imputation using fitted statistics
                    scaler = self.scalers[factor_name]
                    if self.scaler_type == "standard":
                        fill_value = scaler.mean_[0]
                    else:  # robust
                        fill_value = scaler.center_[0]
                    factor_data_imputed = factor_data.copy()
                    factor_data_imputed[factor_mask] = fill_value
            else:
                factor_data_imputed = factor_data.copy()
            
            # Scale
            scaler = self.scalers[factor_name]
            scaled_data[:, i] = scaler.transform(factor_data_imputed.reshape(-1, 1)).flatten()
            
            # Update masks (keep track of originally missing values)
            updated_masks[:, i] = factor_mask
        
        return scaled_data, updated_masks
    
    def inverse_transform(self, scaled_data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data back to original scale."""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        
        original_data = np.zeros_like(scaled_data)
        
        for i, factor_name in enumerate(FACTOR_NAMES):
            scaler = self.scalers[factor_name]
            original_data[:, i] = scaler.inverse_transform(scaled_data[:, i].reshape(-1, 1)).flatten()
        
        return original_data


def create_windows(
    data: np.ndarray,
    masks: np.ndarray,
    window_length: int = 20,
    step: int = 1,
    horizons: List[int] = [1, 5, 10]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding windows from time series data.
    
    Args:
        data: Array of shape (n_timesteps, n_factors)
        masks: Array of shape (n_timesteps, n_factors) indicating missing values
        window_length: Length of input window
        step: Step size between windows
        horizons: List of forecast horizons
        
    Returns:
        Tuple of (windows, window_masks, targets)
    """
    n_timesteps, n_factors = data.shape
    max_horizon = max(horizons)
    
    # Calculate number of windows
    n_windows = (n_timesteps - window_length - max_horizon) // step + 1
    
    if n_windows <= 0:
        raise ValueError(f"Insufficient data: need {window_length + max_horizon} timesteps, got {n_timesteps}")
    
    # Initialize output arrays
    windows = np.zeros((n_windows, window_length, n_factors))
    window_masks = np.zeros((n_windows, window_length, n_factors), dtype=bool)
    targets = np.zeros((n_windows, len(horizons)))
    
    for i in range(n_windows):
        start_idx = i * step
        end_idx = start_idx + window_length
        
        # Input window
        windows[i] = data[start_idx:end_idx]
        window_masks[i] = masks[start_idx:end_idx]
        
        # Targets at different horizons
        for j, horizon in enumerate(horizons):
            target_idx = end_idx + horizon - 1
            if target_idx < n_timesteps:
                # Compute composite standing as target
                factor_values = data[target_idx]
                targets[i, j] = compute_composite_standing(factor_values)
            else:
                targets[i, j] = np.nan
    
    # Remove windows with NaN targets
    valid_mask = ~np.isnan(targets).any(axis=1)
    windows = windows[valid_mask]
    window_masks = window_masks[valid_mask]
    targets = targets[valid_mask]
    
    logger.info(f"Created {len(windows)} windows from {n_timesteps} timesteps")
    
    return windows, window_masks, targets


def windowify_panel(
    panel_data: List[CountryData],
    window_length: int = 20,
    step: int = 1,
    horizons: List[int] = [1, 5, 10]
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create windows for all countries in panel data.
    
    Args:
        panel_data: List of CountryData objects
        window_length: Length of input window
        step: Step size between windows
        horizons: List of forecast horizons
        
    Returns:
        Dictionary mapping country names to (windows, masks, targets)
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame([{
        'country': data.country,
        'year': data.year,
        **{factor: getattr(data, factor) for factor in FACTOR_NAMES}
    } for data in panel_data])
    
    country_windows = {}
    
    for country in df['country'].unique():
        country_data = df[df['country'] == country].sort_values('year')
        
        if len(country_data) < window_length + max(horizons):
            logger.warning(f"Insufficient data for {country}: {len(country_data)} years")
            continue
        
        # Extract factor values
        factor_data = country_data[FACTOR_NAMES].values
        masks = country_data[FACTOR_NAMES].isna().values
        
        try:
            windows, window_masks, targets = create_windows(
                factor_data, masks, window_length, step, horizons
            )
            country_windows[country] = (windows, window_masks, targets)
        except ValueError as e:
            logger.warning(f"Failed to create windows for {country}: {e}")
            continue
    
    logger.info(f"Created windows for {len(country_windows)} countries")
    
    return country_windows


class CountryStandingDataset(Dataset):
    """PyTorch Dataset for country standing forecast data."""
    
    def __init__(
        self,
        windows: np.ndarray,
        masks: np.ndarray,
        targets: np.ndarray,
        device: str = "cpu"
    ):
        """
        Initialize dataset.
        
        Args:
            windows: Array of shape (n_samples, window_length, n_factors)
            masks: Array of shape (n_samples, window_length, n_factors)
            targets: Array of shape (n_samples, n_horizons)
            device: Device to store tensors on
        """
        self.windows = torch.tensor(windows, dtype=torch.float32, device=device)
        self.masks = torch.tensor(masks, dtype=torch.bool, device=device)
        self.targets = torch.tensor(targets, dtype=torch.float32, device=device)
        
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.masks[idx], self.targets[idx]


def create_data_loaders(
    country_windows: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    train_countries: List[str],
    val_countries: List[str],
    test_countries: List[str],
    batch_size: int = 64,
    device: str = "cpu"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test data loaders.
    
    Args:
        country_windows: Dictionary mapping countries to (windows, masks, targets)
        train_countries: List of training countries
        val_countries: List of validation countries
        test_countries: List of test countries
        batch_size: Batch size for data loaders
        device: Device to store tensors on
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    def combine_country_data(countries: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Combine data from multiple countries."""
        all_windows = []
        all_masks = []
        all_targets = []
        
        for country in countries:
            if country in country_windows:
                windows, masks, targets = country_windows[country]
                all_windows.append(windows)
                all_masks.append(masks)
                all_targets.append(targets)
        
        if not all_windows:
            return np.array([]), np.array([]), np.array([])
        
        return (
            np.vstack(all_windows),
            np.vstack(all_masks),
            np.vstack(all_targets)
        )
    
    # Combine data for each split
    train_data = combine_country_data(train_countries)
    val_data = combine_country_data(val_countries)
    test_data = combine_country_data(test_countries)
    
    # Create datasets
    train_dataset = CountryStandingDataset(*train_data, device=device) if len(train_data[0]) > 0 else None
    val_dataset = CountryStandingDataset(*val_data, device=device) if len(val_data[0]) > 0 else None
    test_dataset = CountryStandingDataset(*test_data, device=device) if len(test_data[0]) > 0 else None
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if train_dataset else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None
    
    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_dataset) if train_dataset else 0} samples")
    logger.info(f"  Val: {len(val_dataset) if val_dataset else 0} samples")
    logger.info(f"  Test: {len(test_dataset) if test_dataset else 0} samples")
    
    return train_loader, val_loader, test_loader


def apply_transforms(data: np.ndarray, transform_type: str) -> np.ndarray:
    """
    Apply transformations to factor data.
    
    Args:
        data: Array of shape (n_timesteps, n_factors)
        transform_type: Type of transformation ("level", "yoy", "log", "log_yoy")
        
    Returns:
        Transformed data
    """
    if transform_type == "level":
        return data
    elif transform_type == "yoy":
        # Year-over-year change
        return np.diff(data, axis=0, prepend=data[0:1])
    elif transform_type == "log":
        # Log transformation (add small constant to avoid log(0))
        return np.log(data + 1e-8)
    elif transform_type == "log_yoy":
        # Log year-over-year change
        log_data = np.log(data + 1e-8)
        return np.diff(log_data, axis=0, prepend=log_data[0:1])
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")


def create_feature_summary(windows: np.ndarray, masks: np.ndarray) -> Dict:
    """Create summary statistics for feature data."""
    summary = {
        'n_samples': len(windows),
        'window_length': windows.shape[1],
        'n_factors': windows.shape[2],
        'missing_data_pct': masks.mean(),
        'factor_stats': {}
    }
    
    for i, factor_name in enumerate(FACTOR_NAMES):
        factor_data = windows[:, :, i]
        factor_mask = masks[:, :, i]
        
        summary['factor_stats'][factor_name] = {
            'mean': np.nanmean(factor_data),
            'std': np.nanstd(factor_data),
            'min': np.nanmin(factor_data),
            'max': np.nanmax(factor_data),
            'missing_pct': factor_mask.mean()
        }
    
    return summary
