"""
Training utilities for country standing forecast model.

Implements rolling-origin walk-forward training with LOCO (Leave-One-Country-Out)
support and early stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import yaml
from tqdm import tqdm
import pickle

from .data_schema import CountryData, FACTOR_NAMES, FORECAST_HORIZONS
from .features import FactorScaler, windowify_panel, create_data_loaders
from .models.forecast_net import CountryStandingForecastNet, create_model
from .metrics import ForecastMetrics, quantile_loss

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        metric: str = "val_mase_5y",
        mode: str = "min"
    ):
        """
        Initialize EarlyStopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            metric: Metric to monitor
            mode: "min" or "max"
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        if mode == "min":
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class ForecastTrainer:
    """Main trainer class for country standing forecasting."""
    
    def __init__(
        self,
        model: CountryStandingForecastNet,
        device: str = "cpu",
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-5,
        use_quantiles: bool = True,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ):
        """
        Initialize ForecastTrainer.
        
        Args:
            model: Neural network model
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            use_quantiles: Whether to use quantile regression
            quantiles: List of quantiles for uncertainty estimation
        """
        self.model = model.to(device)
        self.device = device
        self.use_quantiles = use_quantiles
        self.quantiles = quantiles
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        
        # Metrics calculator
        self.metrics_calculator = ForecastMetrics(
            horizons=model.horizons,
            quantiles=quantiles if use_quantiles else None
        )
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": []
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        scaler: FactorScaler
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            scaler: Fitted scaler for inverse transform
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for windows, masks, targets in train_loader:
            windows = windows.to(self.device)
            masks = masks.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            predictions = self.model(windows, masks)
            
            # Calculate loss
            loss = 0.0
            
            # Point forecast loss
            for horizon in self.model.horizons:
                horizon_key = f"standing_{horizon}y"
                if horizon_key in predictions:
                    pred = predictions[horizon_key]
                    true = targets[:, horizon - 1]  # Convert to 0-indexed
                    loss += self.mse_loss(pred, true)
            
            # Quantile loss (optional)
            if self.use_quantiles:
                for horizon in self.model.horizons:
                    quantile_key = f"quantiles_{horizon}y"
                    if quantile_key in predictions:
                        quantile_pred = predictions[quantile_key]
                        true = targets[:, horizon - 1]
                        loss += quantile_loss(true, quantile_pred, self.quantiles)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def validate(
        self,
        val_loader: DataLoader,
        scaler: FactorScaler
    ) -> Tuple[float, Dict[str, Dict[str, float]]]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            scaler: Fitted scaler for inverse transform
            
        Returns:
            Tuple of (validation_loss, metrics)
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        all_predictions = {}
        all_targets = {}
        
        with torch.no_grad():
            for windows, masks, targets in val_loader:
                windows = windows.to(self.device)
                masks = masks.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = self.model(windows, masks)
                
                # Calculate loss
                loss = 0.0
                
                # Point forecast loss
                for horizon in self.model.horizons:
                    horizon_key = f"standing_{horizon}y"
                    if horizon_key in predictions:
                        pred = predictions[horizon_key]
                        true = targets[:, horizon - 1]
                        loss += self.mse_loss(pred, true)
                        
                        # Store for metrics calculation
                        if horizon_key not in all_predictions:
                            all_predictions[horizon_key] = []
                            all_targets[horizon_key] = []
                        
                        all_predictions[horizon_key].append(pred.cpu().numpy())
                        all_targets[horizon_key].append(true.cpu().numpy())
                
                # Quantile loss (optional)
                if self.use_quantiles:
                    for horizon in self.model.horizons:
                        quantile_key = f"quantiles_{horizon}y"
                        if quantile_key in predictions:
                            quantile_pred = predictions[quantile_key]
                            true = targets[:, horizon - 1]
                            loss += quantile_loss(true, quantile_pred, self.quantiles)
                            
                            # Store quantile predictions
                            if quantile_key not in all_predictions:
                                all_predictions[quantile_key] = []
                            all_predictions[quantile_key].append(quantile_pred.cpu().numpy())
                
                total_loss += loss.item()
                n_batches += 1
        
        # Concatenate predictions and targets
        for key in all_predictions:
            all_predictions[key] = np.concatenate(all_predictions[key])
        for key in all_targets:
            all_targets[key] = np.concatenate(all_targets[key])
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(all_targets, all_predictions)
        
        val_loss = total_loss / n_batches if n_batches > 0 else 0.0
        
        return val_loss, metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scaler: FactorScaler,
        epochs: int = 100,
        early_stopping: Optional[EarlyStopping] = None,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            scaler: Fitted scaler
            epochs: Maximum number of epochs
            early_stopping: Early stopping utility
            save_path: Path to save best model
            
        Returns:
            Training history and best metrics
        """
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Model parameters: {self.model.count_parameters():,}")
        
        best_val_loss = float('inf')
        best_model_state = None
        best_metrics = None
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader, scaler)
            
            # Validation
            val_loss, val_metrics = self.validate(val_loader, scaler)
            
            # Store history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_metrics"].append(val_metrics)
            
            # Check for best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                best_metrics = val_metrics
            
            # Save model if specified
            if save_path and val_loss < best_val_loss:
                self.save_model(save_path, scaler)
            
            # Log progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(
                    f"Epoch {epoch:3d}: "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}"
                )
                
                # Log key metrics
                for horizon_key, metrics in val_metrics.items():
                    if "mae" in metrics:
                        logger.info(f"  {horizon_key} MAE: {metrics['mae']:.4f}")
            
            # Early stopping
            if early_stopping:
                # Use MASE for 5-year horizon as early stopping metric
                stop_metric = None
                for horizon_key, metrics in val_metrics.items():
                    if "5y" in horizon_key and "mase" in metrics:
                        stop_metric = metrics["mase"]
                        break
                
                if stop_metric is not None and early_stopping(stop_metric):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Loaded best model state")
        
        return {
            "history": self.history,
            "best_val_loss": best_val_loss,
            "best_metrics": best_metrics
        }
    
    def save_model(self, path: Path, scaler: FactorScaler):
        """Save model and scaler."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model.get_model_summary(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history
        }, path / "model.pt")
        
        with open(path / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path) -> FactorScaler:
        """Load model and scaler."""
        checkpoint = torch.load(path / "model.pt", map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        
        with open(path / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        logger.info(f"Model loaded from {path}")
        return scaler


def rolling_origin_training(
    panel_data: List[CountryData],
    config: Dict[str, Any],
    train_years: List[int],
    val_years: List[int],
    test_years: List[int],
    window_length: int = 20,
    step: int = 1,
    device: str = "cpu",
    save_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Perform rolling-origin walk-forward training.
    
    Args:
        panel_data: List of CountryData objects
        config: Model configuration
        train_years: Years to use for training
        val_years: Years to use for validation
        test_years: Years to use for testing
        window_length: Length of input window
        step: Step size between windows
        device: Device to train on
        save_dir: Directory to save results
        
    Returns:
        Training results and metrics
    """
    logger.info("Starting rolling-origin training")
    
    # Create windows for all countries
    country_windows = windowify_panel(panel_data, window_length, step)
    
    if not country_windows:
        raise ValueError("No valid country windows created")
    
    # Split countries
    all_countries = list(country_windows.keys())
    n_countries = len(all_countries)
    
    # Use 70% for training, 15% for validation, 15% for testing
    n_train = int(0.7 * n_countries)
    n_val = int(0.15 * n_countries)
    
    train_countries = all_countries[:n_train]
    val_countries = all_countries[n_train:n_train + n_val]
    test_countries = all_countries[n_train + n_val:]
    
    logger.info(f"Country splits: {len(train_countries)} train, {len(val_countries)} val, {len(test_countries)} test")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        country_windows, train_countries, val_countries, test_countries,
        batch_size=config.get("batch_size", 64), device=device
    )
    
    if train_loader is None or val_loader is None:
        raise ValueError("Failed to create data loaders")
    
    # Fit scaler on training data
    scaler = FactorScaler(
        scaler_type=config.get("scaler_type", "robust"),
        impute_strategy=config.get("impute_strategy", "knn")
    )
    
    # Extract training data for scaler fitting
    train_data = []
    train_masks = []
    for country in train_countries:
        if country in country_windows:
            windows, masks, _ = country_windows[country]
            train_data.append(windows.reshape(-1, windows.shape[-1]))
            train_masks.append(masks.reshape(-1, masks.shape[-1]))
    
    if train_data:
        train_data = np.vstack(train_data)
        train_masks = np.vstack(train_masks)
        scaler.fit(train_data, train_masks)
    
    # Create model
    model = create_model(config, device)
    
    # Create trainer
    trainer = ForecastTrainer(
        model=model,
        device=device,
        learning_rate=config.get("learning_rate", 3e-3),
        weight_decay=config.get("weight_decay", 1e-5),
        use_quantiles=config.get("use_quantiles", True),
        quantiles=config.get("quantiles", [0.1, 0.5, 0.9])
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get("patience", 10),
        min_delta=config.get("min_delta", 0.001),
        metric="val_mase_5y",
        mode="min"
    )
    
    # Train model
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        scaler=scaler,
        epochs=config.get("epochs", 100),
        early_stopping=early_stopping,
        save_path=save_dir
    )
    
    # Evaluate on test set
    if test_loader is not None:
        test_loss, test_metrics = trainer.validate(test_loader, scaler)
        training_results["test_loss"] = test_loss
        training_results["test_metrics"] = test_metrics
    
    # Add country information
    training_results["country_splits"] = {
        "train": train_countries,
        "val": val_countries,
        "test": test_countries
    }
    
    return training_results


def loco_training(
    panel_data: List[CountryData],
    config: Dict[str, Any],
    window_length: int = 20,
    step: int = 1,
    device: str = "cpu",
    save_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Perform Leave-One-Country-Out (LOCO) training.
    
    Args:
        panel_data: List of CountryData objects
        config: Model configuration
        window_length: Length of input window
        step: Step size between windows
        device: Device to train on
        save_dir: Directory to save results
        
    Returns:
        LOCO results and metrics
    """
    logger.info("Starting LOCO training")
    
    # Create windows for all countries
    country_windows = windowify_panel(panel_data, window_length, step)
    
    if not country_windows:
        raise ValueError("No valid country windows created")
    
    all_countries = list(country_windows.keys())
    logger.info(f"LOCO training on {len(all_countries)} countries")
    
    loco_results = {}
    
    for test_country in tqdm(all_countries, desc="LOCO Training"):
        logger.info(f"Training with {test_country} held out")
        
        # Split countries
        train_countries = [c for c in all_countries if c != test_country]
        
        # Use 80% of remaining countries for training, 20% for validation
        n_train = int(0.8 * len(train_countries))
        train_countries_split = train_countries[:n_train]
        val_countries_split = train_countries[n_train:]
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            country_windows, train_countries_split, val_countries_split, [test_country],
            batch_size=config.get("batch_size", 64), device=device
        )
        
        if train_loader is None or val_loader is None:
            logger.warning(f"Skipping {test_country} - insufficient data")
            continue
        
        # Fit scaler on training data
        scaler = FactorScaler(
            scaler_type=config.get("scaler_type", "robust"),
            impute_strategy=config.get("impute_strategy", "knn")
        )
        
        # Extract training data for scaler fitting
        train_data = []
        train_masks = []
        for country in train_countries_split:
            if country in country_windows:
                windows, masks, _ = country_windows[country]
                train_data.append(windows.reshape(-1, windows.shape[-1]))
                train_masks.append(masks.reshape(-1, masks.shape[-1]))
        
        if train_data:
            train_data = np.vstack(train_data)
            train_masks = np.vstack(train_masks)
            scaler.fit(train_data, train_masks)
        
        # Create model
        model = create_model(config, device)
        
        # Create trainer
        trainer = ForecastTrainer(
            model=model,
            device=device,
            learning_rate=config.get("learning_rate", 3e-3),
            weight_decay=config.get("weight_decay", 1e-5),
            use_quantiles=config.get("use_quantiles", True),
            quantiles=config.get("quantiles", [0.1, 0.5, 0.9])
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=config.get("patience", 10),
            min_delta=config.get("min_delta", 0.001),
            metric="val_mase_5y",
            mode="min"
        )
        
        # Train model
        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            scaler=scaler,
            epochs=config.get("epochs", 100),
            early_stopping=early_stopping
        )
        
        # Evaluate on held-out country
        if test_loader is not None:
            test_loss, test_metrics = trainer.validate(test_loader, scaler)
            training_results["test_loss"] = test_loss
            training_results["test_metrics"] = test_metrics
        
        loco_results[test_country] = training_results
        
        # Save individual model if specified
        if save_dir:
            country_save_dir = save_dir / f"loco_{test_country}"
            trainer.save_model(country_save_dir, scaler)
    
    return loco_results


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
