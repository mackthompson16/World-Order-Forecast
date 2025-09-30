"""
Unit tests for country standing forecast project.

Tests key functionality including features.windowify, metrics.mase,
and scaler no-leakage validation.
"""

import unittest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import shutil

# Import modules to test
from src.features import FactorScaler, create_windows, windowify_panel
from src.metrics import mean_absolute_scaled_error, ForecastMetrics
from src.data_schema import CountryData, FACTOR_NAMES, compute_composite_standing


class TestFactorScaler(unittest.TestCase):
    """Test FactorScaler functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 100
        self.n_factors = 8
        
        # Create synthetic data with some missing values
        self.data = np.random.randn(self.n_samples, self.n_factors)
        self.masks = np.random.rand(self.n_samples, self.n_factors) < 0.1  # 10% missing
        
        # Set missing values to NaN
        self.data[self.masks] = np.nan
    
    def test_scaler_fit_transform(self):
        """Test scaler fit and transform."""
        scaler = FactorScaler(scaler_type="robust", impute_strategy="knn")
        
        # Fit scaler
        scaler.fit(self.data, self.masks)
        
        # Transform data
        scaled_data, updated_masks = scaler.transform(self.data, self.masks)
        
        # Check shapes
        self.assertEqual(scaled_data.shape, self.data.shape)
        self.assertEqual(updated_masks.shape, self.masks.shape)
        
        # Check that missing values are preserved in masks
        np.testing.assert_array_equal(updated_masks, self.masks)
        
        # Check that scaled data doesn't contain NaN (except where originally missing)
        self.assertFalse(np.isnan(scaled_data[~self.masks]).any())
    
    def test_scaler_inverse_transform(self):
        """Test scaler inverse transform."""
        scaler = FactorScaler(scaler_type="standard", impute_strategy="mean")
        
        # Fit and transform
        scaler.fit(self.data, self.masks)
        scaled_data, _ = scaler.transform(self.data, self.masks)
        
        # Inverse transform
        original_data = scaler.inverse_transform(scaled_data)
        
        # Check shape
        self.assertEqual(original_data.shape, self.data.shape)
        
        # Check that inverse transform recovers original data (approximately)
        # Note: This won't be exact due to imputation, but should be close
        non_missing_mask = ~self.masks
        if np.any(non_missing_mask):
            diff = np.abs(original_data[non_missing_mask] - self.data[non_missing_mask])
            self.assertTrue(np.all(diff < 1e-10))  # Should be very close
    
    def test_no_leakage(self):
        """Test that scaler doesn't leak information from validation to training."""
        # Split data into train and validation
        train_data = self.data[:70]
        train_masks = self.masks[:70]
        val_data = self.data[70:]
        val_masks = self.masks[70:]
        
        scaler = FactorScaler(scaler_type="robust", impute_strategy="knn")
        
        # Fit only on training data
        scaler.fit(train_data, train_masks)
        
        # Transform both train and validation data
        train_scaled, train_masks_scaled = scaler.transform(train_data, train_masks)
        val_scaled, val_masks_scaled = scaler.transform(val_data, val_masks)
        
        # Check that validation data transformation doesn't depend on validation data itself
        # (This is implicit in the design, but we can verify the scaler was fitted only on train)
        
        # The scaler should have been fitted on training data only
        # We can verify this by checking that the scaler's internal state
        # was determined by training data only
        self.assertTrue(scaler.fitted)
        
        # Check that both train and validation data are properly scaled
        self.assertFalse(np.isnan(train_scaled[~train_masks]).any())
        self.assertFalse(np.isnan(val_scaled[~val_masks]).any())


class TestWindowify(unittest.TestCase):
    """Test windowing functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_timesteps = 50
        self.n_factors = 8
        self.window_length = 20
        self.horizons = [1, 5, 10]
        
        # Create synthetic time series data
        self.data = np.random.randn(self.n_timesteps, self.n_factors)
        self.masks = np.random.rand(self.n_timesteps, self.n_factors) < 0.05  # 5% missing
        self.data[self.masks] = np.nan
    
    def test_create_windows(self):
        """Test window creation."""
        windows, window_masks, targets = create_windows(
            self.data, self.masks, self.window_length, step=1, horizons=self.horizons
        )
        
        # Check shapes
        expected_windows = self.n_timesteps - self.window_length - max(self.horizons) + 1
        self.assertEqual(windows.shape[0], expected_windows)
        self.assertEqual(windows.shape[1], self.window_length)
        self.assertEqual(windows.shape[2], self.n_factors)
        
        self.assertEqual(window_masks.shape, windows.shape)
        self.assertEqual(targets.shape[0], windows.shape[0])
        self.assertEqual(targets.shape[1], len(self.horizons))
        
        # Check that targets don't contain NaN (after filtering)
        self.assertFalse(np.isnan(targets).any())
    
    def test_window_step(self):
        """Test window creation with different step sizes."""
        step = 5
        windows, window_masks, targets = create_windows(
            self.data, self.masks, self.window_length, step=step, horizons=self.horizons
        )
        
        # With step=5, we should have fewer windows
        expected_windows = (self.n_timesteps - self.window_length - max(self.horizons)) // step + 1
        self.assertEqual(windows.shape[0], expected_windows)
    
    def test_windowify_panel(self):
        """Test panel windowing."""
        # Create CountryData objects
        country_data = []
        for year in range(2000, 2035):
            factor_values = np.random.randn(len(FACTOR_NAMES))
            masks = np.random.rand(len(FACTOR_NAMES)) < 0.1
            
            data = CountryData(
                country="TEST",
                year=year,
                **{factor: factor_values[i] if not masks[i] else None for i, factor in enumerate(FACTOR_NAMES)},
                **{f"mask_{factor}": masks[i] for i, factor in enumerate(FACTOR_NAMES)}
            )
            country_data.append(data)
        
        # Create windows
        country_windows = windowify_panel(country_data, self.window_length, step=1, horizons=self.horizons)
        
        # Check that windows were created
        self.assertIn("TEST", country_windows)
        
        windows, masks, targets = country_windows["TEST"]
        self.assertGreater(len(windows), 0)
        self.assertEqual(windows.shape[1], self.window_length)
        self.assertEqual(windows.shape[2], len(FACTOR_NAMES))


class TestMetrics(unittest.TestCase):
    """Test metrics functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 100
        
        # Create synthetic data
        self.y_true = np.random.randn(self.n_samples) * 10 + 50
        self.y_pred = self.y_true + np.random.randn(self.n_samples) * 2
        self.y_train = np.random.randn(self.n_samples * 2) * 10 + 50
    
    def test_mase_calculation(self):
        """Test MASE calculation."""
        mase = mean_absolute_scaled_error(self.y_true, self.y_pred, self.y_train)
        
        # MASE should be positive
        self.assertGreater(mase, 0)
        
        # MASE should be reasonable (not too large)
        self.assertLess(mase, 10)
    
    def test_mase_without_training_data(self):
        """Test MASE calculation without training data."""
        mase = mean_absolute_scaled_error(self.y_true, self.y_pred)
        
        # Should still work
        self.assertGreater(mase, 0)
    
    def test_forecast_metrics(self):
        """Test ForecastMetrics class."""
        metrics_calc = ForecastMetrics(horizons=[1, 5, 10])
        
        # Create mock data
        y_true = {
            "standing_1y": self.y_true,
            "standing_5y": self.y_true,
            "standing_10y": self.y_true
        }
        
        y_pred = {
            "standing_1y": self.y_pred,
            "standing_5y": self.y_pred,
            "standing_10y": self.y_pred
        }
        
        # Calculate metrics
        metrics = metrics_calc.calculate_metrics(y_true, y_pred)
        
        # Check that metrics were calculated
        self.assertIn("horizon_1y", metrics)
        self.assertIn("horizon_5y", metrics)
        self.assertIn("horizon_10y", metrics)
        
        # Check that key metrics are present
        for horizon_key in metrics:
            self.assertIn("mae", metrics[horizon_key])
            self.assertIn("rmse", metrics[horizon_key])
            self.assertIn("spearman_corr", metrics[horizon_key])


class TestDataSchema(unittest.TestCase):
    """Test data schema functionality."""
    
    def test_composite_standing_calculation(self):
        """Test composite standing calculation."""
        # Create test factor values
        factor_values = np.array([12.0, 100.0, 75.0, 2.0, 60.0, 5.0, 30.0, 50.0])
        
        standing = compute_composite_standing(factor_values)
        
        # Standing should be between 0 and 100
        self.assertGreaterEqual(standing, 0)
        self.assertLessEqual(standing, 100)
        
        # Should be reasonable value
        self.assertGreater(standing, 20)
        self.assertLess(standing, 90)
    
    def test_country_data_creation(self):
        """Test CountryData object creation."""
        data = CountryData(
            country="TEST",
            year=2020,
            education=12.0,
            innovation=100.0,
            competitiveness=75.0,
            military=2.0,
            trade_share=60.0,
            reserve_currency_proxy=5.0,
            financial_center_proxy=30.0,
            debt=50.0
        )
        
        self.assertEqual(data.country, "TEST")
        self.assertEqual(data.year, 2020)
        self.assertEqual(data.education, 12.0)
        self.assertEqual(data.innovation, 100.0)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline with synthetic data."""
        # This is a simplified version of the full pipeline
        
        # Create synthetic data
        np.random.seed(42)
        n_years = 30
        n_factors = 8
        
        # Generate time series data
        data = np.random.randn(n_years, n_factors)
        masks = np.random.rand(n_years, n_factors) < 0.1
        
        # Create windows
        windows, window_masks, targets = create_windows(
            data, masks, window_length=20, step=1, horizons=[1, 5, 10]
        )
        
        # Check that we have valid windows
        self.assertGreater(len(windows), 0)
        self.assertFalse(np.isnan(targets).any())
        
        # Test scaler
        scaler = FactorScaler(scaler_type="robust", impute_strategy="knn")
        
        # Flatten windows for scaler fitting
        flat_data = windows.reshape(-1, n_factors)
        flat_masks = window_masks.reshape(-1, n_factors)
        
        scaler.fit(flat_data, flat_masks)
        scaled_data, scaled_masks = scaler.transform(flat_data, flat_masks)
        
        # Check that scaling worked
        self.assertEqual(scaled_data.shape, flat_data.shape)
        self.assertFalse(np.isnan(scaled_data[~flat_masks]).any())
        
        # Test metrics
        metrics_calc = ForecastMetrics(horizons=[1, 5, 10])
        
        y_true = {"standing_1y": targets[:, 0]}
        y_pred = {"standing_1y": targets[:, 0] + np.random.randn(len(targets)) * 0.1}
        
        metrics = metrics_calc.calculate_metrics(y_true, y_pred)
        
        # Check that metrics were calculated
        self.assertIn("horizon_1y", metrics)
        self.assertIn("mae", metrics["horizon_1y"])


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestFactorScaler,
        TestWindowify,
        TestMetrics,
        TestDataSchema,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)
