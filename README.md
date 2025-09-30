# World Order Forecast

A machine learning approach to forecasting a country's "standing" using 8 metrics inspired by Ray Dalio's framework for measuring national strength.

🌐 **[Live Demo & Documentation](https://mackthompson16.github.io/World-Order-Forecast)**

## Overview

This project implements a neural network-based forecasting system that predicts a country's composite standing score at multiple horizons (1, 5, and 10 years) using time series data of macro factors. The approach combines factor-wise convolution with temporal convolutional networks (TCN) to capture both cross-factor interactions and temporal dependencies.

## Key Features

- **8 Macro Factors**: Education, Innovation, Competitiveness, Military, Trade Share, Reserve Currency, Financial Center, and Debt
- **Neural Network Architecture**: Factor convolution + Temporal TCN with attention mechanism
- **Multiple Forecast Horizons**: 1, 5, and 10-year predictions
- **Uncertainty Quantification**: Quantile regression for confidence intervals
- **Robust Evaluation**: Leave-One-Country-Out (LOCO) cross-validation
- **Interactive UI**: Streamlit web application for exploration and what-if analysis
- **Synthetic Data**: Plausible synthetic data generator for immediate testing

## Project Structure

```
country_standing_forecast/
├── src/                    # Core Python modules
├── configs/               # Model configurations
├── notebooks/             # Jupyter tutorials
├── tests/                 # Unit tests
├── data/                  # Data directory
├── results/               # Results and models
├── docs/                  # React documentation app
├── make_synth_data.py     # Synthetic data generator
└── requirements.txt       # Python dependencies
```

## Data Schema

The system uses 8 macro factors to measure country standing:

| Factor | Description | Unit | Higher is Better | Source |
|--------|-------------|------|------------------|---------|
| Education | Average years of schooling (population 25+) | years | Yes | World Bank WDI |
| Innovation | Patent applications per million population | patents/million | Yes | WIPO |
| Competitiveness | Global Competitiveness Index | index (0-100) | Yes | WEF |
| Military | Military expenditure as % of GDP | % of GDP | Yes | SIPRI |
| Trade Share | Trade (exports + imports) as % of GDP | % of GDP | Yes | World Bank WDI |
| Reserve Currency | Currency share in global foreign exchange reserves | % of global reserves | Yes | IMF COFER |
| Financial Center | Financial center development index | index (0-100) | Yes | GFCI |
| Debt | Government debt as % of GDP | % of GDP | No (inverted) | World Bank WDI |

## Model Architecture

- **Input**: `[batch_size, 20, 8]` (20 years × 8 factors)
- **Factor Convolution**: 1D conv across factors (kernel 2-3) → captures factor interactions
- **Temporal TCN**: Dilated convolutions [1,2,4,8] → captures temporal dependencies
- **Attention**: Temporal attention for focusing on critical periods
- **Output**: `[batch_size, 3]` (1, 5, 10-year forecasts) + uncertainty quantiles
- **Parameters**: ~200k (lightweight design)

## Training & Evaluation

- **Rolling-Origin Walk-Forward**: Time-based validation with no future leakage
- **Leave-One-Country-Out**: Tests generalization across countries
- **Early Stopping**: Prevents overfitting on validation MASE
- **Metrics**: MAE, RMSE, MASE, Spearman correlation, interval coverage

## Acknowledgments

- Inspired by Ray Dalio's framework for measuring country strength
- Built with PyTorch, Streamlit, and scikit-learn
- Uses synthetic data for immediate testing and demonstration
