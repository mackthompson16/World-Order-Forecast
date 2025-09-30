# Country Standing Forecast

A machine learning approach to forecasting a country's "standing" using 8 macro factors inspired by Ray Dalio's framework for measuring national strength.

## Overview

This project implements a neural network-based forecasting system that predicts a country's composite standing score at multiple horizons (1, 5, and 10 years) using time series data of macro factors. The approach combines factor-wise convolution with temporal convolutional networks (TCN) to capture both cross-factor interactions and temporal dependencies.

## Architectural Justification

### Why This Architecture for Country Standing Prediction?

The choice of neural network architecture is driven by the unique challenges of predicting country futures using macro factors:

#### **1. Factor-Wise Convolution: Capturing Macro Factor Interactions**

**Problem**: The 8 macro factors don't exist in isolation. Education affects innovation capacity, military spending influences trade relationships, and financial center development impacts reserve currency status. Traditional approaches treat factors independently.

**Solution**: Factor-wise 1D convolution across the factor dimension at each time step.

**Why Convolution?**
- **Local Interactions**: Kernel size 2-3 captures immediate factor relationships (e.g., education ↔ innovation)
- **Translation Invariance**: Same relationships apply regardless of factor ordering
- **Parameter Efficiency**: Shared weights across time steps reduce overfitting
- **Interpretability**: Convolution patterns can reveal which factor combinations matter most

**Alternative Considered**: Fully connected layers would require O(n²) parameters and lack spatial structure.

#### **2. Temporal Convolutional Network (TCN): Long-Range Dependencies**

**Problem**: Country standing evolves over decades. A policy change in education today affects competitiveness 10 years later. Traditional RNNs suffer from vanishing gradients and sequential processing bottlenecks.

**Solution**: Dilated temporal convolutions with dilations [1, 2, 4, 8].

**Why TCN over RNNs/LSTMs?**
- **Parallel Processing**: All time steps processed simultaneously (faster training)
- **Long Memory**: Dilated convolutions capture dependencies across 20+ years
- **Stable Gradients**: No vanishing gradient problem
- **Causal Structure**: Only past information influences predictions (no future leakage)

**Why These Dilations?**
- **Multi-Scale Patterns**: Captures both short-term (1-2 year) and long-term (8+ year) cycles
- **Exponential Growth**: Each dilation doubles the receptive field efficiently
- **Economic Cycles**: Aligns with typical business/political cycles (2-4 years) and structural changes (8+ years)

#### **3. Attention Mechanism: Focusing on Critical Periods**

**Problem**: Not all historical periods are equally relevant. Crisis years (2008, 2020) may be more predictive than stable periods.

**Solution**: Temporal attention over the TCN output.

**Why Attention?**
- **Adaptive Weighting**: Model learns which time periods are most predictive
- **Crisis Sensitivity**: Can focus on economic shocks, wars, or policy changes
- **Interpretability**: Attention weights reveal which historical periods matter most
- **Flexibility**: Different countries may have different critical periods

#### **4. Quantile Regression: Uncertainty Quantification**

**Problem**: Country futures are inherently uncertain. Point predictions are insufficient for policy decisions.

**Solution**: Quantile regression at [0.1, 0.5, 0.9] quantiles.

**Why Quantiles?**
- **Policy Planning**: Decision-makers need confidence intervals, not just best guesses
- **Risk Assessment**: Lower quantiles show worst-case scenarios
- **Robust Predictions**: Median (0.5) is more robust than mean for skewed distributions
- **Regulatory Compliance**: Many applications require uncertainty estimates

#### **5. Multi-Horizon Forecasting: Different Time Scales**

**Problem**: Different policy decisions require different time horizons. Short-term tactical decisions vs. long-term strategic planning.

**Solution**: Separate prediction heads for 1, 5, and 10-year horizons.

**Why Multiple Horizons?**
- **Policy Relevance**: Different stakeholders need different time scales
- **Model Specialization**: Each horizon can learn different patterns
- **Validation**: Can assess model performance across time scales
- **Uncertainty Scaling**: Longer horizons naturally have higher uncertainty

#### **6. Leave-One-Country-Out (LOCO): Generalization Testing**

**Problem**: Countries are unique. A model trained on developed countries may not generalize to emerging markets.

**Solution**: LOCO cross-validation to test generalization across countries.

**Why LOCO?**
- **Realistic Evaluation**: Tests if model works for unseen countries
- **Bias Detection**: Reveals if model is biased toward certain country types
- **Robustness**: Ensures model doesn't memorize country-specific patterns
- **Policy Applicability**: Validates model for new country applications

### **Architectural Trade-offs and Limitations**

#### **Current Limitations:**

1. **Static Factor Set**: Fixed 8 factors may miss emerging indicators (e.g., digital infrastructure, climate resilience)
2. **Linear Interactions**: Convolution captures local interactions but may miss complex non-linear relationships
3. **No External Shocks**: Model doesn't explicitly account for global events (pandemics, wars, technological disruptions)
4. **Homogeneous Treatment**: All countries treated equally despite different development stages

#### **Potential Improvements:**

1. **Dynamic Factor Selection**: Attention-based factor weighting or learned factor importance
2. **Hierarchical Modeling**: Different models for different country groups (developed vs. emerging)
3. **External Event Integration**: Incorporate global indicators (oil prices, geopolitical indices)
4. **Causal Discovery**: Use causal inference to identify true causal relationships vs. correlations
5. **Ensemble Methods**: Combine multiple architectures (CNN + Transformer + Graph Neural Networks)

### **Why Not Alternative Architectures?**

#### **Transformers**: 
- **Pros**: Excellent at capturing long-range dependencies
- **Cons**: Require much more data, computationally expensive, less interpretable for this problem size

#### **Graph Neural Networks**:
- **Pros**: Could model country relationships and dependencies
- **Cons**: Requires country relationship data, adds complexity, unclear if country relationships are stable enough

#### **Traditional Econometric Models**:
- **Pros**: Highly interpretable, well-established
- **Cons**: Assume linear relationships, struggle with non-stationarity, limited to short horizons

#### **Random Forests/XGBoost**:
- **Pros**: Robust, interpretable, good with missing data
- **Cons**: Don't capture temporal dependencies well, struggle with long sequences

### **Data Architecture Choices**

#### **Robust Scaling over Standard Scaling**:
- **Why**: Macro factors have outliers (e.g., military spending spikes during wars)
- **Benefit**: More robust to extreme values, better for skewed distributions

#### **KNN Imputation over Mean/Median**:
- **Why**: Missing data patterns may be informative (e.g., countries with missing innovation data may be less innovative)
- **Benefit**: Preserves data structure, more sophisticated than simple imputation

#### **20-Year Windows**:
- **Why**: Balances sufficient history with computational efficiency
- **Benefit**: Captures multiple business cycles while keeping model manageable

This architecture represents a careful balance between **model sophistication** and **practical constraints**, optimized specifically for the challenge of predicting country futures using macro factors.

## Architectural Analysis: Improving Country Future Prediction

### **Critical Questions for Country Future Prediction**

Given the primary goal of predicting country futures using 8 macro factors, several architectural decisions warrant deeper analysis:

#### **1. Factor Interaction Modeling: Are We Missing Key Relationships?**

**Current Approach**: Factor-wise convolution with kernel size 2-3 captures local factor interactions.

**Critical Question**: Do the 8 factors interact in more complex ways than local convolutions can capture?

**Potential Issues**:
- **Non-local Dependencies**: Education may affect innovation through multiple pathways (R&D spending, talent pool, institutional quality)
- **Threshold Effects**: Military spending may have diminishing returns beyond certain levels
- **Regime Changes**: Factor relationships may change during political transitions or economic crises

**Proposed Improvements**:
```python
# Multi-scale factor interaction network
class HierarchicalFactorConv(nn.Module):
    def __init__(self):
        self.local_conv = nn.Conv1d(1, 16, kernel_size=3)  # Local interactions
        self.global_conv = nn.Conv1d(1, 16, kernel_size=8)  # Global interactions
        self.attention = nn.MultiheadAttention(8, num_heads=2)  # Factor attention
```

#### **2. Temporal Dynamics: Are We Capturing the Right Time Scales?**

**Current Approach**: TCN with dilations [1, 2, 4, 8] captures multi-scale temporal patterns.

**Critical Question**: Do country futures follow predictable cycles, or are they more chaotic?

**Potential Issues**:
- **Structural Breaks**: Economic models may change fundamentally (e.g., post-2008 financial system)
- **Non-stationarity**: Factor relationships may evolve over time
- **Crisis Propagation**: Global shocks may have delayed, non-linear effects

**Proposed Improvements**:
```python
# Adaptive temporal modeling
class AdaptiveTCN(nn.Module):
    def __init__(self):
        self.stationary_tcn = TemporalConvNet(...)  # Stable patterns
        self.nonstationary_tcn = TemporalConvNet(...)  # Evolving patterns
        self.change_detector = ChangePointDetection()  # Detect structural breaks
```

#### **3. Country Heterogeneity: One Size Fits All?**

**Current Approach**: Single model trained on all countries with LOCO validation.

**Critical Question**: Should developed and emerging countries be modeled differently?

**Potential Issues**:
- **Different Growth Patterns**: Emerging markets may follow different development trajectories
- **Institutional Differences**: Factor relationships may vary by political system
- **Data Quality**: Emerging markets may have more missing data or measurement errors

**Proposed Improvements**:
```python
# Hierarchical country modeling
class HierarchicalCountryModel(nn.Module):
    def __init__(self):
        self.shared_backbone = FactorConvTCN(...)  # Shared patterns
        self.country_specific = nn.ModuleDict({
            'developed': CountrySpecificHead(...),
            'emerging': CountrySpecificHead(...),
            'frontier': CountrySpecificHead(...)
        })
```

#### **4. External Shocks: Missing Global Context?**

**Current Approach**: Model only uses country-specific factors.

**Critical Question**: Can we predict country futures without considering global context?

**Critical Missing Factors**:
- **Global Economic Cycles**: Recessions, commodity price shocks
- **Geopolitical Events**: Wars, trade wars, sanctions
- **Technological Disruptions**: AI, climate change, pandemics
- **Regional Integration**: EU, ASEAN, trade agreements

**Proposed Improvements**:
```python
# Multi-modal architecture
class GlobalAwareModel(nn.Module):
    def __init__(self):
        self.country_factors = FactorConvTCN(...)  # Country-specific
        self.global_factors = GlobalFactorEncoder(...)  # Global context
        self.interaction_layer = CrossModalAttention(...)  # Country-global interactions
```

#### **5. Causal vs. Correlational: Are We Predicting or Explaining?**

**Current Approach**: Predictive modeling without explicit causal structure.

**Critical Question**: Should we model causal relationships explicitly?

**Potential Issues**:
- **Spurious Correlations**: High military spending may correlate with low education due to budget constraints
- **Reverse Causality**: Strong countries may attract more trade, not vice versa
- **Confounding Variables**: Missing factors may drive apparent relationships

**Proposed Improvements**:
```python
# Causal-aware architecture
class CausalForecastNet(nn.Module):
    def __init__(self):
        self.causal_graph = CausalGraphLearner(...)  # Learn causal structure
        self.intervention_simulator = InterventionSimulator(...)  # What-if analysis
        self.counterfactual_head = CounterfactualHead(...)  # Alternative scenarios
```

### **Specific Architectural Recommendations**

#### **Priority 1: Multi-Scale Factor Interactions**
```python
# Enhanced factor convolution with multiple scales
class MultiScaleFactorConv(nn.Module):
    def __init__(self, n_factors=8):
        self.local_conv = nn.Conv1d(1, 16, kernel_size=2)  # Adjacent factors
        self.medium_conv = nn.Conv1d(1, 16, kernel_size=4)  # Medium-range
        self.global_conv = nn.Conv1d(1, 16, kernel_size=8)  # All factors
        self.factor_attention = nn.MultiheadAttention(8, num_heads=2)
```

#### **Priority 2: Country-Specific Adaptation**
```python
# Adaptive country modeling
class AdaptiveCountryModel(nn.Module):
    def __init__(self):
        self.shared_encoder = FactorConvTCN(...)
        self.country_classifier = CountryTypeClassifier(...)  # Developed/Emerging/Frontier
        self.adaptive_heads = nn.ModuleDict({
            'developed': DevelopedCountryHead(...),
            'emerging': EmergingCountryHead(...),
            'frontier': FrontierCountryHead(...)
        })
```

#### **Priority 3: External Shock Integration**
```python
# Global context integration
class GlobalContextModel(nn.Module):
    def __init__(self):
        self.country_encoder = FactorConvTCN(...)
        self.global_encoder = GlobalFactorEncoder(...)  # Oil prices, geopolitical indices
        self.shock_detector = ShockDetector(...)  # Detect global events
        self.interaction_net = CrossModalAttention(...)
```

#### **Priority 4: Uncertainty Decomposition**
```python
# Decomposed uncertainty estimation
class DecomposedUncertaintyHead(nn.Module):
    def __init__(self):
        self.aleatoric_head = AleatoricUncertaintyHead(...)  # Data uncertainty
        self.epistemic_head = EpistemicUncertaintyHead(...)  # Model uncertainty
        self.external_head = ExternalUncertaintyHead(...)  # Global shock uncertainty
```

### **Implementation Priority**

1. **Immediate (Next Sprint)**: Multi-scale factor interactions
2. **Short-term (1-2 months)**: Country-specific adaptation
3. **Medium-term (3-6 months)**: External shock integration
4. **Long-term (6+ months)**: Causal modeling and uncertainty decomposition

### **Validation Strategy**

To validate these improvements:

1. **Ablation Studies**: Test each component individually
2. **Cross-Country Analysis**: Compare performance across country groups
3. **Crisis Testing**: Evaluate performance during known crisis periods
4. **What-if Analysis**: Test counterfactual scenarios
5. **Expert Validation**: Consult with economists and policy experts

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
├── src/
│   ├── data_schema.py          # Pydantic models and factor definitions
│   ├── features.py            # Data preprocessing and windowing
│   ├── metrics.py             # Evaluation metrics
│   ├── training.py            # Training utilities and LOCO
│   ├── eval_loco.py           # LOCO evaluation and visualization
│   ├── ui_demo.py             # Streamlit web application
│   ├── data_ingest/           # Data loading modules
│   │   ├── load_wdi.py        # World Bank WDI data
│   │   ├── load_wipo.py       # WIPO patent data
│   │   ├── load_sipri.py      # SIPRI military data
│   │   ├── load_imf_cofer.py  # IMF COFER data
│   │   └── merge_panel.py     # Panel data merging
│   └── models/                # Neural network models
│       ├── factor_conv.py     # Factor-wise convolution
│       ├── temporal_tcn.py    # Temporal convolutional network
│       └── forecast_net.py    # Main forecasting network
├── configs/
│   └── base.yaml              # Model configuration
├── notebooks/
│   └── 00_quickstart.ipynb   # End-to-end tutorial
├── tests/
│   └── test_core.py          # Unit tests
├── data/                      # Data directory
├── results/                   # Results and models
├── make_synth_data.py        # Synthetic data generator
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd country_standing_forecast

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
python make_synth_data.py
```

This creates `data/synthetic_country_data.csv` with realistic synthetic data for 16 countries over 24 years.

### 3. Run Training

```bash
python -m src.training --config configs/base.yaml
```

### 4. Launch Interactive UI

```bash
streamlit run src/ui_demo.py
```

### 5. Run Jupyter Tutorial

```bash
jupyter notebook notebooks/00_quickstart.ipynb
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

### Composite Standing Score

The composite standing score is calculated as a weighted average of normalized factors:

```
Standing = Σ(wi × Ni)
```

Where:
- `wi` = weight for factor i
- `Ni` = normalized factor i (0-100 scale)
- Debt factor is inverted (lower debt = higher score)

## Model Architecture

### Factor Convolution Network
- **Purpose**: Capture interactions between macro factors
- **Architecture**: 1D convolution across factor dimension
- **Parameters**: Kernel size 2-3, multiple channels (16, 32)

### Temporal Convolutional Network (TCN)
- **Purpose**: Model temporal dependencies in time series
- **Architecture**: Dilated convolutions with residual connections
- **Parameters**: Dilations [1, 2, 4, 8], channels [32, 64]

### Prediction Heads
- **Point Forecasts**: Regression heads for each horizon
- **Quantile Regression**: Uncertainty estimation (0.1, 0.5, 0.9 quantiles)
- **Factor Forecasts**: Optional multi-task learning

### Model Specifications
- **Input**: `[batch_size, window_length, n_factors]` = `[B, 20, 8]`
- **Output**: `[batch_size, n_horizons]` = `[B, 3]`
- **Parameters**: ~200k (lightweight design)
- **Training**: Adam optimizer, learning rate 3e-3, early stopping

## Training Strategy

### Rolling-Origin Walk-Forward
- **Window Length**: 20 years of historical data
- **Step Size**: 1 year (overlapping windows)
- **Horizons**: 1, 5, 10 years ahead
- **Validation**: Time-based split (no future leakage)

### Leave-One-Country-Out (LOCO)
- **Purpose**: Assess generalization across countries
- **Method**: Train on N-1 countries, test on held-out country
- **Evaluation**: Per-country metrics and aggregated statistics

### Data Preprocessing
- **Scaling**: Robust scaler (median, IQR) to handle outliers
- **Imputation**: KNN imputation for missing values
- **Windowing**: Strict no-leakage (fit scalers on train only)

## Evaluation Metrics

### Point Forecast Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MASE**: Mean Absolute Scaled Error (vs naive forecast)
- **Spearman Correlation**: Rank correlation

### Uncertainty Metrics
- **Interval Coverage**: Percentage of true values within prediction intervals
- **Pinball Loss**: Quantile regression loss

### Cross-Validation
- **LOCO**: Leave-One-Country-Out evaluation
- **Rolling Origin**: Time-based validation
- **Country Rankings**: Performance comparison across countries

## Usage Examples

### Basic Training

```python
from src.training import rolling_origin_training
from src.data_ingest.merge_panel import load_all_data

# Load data
panel_data = load_all_data(Path("data"))

# Train model
results = rolling_origin_training(
    panel_data=panel_data,
    config=config,
    train_years=list(range(2000, 2015)),
    val_years=list(range(2015, 2020)),
    test_years=list(range(2020, 2024))
)
```

### LOCO Evaluation

```python
from src.eval_loco import run_loco_evaluation

# Run LOCO evaluation
loco_results = run_loco_evaluation(
    panel_data=panel_data,
    config=config,
    device="cpu"
)
```

### Generate Forecasts

```python
from src.models.forecast_net import create_model
from src.features import FactorScaler

# Load trained model
model = create_model(config["model"], device="cpu")
scaler = FactorScaler()

# Generate forecast
predictions = model(input_tensor)
```

## Configuration

The model behavior is controlled via `configs/base.yaml`:

```yaml
model:
  n_factors: 8
  window_length: 20
  horizons: [1, 5, 10]
  factor_conv_channels: [16, 32]
  tcn_channels: [32, 64]
  use_quantiles: true
  quantiles: [0.1, 0.5, 0.9]

training:
  learning_rate: 0.003
  batch_size: 64
  epochs: 100
  patience: 10

data:
  scaler_type: "robust"
  impute_strategy: "knn"
```

## Adding Real Data

To replace synthetic data with real sources:

1. **Implement Data Loaders**: Complete the TODO sections in `src/data_ingest/`
2. **Update Data Schema**: Modify factor definitions in `src/data_schema.py`
3. **Validate Data Quality**: Check missing data patterns and outliers
4. **Retrain Models**: Run training with real data

### Data Sources

- **World Bank WDI**: Education, trade, debt indicators
- **WIPO**: Patent statistics database
- **WEF**: Global Competitiveness Report
- **SIPRI**: Military expenditure database
- **IMF COFER**: Currency composition of foreign exchange reserves
- **GFCI**: Global Financial Centres Index

## Testing

Run unit tests to verify functionality:

```bash
python -m pytest tests/
```

Or run specific test file:

```bash
python tests/test_core.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by Ray Dalio's framework for measuring country strength
- Built with PyTorch, Streamlit, and scikit-learn
- Uses synthetic data for immediate testing and demonstration

## Future Enhancements

- **Real Data Integration**: Complete implementation of data loaders
- **Ensemble Methods**: Combine multiple models for improved accuracy
- **Feature Engineering**: Add more sophisticated transformations
- **Causal Analysis**: Incorporate causal inference methods
- **Real-time Updates**: Stream processing for live forecasts
- **API Interface**: REST API for programmatic access
