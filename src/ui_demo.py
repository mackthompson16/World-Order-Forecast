"""
Streamlit UI demo for country standing forecast.

Interactive web application with country selector, year slider,
predicted vs actual plots, and what-if scenario analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from pathlib import Path
import pickle
import logging
from typing import Dict, List, Optional, Tuple

# Import our modules
from src.data_schema import CountryData, FACTOR_NAMES, FACTORS, compute_composite_standing
from src.features import FactorScaler, windowify_panel, create_data_loaders
from src.models.forecast_net import CountryStandingForecastNet, create_model
from src.training import ForecastTrainer
from src.metrics import ForecastMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Country Standing Forecast",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .factor-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_synthetic_data() -> pd.DataFrame:
    """Load synthetic data."""
    data_path = Path("data/synthetic_country_data.csv")
    
    if not data_path.exists():
        st.error(f"Data file not found: {data_path}")
        st.info("Please run `python make_synth_data.py` to generate synthetic data first.")
        return pd.DataFrame()
    
    df = pd.read_csv(data_path)
    return df


@st.cache_data
def load_model_and_scaler(model_path: Path):
    """Load trained model and scaler."""
    try:
        # Load model checkpoint
        checkpoint = torch.load(model_path / "model.pt", map_location="cpu")
        
        # Create model
        model_config = checkpoint["model_config"]
        model = create_model(model_config, device="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        # Load scaler
        with open(model_path / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        return model, scaler, model_config
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def create_country_data_objects(df: pd.DataFrame) -> List[CountryData]:
    """Convert DataFrame to CountryData objects."""
    country_data = []
    
    for _, row in df.iterrows():
        data = CountryData(
            country=row['country'],
            year=row['year'],
            education=row.get('education'),
            innovation=row.get('innovation'),
            competitiveness=row.get('competitiveness'),
            military=row.get('military'),
            trade_share=row.get('trade_share'),
            reserve_currency_proxy=row.get('reserve_currency_proxy'),
            financial_center_proxy=row.get('financial_center_proxy'),
            debt=row.get('debt'),
            mask_education=row.get('mask_education', False),
            mask_innovation=row.get('mask_innovation', False),
            mask_competitiveness=row.get('mask_competitiveness', False),
            mask_military=row.get('mask_military', False),
            mask_trade_share=row.get('mask_trade_share', False),
            mask_reserve_currency_proxy=row.get('mask_reserve_currency_proxy', False),
            mask_financial_center_proxy=row.get('mask_financial_center_proxy', False),
            mask_debt=row.get('mask_debt', False)
        )
        country_data.append(data)
    
    return country_data


def get_country_timeseries(country_data: List[CountryData], country: str) -> Tuple[np.ndarray, np.ndarray]:
    """Get time series data for a specific country."""
    country_data_filtered = [d for d in country_data if d.country == country]
    country_data_filtered.sort(key=lambda x: x.year)
    
    years = np.array([d.year for d in country_data_filtered])
    factor_values = np.array([[getattr(d, factor) or 0.0 for factor in FACTOR_NAMES] for d in country_data_filtered])
    
    return years, factor_values


def plot_factor_evolution(country: str, years: np.ndarray, factor_values: np.ndarray):
    """Plot factor evolution over time."""
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=FACTOR_NAMES,
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    for i, factor in enumerate(FACTOR_NAMES):
        row = i // 4 + 1
        col = i % 4 + 1
        
        fig.add_trace(
            go.Scatter(
                x=years,
                y=factor_values[:, i],
                mode='lines+markers',
                name=factor,
                line=dict(width=2),
                marker=dict(size=4)
            ),
            row=row, col=col
        )
        
        # Add factor description
        factor_def = FACTORS[factor]
        fig.update_xaxes(title_text="Year", row=row, col=col)
        fig.update_yaxes(title_text=factor_def.unit, row=row, col=col)
    
    fig.update_layout(
        title=f"Factor Evolution: {country}",
        height=600,
        showlegend=False
    )
    
    return fig


def plot_standing_forecast(
    country: str,
    years: np.ndarray,
    actual_standing: np.ndarray,
    predictions: Dict[str, np.ndarray],
    cutoff_year: int
):
    """Plot standing forecast with confidence intervals."""
    fig = go.Figure()
    
    # Split data into historical and forecast periods
    hist_mask = years <= cutoff_year
    forecast_mask = years > cutoff_year
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=years[hist_mask],
        y=actual_standing[hist_mask],
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue', width=3),
        marker=dict(size=6)
    ))
    
    # Forecast data
    if np.any(forecast_mask):
        forecast_years = years[forecast_mask]
        
        # Point forecasts
        for horizon in [1, 5, 10]:
            horizon_key = f"standing_{horizon}y"
            if horizon_key in predictions:
                pred_values = predictions[horizon_key]
                if len(pred_values) > 0:
                    fig.add_trace(go.Scatter(
                        x=forecast_years,
                        y=pred_values,
                        mode='lines+markers',
                        name=f'Forecast {horizon}y',
                        line=dict(dash='dash', width=2),
                        marker=dict(size=4)
                    ))
        
        # Confidence intervals (if available)
        for horizon in [1, 5, 10]:
            quantile_key = f"quantiles_{horizon}y"
            if quantile_key in predictions:
                quantiles = predictions[quantile_key]
                if len(quantiles) > 0 and quantiles.shape[1] >= 3:
                    lower = quantiles[:, 0]  # 0.1 quantile
                    upper = quantiles[:, 2]   # 0.9 quantile
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_years,
                        y=upper,
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_years,
                        y=lower,
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor=f'rgba(0, 100, 200, 0.2)',
                        name=f'{horizon}y Confidence',
                        hoverinfo='skip'
                    ))
    
    fig.update_layout(
        title=f"Country Standing Forecast: {country}",
        xaxis_title="Year",
        yaxis_title="Standing Score (0-100)",
        height=500,
        hovermode='x unified'
    )
    
    return fig


def run_what_if_analysis(
    model: CountryStandingForecastNet,
    scaler: FactorScaler,
    base_factors: np.ndarray,
    factor_shocks: Dict[str, float],
    horizons: List[int] = [1, 5, 10]
) -> Dict[str, np.ndarray]:
    """Run what-if scenario analysis."""
    # Apply shocks to factors
    shocked_factors = base_factors.copy()
    
    for factor_name, shock in factor_shocks.items():
        if factor_name in FACTOR_NAMES:
            factor_idx = FACTOR_NAMES.index(factor_name)
            shocked_factors[factor_idx] *= (1 + shock)
    
    # Scale the shocked factors
    scaled_factors, _ = scaler.transform(shocked_factors.reshape(1, -1))
    
    # Create input tensor
    input_tensor = torch.tensor(scaled_factors, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    # Generate predictions
    with torch.no_grad():
        predictions = model(input_tensor)
    
    # Extract predictions
    results = {}
    for horizon in horizons:
        horizon_key = f"standing_{horizon}y"
        if horizon_key in predictions:
            results[horizon_key] = predictions[horizon_key].numpy()
    
    return results


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">🌍 Country Standing Forecast</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_synthetic_data()
    
    if df.empty:
        st.stop()
    
    # Convert to CountryData objects
    country_data = create_country_data_objects(df)
    
    # Sidebar
    st.sidebar.title("Controls")
    
    # Country selector
    countries = sorted(df['country'].unique())
    selected_country = st.sidebar.selectbox(
        "Select Country",
        countries,
        index=countries.index("USA") if "USA" in countries else 0
    )
    
    # Year slider
    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    cutoff_year = st.sidebar.slider(
        "Forecast Cutoff Year",
        min_value=min_year + 5,
        max_value=max_year - 5,
        value=max_year - 10,
        step=1
    )
    
    # Get country data
    years, factor_values = get_country_timeseries(country_data, selected_country)
    
    if len(years) == 0:
        st.error(f"No data found for {selected_country}")
        st.stop()
    
    # Calculate actual standing scores
    actual_standing = np.array([compute_composite_standing(fv) for fv in factor_values])
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Factor Evolution")
        fig_factors = plot_factor_evolution(selected_country, years, factor_values)
        st.plotly_chart(fig_factors, use_container_width=True)
    
    with col2:
        st.subheader("Current Standing")
        current_standing = actual_standing[years <= cutoff_year][-1] if np.any(years <= cutoff_year) else 0
        
        st.metric(
            label="Standing Score",
            value=f"{current_standing:.1f}",
            delta=None
        )
        
        # Factor values at cutoff year
        cutoff_idx = np.where(years == cutoff_year)[0]
        if len(cutoff_idx) > 0:
            cutoff_factors = factor_values[cutoff_idx[0]]
            
            st.subheader("Factor Values")
            for i, factor in enumerate(FACTOR_NAMES):
                factor_def = FACTORS[factor]
                st.metric(
                    label=factor_def.description,
                    value=f"{cutoff_factors[i]:.2f}",
                    help=f"Unit: {factor_def.unit}"
                )
    
    # Forecast section
    st.subheader("Standing Forecast")
    
    # Check if model exists
    model_path = Path("results/models")
    if model_path.exists() and (model_path / "model.pt").exists():
        model, scaler, model_config = load_model_and_scaler(model_path)
        
        if model is not None:
            # Generate forecasts (simplified - using last available data)
            last_data_idx = np.where(years <= cutoff_year)[0][-1]
            last_factors = factor_values[last_data_idx]
            
            # Scale factors
            scaled_factors, _ = scaler.transform(last_factors.reshape(1, -1))
            
            # Create input tensor (simplified - using single timestep)
            input_tensor = torch.tensor(scaled_factors, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # Generate predictions
            with torch.no_grad():
                predictions = model(input_tensor)
            
            # Extract predictions
            forecast_results = {}
            for horizon in [1, 5, 10]:
                horizon_key = f"standing_{horizon}y"
                if horizon_key in predictions:
                    forecast_results[horizon_key] = predictions[horizon_key].numpy()
            
            # Plot forecast
            fig_forecast = plot_standing_forecast(
                selected_country, years, actual_standing, forecast_results, cutoff_year
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Forecast metrics
            st.subheader("Forecast Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("1-Year Forecast", f"{forecast_results.get('standing_1y', [0])[0]:.1f}")
            with col2:
                st.metric("5-Year Forecast", f"{forecast_results.get('standing_5y', [0])[0]:.1f}")
            with col3:
                st.metric("10-Year Forecast", f"{forecast_results.get('standing_10y', [0])[0]:.1f}")
        else:
            st.warning("Model could not be loaded. Please train a model first.")
    else:
        st.warning("No trained model found. Please run training first.")
        
        # Show actual data only
        fig_actual = go.Figure()
        fig_actual.add_trace(go.Scatter(
            x=years,
            y=actual_standing,
            mode='lines+markers',
            name='Actual Standing',
            line=dict(color='blue', width=3),
            marker=dict(size=6)
        ))
        
        fig_actual.update_layout(
            title=f"Actual Standing: {selected_country}",
            xaxis_title="Year",
            yaxis_title="Standing Score (0-100)",
            height=400
        )
        
        st.plotly_chart(fig_actual, use_container_width=True)
    
    # What-if analysis
    st.subheader("What-If Analysis")
    st.write("Adjust factor values to see impact on standing forecast:")
    
    # Factor shock controls
    factor_shocks = {}
    cols = st.columns(4)
    
    for i, factor in enumerate(FACTOR_NAMES):
        with cols[i % 4]:
            factor_def = FACTORS[factor]
            shock = st.slider(
                f"{factor_def.description}",
                min_value=-0.5,
                max_value=0.5,
                value=0.0,
                step=0.05,
                format="%.1%",
                help=f"Shock to {factor_def.description}"
            )
            factor_shocks[factor] = shock
    
    # Run what-if analysis
    if st.button("Run What-If Analysis"):
        if model is not None and scaler is not None:
            # Get base factors
            base_factors = factor_values[years <= cutoff_year][-1]
            
            # Run analysis
            whatif_results = run_what_if_analysis(
                model, scaler, base_factors, factor_shocks
            )
            
            # Display results
            st.subheader("What-If Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                base_1y = forecast_results.get('standing_1y', [0])[0]
                whatif_1y = whatif_results.get('standing_1y', [0])[0]
                st.metric(
                    "1-Year Forecast",
                    value=f"{whatif_1y:.1f}",
                    delta=f"{whatif_1y - base_1y:.1f}"
                )
            
            with col2:
                base_5y = forecast_results.get('standing_5y', [0])[0]
                whatif_5y = whatif_results.get('standing_5y', [0])[0]
                st.metric(
                    "5-Year Forecast",
                    value=f"{whatif_5y:.1f}",
                    delta=f"{whatif_5y - base_5y:.1f}"
                )
            
            with col3:
                base_10y = forecast_results.get('standing_10y', [0])[0]
                whatif_10y = whatif_results.get('standing_10y', [0])[0]
                st.metric(
                    "10-Year Forecast",
                    value=f"{whatif_10y:.1f}",
                    delta=f"{whatif_10y - base_10y:.1f}"
                )
        else:
            st.error("Model not available for what-if analysis")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Country Standing Forecast** - A machine learning approach to predicting national strength "
        "using macro factors inspired by Ray Dalio's framework."
    )


if __name__ == "__main__":
    main()
