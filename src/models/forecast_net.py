"""
Main forecasting network combining factor convolution and temporal TCN.

This module implements the complete neural network architecture for
country standing forecasting with multiple prediction heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
import math

from .factor_conv import MultiFactorConv
from .temporal_tcn import TemporalConvNetWithAttention


class ForecastHead(nn.Module):
    """
    Prediction head for forecasting at specific horizons.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        Initialize ForecastHead.
        
        Args:
            in_channels: Number of input channels
            hidden_dim: Hidden dimension size
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights."""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x


class QuantileHead(nn.Module):
    """
    Quantile regression head for uncertainty estimation.
    """
    
    def __init__(
        self,
        in_channels: int,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize QuantileHead.
        
        Args:
            in_channels: Number of input channels
            quantiles: List of quantiles to predict
            hidden_dim: Hidden dimension size
            dropout: Dropout probability
        """
        super().__init__()
        
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        
        # Shared layers
        self.shared_fc1 = nn.Linear(in_channels, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Quantile-specific heads
        self.quantile_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(self.n_quantiles)
        ])
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights."""
        for layer in [self.shared_fc1, self.shared_fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        for head in self.quantile_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels)
            
        Returns:
            Output tensor of shape (batch_size, n_quantiles)
        """
        # Shared layers
        x = self.shared_fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.shared_fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Quantile-specific predictions
        quantile_outputs = []
        for head in self.quantile_heads:
            quantile_outputs.append(head(x))
        
        return torch.cat(quantile_outputs, dim=1)


class FactorForecastHead(nn.Module):
    """
    Multi-task head for forecasting individual factors.
    """
    
    def __init__(
        self,
        in_channels: int,
        n_factors: int = 8,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize FactorForecastHead.
        
        Args:
            in_channels: Number of input channels
            n_factors: Number of factors to forecast
            hidden_dim: Hidden dimension size
            dropout: Dropout probability
        """
        super().__init__()
        
        self.n_factors = n_factors
        
        # Shared layers
        self.shared_fc1 = nn.Linear(in_channels, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Factor-specific heads
        self.factor_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(n_factors)
        ])
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights."""
        for layer in [self.shared_fc1, self.shared_fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        for head in self.factor_heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels)
            
        Returns:
            Output tensor of shape (batch_size, n_factors)
        """
        # Shared layers
        x = self.shared_fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.shared_fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Factor-specific predictions
        factor_outputs = []
        for head in self.factor_heads:
            factor_outputs.append(head(x))
        
        return torch.cat(factor_outputs, dim=1)


class CountryStandingForecastNet(nn.Module):
    """
    Main neural network for country standing forecasting.
    
    Combines factor-wise convolution with temporal TCN and multiple prediction heads.
    """
    
    def __init__(
        self,
        n_factors: int = 8,
        window_length: int = 20,
        horizons: List[int] = [1, 5, 10],
        factor_conv_channels: List[int] = [16, 32],
        factor_conv_kernels: List[int] = [3, 3],
        tcn_channels: List[int] = [32, 64],
        tcn_dilations: List[int] = [1, 2, 4, 8],
        tcn_kernel_size: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        activation: str = "relu",
        use_attention: bool = True,
        use_quantiles: bool = True,
        use_factor_forecast: bool = False,
        quantiles: List[float] = [0.1, 0.5, 0.9]
    ):
        """
        Initialize CountryStandingForecastNet.
        
        Args:
            n_factors: Number of input factors
            window_length: Length of input time window
            horizons: List of forecast horizons
            factor_conv_channels: List of factor conv channel sizes
            factor_conv_kernels: List of factor conv kernel sizes
            tcn_channels: List of TCN channel sizes
            tcn_dilations: List of TCN dilation rates
            tcn_kernel_size: TCN kernel size
            hidden_dim: Hidden dimension for prediction heads
            dropout: Dropout probability
            activation: Activation function
            use_attention: Whether to use temporal attention
            use_quantiles: Whether to use quantile regression
            use_factor_forecast: Whether to forecast individual factors
            quantiles: List of quantiles for uncertainty estimation
        """
        super().__init__()
        
        self.n_factors = n_factors
        self.window_length = window_length
        self.horizons = horizons
        self.use_quantiles = use_quantiles
        self.use_factor_forecast = use_factor_forecast
        self.quantiles = quantiles
        
        # Factor convolution network
        self.factor_conv = MultiFactorConv(
            n_factors=n_factors,
            hidden_channels=factor_conv_channels,
            kernel_sizes=factor_conv_kernels,
            activation=activation,
            dropout=dropout,
            use_residual=True
        )
        
        # Get output channels from factor conv
        factor_conv_out_channels = self.factor_conv.get_output_channels()
        
        # Temporal TCN
        self.temporal_tcn = TemporalConvNetWithAttention(
            in_channels=factor_conv_out_channels,
            hidden_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dilations=tcn_dilations,
            dropout=dropout,
            activation=activation,
            use_attention=use_attention,
            attention_dim=hidden_dim
        )
        
        # Get output channels from TCN
        tcn_out_channels = self.temporal_tcn.get_output_channels()
        
        # Global pooling to get final representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Prediction heads
        self.horizon_heads = nn.ModuleDict({
            f"horizon_{h}": ForecastHead(
                in_channels=tcn_out_channels,
                hidden_dim=hidden_dim,
                dropout=dropout,
                activation=activation
            ) for h in horizons
        })
        
        # Quantile heads (optional)
        if use_quantiles:
            self.quantile_heads = nn.ModuleDict({
                f"horizon_{h}": QuantileHead(
                    in_channels=tcn_out_channels,
                    quantiles=quantiles,
                    hidden_dim=hidden_dim,
                    dropout=dropout
                ) for h in horizons
            })
        
        # Factor forecast heads (optional)
        if use_factor_forecast:
            self.factor_heads = nn.ModuleDict({
                f"horizon_{h}": FactorForecastHead(
                    in_channels=tcn_out_channels,
                    n_factors=n_factors,
                    hidden_dim=hidden_dim,
                    dropout=dropout
                ) for h in horizons
            })
    
    def forward(
        self, 
        x: torch.Tensor, 
        masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, window_length, n_factors)
            masks: Optional mask tensor of shape (batch_size, window_length, n_factors)
            
        Returns:
            Dictionary containing predictions for each horizon
        """
        batch_size, seq_len, n_factors = x.shape
        
        # Factor convolution
        factor_features = self.factor_conv(x)  # (batch_size, seq_len, factor_conv_channels)
        
        # Transpose for TCN (batch_size, channels, seq_len)
        tcn_input = factor_features.transpose(1, 2)
        
        # Temporal TCN
        tcn_output, attention_weights = self.temporal_tcn(tcn_input)
        
        # Global pooling
        pooled = self.global_pool(tcn_output)  # (batch_size, channels, 1)
        pooled = pooled.squeeze(-1)  # (batch_size, channels)
        
        # Generate predictions for each horizon
        predictions = {}
        
        for horizon in self.horizons:
            # Point forecasts
            horizon_key = f"horizon_{horizon}"
            point_pred = self.horizon_heads[horizon_key](pooled)
            predictions[f"standing_{horizon}y"] = point_pred.squeeze(-1)
            
            # Quantile forecasts (optional)
            if self.use_quantiles:
                quantile_pred = self.quantile_heads[horizon_key](pooled)
                predictions[f"quantiles_{horizon}y"] = quantile_pred
            
            # Factor forecasts (optional)
            if self.use_factor_forecast:
                factor_pred = self.factor_heads[horizon_key](pooled)
                predictions[f"factors_{horizon}y"] = factor_pred
        
        # Add attention weights if available
        if attention_weights is not None:
            predictions["attention_weights"] = attention_weights
        
        return predictions
    
    def count_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_summary(self) -> Dict:
        """Get model architecture summary."""
        return {
            "total_parameters": self.count_parameters(),
            "n_factors": self.n_factors,
            "window_length": self.window_length,
            "horizons": self.horizons,
            "factor_conv_channels": self.factor_conv.hidden_channels,
            "tcn_channels": self.temporal_tcn.tcn.hidden_channels,
            "tcn_dilations": self.temporal_tcn.tcn.dilations,
            "use_attention": self.temporal_tcn.use_attention,
            "use_quantiles": self.use_quantiles,
            "use_factor_forecast": self.use_factor_forecast,
            "quantiles": self.quantiles if self.use_quantiles else None
        }


def create_model(
    config: Dict,
    device: str = "cpu"
) -> CountryStandingForecastNet:
    """
    Create model from configuration.
    
    Args:
        config: Model configuration dictionary
        device: Device to create model on
        
    Returns:
        Initialized model
    """
    model = CountryStandingForecastNet(**config)
    model = model.to(device)
    
    return model
