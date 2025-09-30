"""
Factor-wise convolutional layer for processing macro factors.

Applies 1D convolution across the factor dimension to capture
interactions between different macro factors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FactorConv1D(nn.Module):
    """
    1D Convolutional layer applied across the factor dimension.
    
    This layer processes each time step independently, applying convolution
    across the factor dimension to capture interactions between macro factors.
    """
    
    def __init__(
        self,
        n_factors: int = 8,
        out_channels: int = 16,
        kernel_size: int = 3,
        padding: str = "same",
        activation: str = "relu",
        dropout: float = 0.1
    ):
        """
        Initialize FactorConv1D layer.
        
        Args:
            n_factors: Number of input factors
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            padding: Padding mode ("same", "valid", or integer)
            activation: Activation function ("relu", "gelu", "tanh", "none")
            dropout: Dropout probability
        """
        super().__init__()
        
        self.n_factors = n_factors
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Calculate padding
        if padding == "same":
            self.padding = (kernel_size - 1) // 2
        elif padding == "valid":
            self.padding = 0
        else:
            self.padding = padding
        
        # Convolution layer
        self.conv = nn.Conv1d(
            in_channels=1,  # Treat factors as single channel
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=self.padding
        )
        
        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights."""
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_factors)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, out_channels)
        """
        batch_size, seq_len, n_factors = x.shape
        
        # Reshape to (batch_size * seq_len, 1, n_factors)
        x_reshaped = x.view(batch_size * seq_len, 1, n_factors)
        
        # Apply convolution
        conv_out = self.conv(x_reshaped)  # (batch_size * seq_len, out_channels, n_factors)
        
        # Apply activation
        conv_out = self.activation(conv_out)
        
        # Apply dropout
        conv_out = self.dropout(conv_out)
        
        # Reshape back to (batch_size, seq_len, out_channels)
        # Take mean across the factor dimension
        output = conv_out.mean(dim=2)  # (batch_size * seq_len, out_channels)
        output = output.view(batch_size, seq_len, self.out_channels)
        
        return output


class FactorConvBlock(nn.Module):
    """
    Block combining factor convolution with residual connection.
    """
    
    def __init__(
        self,
        n_factors: int = 8,
        out_channels: int = 16,
        kernel_size: int = 3,
        activation: str = "relu",
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        """
        Initialize FactorConvBlock.
        
        Args:
            n_factors: Number of input factors
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            activation: Activation function
            dropout: Dropout probability
            use_residual: Whether to use residual connection
        """
        super().__init__()
        
        self.use_residual = use_residual
        
        # Factor convolution
        self.factor_conv = FactorConv1D(
            n_factors=n_factors,
            out_channels=out_channels,
            kernel_size=kernel_size,
            activation=activation,
            dropout=dropout
        )
        
        # Projection layer for residual connection
        if use_residual and n_factors != out_channels:
            self.projection = nn.Linear(n_factors, out_channels)
        else:
            self.projection = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_factors)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, out_channels)
        """
        # Factor convolution
        conv_out = self.factor_conv(x)
        
        # Residual connection
        if self.use_residual:
            if self.projection is not None:
                residual = self.projection(x)
            else:
                residual = x
            conv_out = conv_out + residual
        
        return conv_out


class MultiFactorConv(nn.Module):
    """
    Multi-layer factor convolution network.
    """
    
    def __init__(
        self,
        n_factors: int = 8,
        hidden_channels: List[int] = [16, 32],
        kernel_sizes: List[int] = [3, 3],
        activation: str = "relu",
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        """
        Initialize MultiFactorConv.
        
        Args:
            n_factors: Number of input factors
            hidden_channels: List of hidden channel sizes
            kernel_sizes: List of kernel sizes for each layer
            activation: Activation function
            dropout: Dropout probability
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.n_factors = n_factors
        self.hidden_channels = hidden_channels
        
        # Build layers
        layers = []
        in_channels = n_factors
        
        for i, (out_channels, kernel_size) in enumerate(zip(hidden_channels, kernel_sizes)):
            layers.append(
                FactorConvBlock(
                    n_factors=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    activation=activation,
                    dropout=dropout,
                    use_residual=use_residual
                )
            )
            in_channels = out_channels
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_factors)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, final_channels)
        """
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def get_output_channels(self) -> int:
        """Get number of output channels."""
        return self.hidden_channels[-1] if self.hidden_channels else self.n_factors
