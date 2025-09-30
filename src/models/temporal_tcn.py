"""
Temporal Convolutional Network (TCN) for time series forecasting.

Implements dilated convolutions to capture long-range temporal dependencies
in the country standing forecast model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class TemporalBlock(nn.Module):
    """
    Temporal convolution block with residual connection and dropout.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        Initialize TemporalBlock.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            dilation: Dilation rate
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Calculate padding for causal convolution
        self.padding = (kernel_size - 1) * dilation
        
        # Convolution layers
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        
        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_conv = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights."""
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
        if self.residual_conv is not None:
            nn.init.xavier_uniform_(self.residual_conv.weight)
            if self.residual_conv.bias is not None:
                nn.init.zeros_(self.residual_conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, seq_len)
        """
        residual = x
        
        # First convolution
        out = self.conv1(x)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Second convolution
        out = self.conv2(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        
        # Add residual and apply final activation
        out = out + residual
        out = self.activation(out)
        
        return out


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network with dilated convolutions.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int] = [32, 64],
        kernel_size: int = 3,
        dilations: List[int] = [1, 2, 4, 8],
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        Initialize TemporalConvNet.
        
        Args:
            in_channels: Number of input channels
            hidden_channels: List of hidden channel sizes
            kernel_size: Size of convolution kernel
            dilations: List of dilation rates
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilations = dilations
        
        # Build temporal blocks
        layers = []
        num_levels = len(dilations)
        
        for i in range(num_levels):
            dilation = dilations[i]
            in_ch = in_channels if i == 0 else hidden_channels[i-1]
            out_ch = hidden_channels[i] if i < len(hidden_channels) else hidden_channels[-1]
            
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    activation=activation
                )
            )
        
        self.temporal_blocks = nn.ModuleList(layers)
        
        # Final projection
        self.final_projection = nn.Conv1d(
            in_channels=hidden_channels[-1],
            out_channels=hidden_channels[-1],
            kernel_size=1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, final_channels, seq_len)
        """
        for block in self.temporal_blocks:
            x = block(x)
        
        x = self.final_projection(x)
        
        return x
    
    def get_output_channels(self) -> int:
        """Get number of output channels."""
        return self.hidden_channels[-1]


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for focusing on relevant time steps.
    """
    
    def __init__(
        self,
        in_channels: int,
        attention_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize TemporalAttention.
        
        Args:
            in_channels: Number of input channels
            attention_dim: Dimension of attention computation
            dropout: Dropout probability
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.attention_dim = attention_dim
        
        # Attention layers
        self.query = nn.Linear(in_channels, attention_dim)
        self.key = nn.Linear(in_channels, attention_dim)
        self.value = nn.Linear(in_channels, in_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(attention_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, channels)
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        batch_size, seq_len, channels = x.shape
        
        # Compute queries, keys, values
        Q = self.query(x)  # (batch_size, seq_len, attention_dim)
        K = self.key(x)    # (batch_size, seq_len, attention_dim)
        V = self.value(x)  # (batch_size, seq_len, channels)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        
        return attended, attention_weights


class TemporalConvNetWithAttention(nn.Module):
    """
    TCN with temporal attention mechanism.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int] = [32, 64],
        kernel_size: int = 3,
        dilations: List[int] = [1, 2, 4, 8],
        dropout: float = 0.1,
        activation: str = "relu",
        use_attention: bool = True,
        attention_dim: int = 64
    ):
        """
        Initialize TemporalConvNetWithAttention.
        
        Args:
            in_channels: Number of input channels
            hidden_channels: List of hidden channel sizes
            kernel_size: Size of convolution kernel
            dilations: List of dilation rates
            dropout: Dropout probability
            activation: Activation function
            use_attention: Whether to use attention mechanism
            attention_dim: Dimension of attention computation
        """
        super().__init__()
        
        self.use_attention = use_attention
        
        # TCN
        self.tcn = TemporalConvNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
            activation=activation
        )
        
        # Attention
        if use_attention:
            self.attention = TemporalAttention(
                in_channels=hidden_channels[-1],
                attention_dim=attention_dim,
                dropout=dropout
            )
        else:
            self.attention = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, seq_len)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # TCN forward pass
        tcn_out = self.tcn(x)  # (batch_size, channels, seq_len)
        
        # Transpose for attention (batch_size, seq_len, channels)
        tcn_out_t = tcn_out.transpose(1, 2)
        
        if self.use_attention and self.attention is not None:
            # Apply attention
            attended, attention_weights = self.attention(tcn_out_t)
            # Transpose back (batch_size, channels, seq_len)
            output = attended.transpose(1, 2)
            return output, attention_weights
        else:
            return tcn_out, None
    
    def get_output_channels(self) -> int:
        """Get number of output channels."""
        return self.tcn.get_output_channels()
