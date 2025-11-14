"""
Alternative classification heads for ablation studies.
"""

import torch
import torch.nn as nn
from typing import Optional


class LSTMClassificationHead(nn.Module):
    """
    LSTM-based classification head.
    
    Uses bidirectional LSTM to process features as sequence,
    then projects to classification space.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
        num_layers: int = 2,
        bidirectional: bool = True
    ):
        """
        Args:
            hidden_dim: Input feature dimension (output from fusion layers)
            num_classes: Number of output classes
            dropout: Dropout rate
            num_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Projection after LSTM
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Final classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            fused_features: Features from fusion module (batch_size, hidden_dim)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Add sequence dimension for LSTM
        x = fused_features.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            h_forward = hidden[0]  # Last forward hidden state
            h_backward = hidden[1]  # Last backward hidden state
            h_concat = torch.cat([h_forward, h_backward], dim=-1)
        else:
            h_concat = hidden[-1]  # Last layer hidden state
        
        # Projection and classification
        x = self.projection(h_concat)
        logits = self.classifier(x)
        
        return logits


class TransformerClassificationHead(nn.Module):
    """
    Transformer-based classification head.
    
    Uses a small transformer encoder to process features,
    with attention pooling and residual connections.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
        num_layers: int = 1,
        num_heads: int = 8
    ):
        """
        Args:
            hidden_dim: Input feature dimension
            num_classes: Number of output classes
            dropout: Dropout rate
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Attention pooling
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP with residual connection
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Final classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            fused_features: Features from fusion module (batch_size, hidden_dim)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Add sequence dimension
        x = fused_features.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Transformer encoding
        x_transformed = self.transformer(x)
        
        # Attention pooling
        pooled, attention_weights = self.attention_pool(
            query=x_transformed,
            key=x_transformed,
            value=x_transformed
        )
        
        # Remove sequence dimension
        x = pooled.squeeze(1)  # (batch_size, hidden_dim)
        
        # Residual connection + MLP
        x = x + self.mlp(x)
        
        # Final classification
        logits = self.classifier(x)
        
        return logits


class CNN1DClassificationHead(nn.Module):
    """
    CNN 1D-based classification head.
    
    Uses 1D convolutions to process features, treating them as
    a temporal sequence for pattern extraction.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
        num_layers: int = 2
    ):
        """
        Args:
            hidden_dim: Input feature dimension
            num_classes: Number of output classes
            dropout: Dropout rate
            num_layers: Number of convolutional layers
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Build convolutional layers
        conv_layers = []
        in_channels = hidden_dim
        
        for i in range(num_layers):
            out_channels = hidden_dim * 2 if i == 0 else hidden_dim
            conv_layers.extend([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding=0
                ),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(dropout if i == 0 else dropout * 0.8)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            fused_features: Features from fusion module (batch_size, hidden_dim)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Transpose for CNN 1D: (batch_size, hidden_dim, 1)
        x = fused_features.unsqueeze(-1)  # (batch_size, hidden_dim, 1)
        
        # Pass through convolutional layers
        x = self.conv_layers(x)  # (batch_size, hidden_dim, 1)
        
        # Global pooling
        x = self.global_pool(x)  # (batch_size, hidden_dim, 1)
        x = x.squeeze(-1)  # (batch_size, hidden_dim)
        
        # Classification
        logits = self.classifier(x)
        
        return logits

