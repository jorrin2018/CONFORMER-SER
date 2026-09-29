"""
Audio Context Network (ACN) to elevate audio features before fusion.

Applies a Conv1D layer followed by a lightweight Transformer Encoder.
"""

from typing import Optional

import torch
import torch.nn as nn


class AudioContextNetwork(nn.Module):
    """
    Audio Context Network (ACN).

    Args:
        input_dim: Input feature dimension from audio encoder.
        output_dim: Output feature dimension after ACN.
        conv_kernel_size: Kernel size for Conv1D.
        num_transformer_layers: Number of Transformer Encoder layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        conv_kernel_size: int = 3,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = conv_kernel_size // 2
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=conv_kernel_size,
            padding=padding,
        )
        self.conv_norm = nn.LayerNorm(output_dim)
        self.conv_act = nn.GELU()
        self.conv_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_transformer_layers,
        )
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        audio_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            audio_features: (batch_size, seq_len, input_dim)
            attention_mask: Optional mask (batch_size, seq_len), 1 for valid.

        Returns:
            Contextualized audio features (batch_size, seq_len, output_dim).
        """
        # Conv1D expects (B, C, S)
        x = audio_features.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.conv_norm(x)
        x = self.conv_act(x)
        x = self.conv_dropout(x)

        if attention_mask is not None:
            # Transformer expects True for padding positions
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.output_norm(x)
        return x


