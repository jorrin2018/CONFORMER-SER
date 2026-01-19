"""
Temporal Alignment Module using Adaptive DTW.

Learns to align temporal sequences from different modalities (audio and text)
by learning warping paths that match events across modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TemporalAlignmentModule(nn.Module):
    """
    Adaptive DTW-based temporal alignment module.
    
    Learns to align sequences from two modalities by computing a learned
    warping path that matches events across time. Uses attention mechanisms
    to learn the alignment in a differentiable way.
    """

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        alignment_dim: int = 128,
        num_layers: int = 2,
        max_warp_factor: float = 1.5,
        dropout: float = 0.3,
    ):
        """
        Args:
            audio_dim: Dimension of audio features
            text_dim: Dimension of text features
            alignment_dim: Hidden dimension for alignment computation
            num_layers: Number of alignment layers
            max_warp_factor: Maximum temporal stretch/compression factor (1.0 = no warp)
            dropout: Dropout rate
        """
        super().__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.alignment_dim = alignment_dim
        self.max_warp_factor = max_warp_factor

        # Project both modalities to alignment space
        self.audio_proj = nn.Linear(audio_dim, alignment_dim)
        self.text_proj = nn.Linear(text_dim, alignment_dim)

        # Alignment layers - learn correspondence between modalities
        self.alignment_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=alignment_dim,
                nhead=8,
                dim_feedforward=alignment_dim * 4,
                dropout=dropout,
                batch_first=True,
            ) for _ in range(num_layers)
        ])

        # Warping path computation network
        # Computes alignment weights for each (audio_time, text_time) pair
        self.warping_net = nn.Sequential(
            nn.Linear(alignment_dim * 2, alignment_dim),
            nn.LayerNorm(alignment_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(alignment_dim, 1),
        )

        # Smoothness regularization (encourages smooth warping paths)
        self.smoothness_weight = 0.1

    def compute_warping_path(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute warping path between audio and text sequences.
        
        Args:
            audio_features: (batch_size, audio_seq_len, audio_dim)
            text_features: (batch_size, text_seq_len, text_dim)
        
        Returns:
            warping_matrix: (batch_size, audio_seq_len, text_seq_len) - alignment weights
            alignment_loss: Scalar regularization loss for smoothness
        """
        batch_size, audio_len, _ = audio_features.shape
        _, text_len, _ = text_features.shape

        # Project to alignment space
        audio_aligned = self.audio_proj(audio_features)  # (B, A, H)
        text_aligned = self.text_proj(text_features)     # (B, T, H)

        # Apply alignment layers (cross-modal attention)
        # Stack both sequences for joint processing
        combined = torch.cat([audio_aligned, text_aligned], dim=1)  # (B, A+T, H)
        
        for layer in self.alignment_layers:
            combined = layer(combined)  # (B, A+T, H)
        
        # Split back
        audio_refined = combined[:, :audio_len, :]  # (B, A, H)
        text_refined = combined[:, audio_len:, :]    # (B, T, H)

        # Compute pairwise similarity for warping path
        # Expand for pairwise computation
        audio_expanded = audio_refined.unsqueeze(2)  # (B, A, 1, H)
        text_expanded = text_refined.unsqueeze(1)    # (B, 1, T, H)

        # Concatenate pairs
        pairs = torch.cat([audio_expanded.expand(-1, -1, text_len, -1),
                          text_expanded.expand(-1, audio_len, -1, -1)], dim=-1)  # (B, A, T, 2*H)

        # Compute warping weights
        warping_logits = self.warping_net(pairs).squeeze(-1)  # (B, A, T)
        
        # Apply softmax to get alignment probabilities
        # Normalize across text dimension for each audio timestep
        warping_matrix = F.softmax(warping_logits, dim=-1)  # (B, A, T)

        # Compute smoothness regularization loss
        # Encourage monotonic and smooth warping paths
        alignment_loss = self._compute_smoothness_loss(warping_matrix)

        return warping_matrix, alignment_loss

    def _compute_smoothness_loss(self, warping_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute smoothness regularization loss for warping path.
        
        Encourages:
        1. Monotonicity: warping path should generally move forward
        2. Smoothness: adjacent audio timesteps should map to nearby text timesteps
        
        Args:
            warping_matrix: (batch_size, audio_len, text_len)
        
        Returns:
            Smoothness loss (scalar)
        """
        batch_size, audio_len, text_len = warping_matrix.shape

        # Monotonicity loss: penalize backward movements
        # Compute expected text position for each audio timestep
        text_positions = torch.arange(text_len, device=warping_matrix.device, dtype=torch.float32)
        expected_positions = torch.sum(
            warping_matrix * text_positions.unsqueeze(0).unsqueeze(0), dim=-1
        )  # (B, A)

        # Penalize if expected position decreases (backward movement)
        position_diffs = expected_positions[:, 1:] - expected_positions[:, :-1]  # (B, A-1)
        monotonicity_loss = F.relu(-position_diffs).mean()

        # Smoothness loss: penalize large jumps
        # Variance of expected positions
        smoothness_loss = torch.var(expected_positions, dim=-1).mean()

        total_loss = self.smoothness_weight * (monotonicity_loss + smoothness_loss)
        return total_loss

    def apply_warping(
        self,
        text_features: torch.Tensor,
        warping_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply warping to text features to align with audio.
        
        Args:
            text_features: (batch_size, text_seq_len, text_dim)
            warping_matrix: (batch_size, audio_seq_len, text_seq_len)
        
        Returns:
            aligned_text: (batch_size, audio_seq_len, text_dim)
        """
        # Warp text features to match audio sequence length
        # For each audio timestep, compute weighted average of text features
        aligned_text = torch.bmm(warping_matrix, text_features)  # (B, A, text_dim)
        return aligned_text

    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Align temporal sequences from audio and text modalities.
        
        Args:
            audio_features: (batch_size, audio_seq_len, audio_dim)
            text_features: (batch_size, text_seq_len, text_dim)
        
        Returns:
            aligned_audio: (batch_size, audio_seq_len, audio_dim) - may be warped
            aligned_text: (batch_size, audio_seq_len, text_dim) - warped to match audio
            alignment_loss: Regularization loss for smoothness
        """
        # Compute warping path
        warping_matrix, alignment_loss = self.compute_warping_path(
            audio_features, text_features
        )

        # Apply warping to text to align with audio
        aligned_text = self.apply_warping(text_features, warping_matrix)

        # Audio can optionally be warped too, but for simplicity we keep it fixed
        # and only warp text to match audio
        aligned_audio = audio_features

        return aligned_audio, aligned_text, alignment_loss






