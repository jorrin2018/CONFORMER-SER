"""
Bidirectional Connection Attention Fusion (BCAF).

Implements an autoencoder-style module to learn a shared latent space
and reconstruct modality features, with an alignment loss across modalities.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalConnectionAttentionFusion(nn.Module):
    """
    BCAF Autoencoder for audio, text, and vision features.

    Args:
        audio_dim: Audio feature dimension.
        text_dim: Text feature dimension.
        vision_dim: Vision feature dimension.
        latent_dim: Latent space dimension.
        hidden_dim: Hidden size for encoder/decoder MLP.
        dropout: Dropout probability.
        reconstruction_weight: Weight for reconstruction loss.
        alignment_weight: Weight for alignment loss.
    """

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        vision_dim: int,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        reconstruction_weight: float = 1.0,
        alignment_weight: float = 0.5,
    ):
        super().__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.latent_dim = latent_dim
        self.reconstruction_weight = reconstruction_weight
        self.alignment_weight = alignment_weight

        input_dim = audio_dim + text_dim + vision_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        self.audio_proj = nn.Linear(audio_dim, latent_dim)
        self.text_proj = nn.Linear(text_dim, latent_dim)
        self.vision_proj = nn.Linear(vision_dim, latent_dim)

    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            audio_features: (batch_size, audio_dim)
            text_features: (batch_size, text_dim)
            vision_features: (batch_size, vision_dim)

        Returns:
            A tuple of (reconstructed_features_dict, loss).
        """
        inputs = torch.cat([audio_features, text_features, vision_features], dim=-1)
        latent = self.encoder(inputs)
        recon = self.decoder(latent)

        audio_recon, text_recon, vision_recon = torch.split(
            recon, [self.audio_dim, self.text_dim, self.vision_dim], dim=-1
        )

        recon_loss = F.mse_loss(recon, inputs)

        audio_latent = self.audio_proj(audio_features)
        text_latent = self.text_proj(text_features)
        vision_latent = self.vision_proj(vision_features)

        align_at = 1 - F.cosine_similarity(audio_latent, text_latent, dim=-1).mean()
        align_av = 1 - F.cosine_similarity(audio_latent, vision_latent, dim=-1).mean()
        align_tv = 1 - F.cosine_similarity(text_latent, vision_latent, dim=-1).mean()
        align_loss = (align_at + align_av + align_tv) / 3.0

        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.alignment_weight * align_loss
        )

        outputs = {
            "audio": audio_recon,
            "text": text_recon,
            "vision": vision_recon,
            "latent": latent,
        }
        return outputs, total_loss


