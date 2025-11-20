"""
Fusion modules for combining audio and text modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FusionModule(nn.Module):
    """
    Simple concatenation-based fusion module.
    """
    
    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3
    ):
        """
        Args:
            audio_dim: Dimension of audio features
            text_dim: Dimension of text features
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # Projection layers to common dimension
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Fusion layers - Improved MLP with GELU and LayerNorm (better for small batch sizes)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.6)
        )
        
        # Improved classification head with projection
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 4, num_classes)
        )
    
    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            audio_features: Audio features (batch_size, audio_dim)
            text_features: Text features (batch_size, text_dim)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Project to common dimension
        audio_emb = self.audio_proj(audio_features)
        text_emb = self.text_proj(text_features)
        
        # Concatenate
        fused = torch.cat([audio_emb, text_emb], dim=-1)
        
        # Fusion
        fused = self.fusion(fused)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention based fusion module.
    """
    
    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_heads: int = 8,
        dropout: float = 0.3
    ):
        """
        Args:
            audio_dim: Dimension of audio features
            text_dim: Dimension of text features
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # Projection layers
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-attention: audio attends to text
        self.audio_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention: text attends to audio
        self.text_to_audio_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.audio_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        
        # Fusion layers - Improved MLP with GELU and LayerNorm (better for small batch sizes)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.6)
        )
        
        # Improved classification head with projection
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 4, num_classes)
        )
    
    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with cross-attention.
        
        Args:
            audio_features: Audio features (batch_size, audio_dim)
            text_features: Text features (batch_size, text_dim)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Project to common dimension
        audio_emb = self.audio_proj(audio_features)
        text_emb = self.text_proj(text_features)
        
        # Add sequence dimension for attention (batch_size, 1, hidden_dim)
        audio_emb = audio_emb.unsqueeze(1)
        text_emb = text_emb.unsqueeze(1)
        
        # Cross-attention: audio attends to text
        audio_attended, _ = self.audio_to_text_attn(
            query=audio_emb,
            key=text_emb,
            value=text_emb
        )
        audio_attended = self.audio_norm(audio_attended + audio_emb)
        
        # Cross-attention: text attends to audio
        text_attended, _ = self.text_to_audio_attn(
            query=text_emb,
            key=audio_emb,
            value=audio_emb
        )
        text_attended = self.text_norm(text_attended + text_emb)
        
        # Remove sequence dimension
        audio_attended = audio_attended.squeeze(1)
        text_attended = text_attended.squeeze(1)
        
        # Concatenate attended features
        fused = torch.cat([audio_attended, text_attended], dim=-1)
        
        # Fusion
        fused = self.fusion(fused)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits


class GatedFusion(nn.Module):
    """
    Gated fusion module with learnable gating mechanism.
    """
    
    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3
    ):
        """
        Args:
            audio_dim: Dimension of audio features
            text_dim: Dimension of text features
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        # Projection layers
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # Fusion layers - Improved MLP with GELU and BatchNorm
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.6)
        )
        
        # Improved classification head with projection
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 4, num_classes)
        )
    
    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with gating.
        
        Args:
            audio_features: Audio features (batch_size, audio_dim)
            text_features: Text features (batch_size, text_dim)
            
        Returns:
            Tuple of (logits, gate_weights)
        """
        # Project to common dimension
        audio_emb = self.audio_proj(audio_features)
        text_emb = self.text_proj(text_features)
        
        # Compute gate weights
        concat = torch.cat([audio_emb, text_emb], dim=-1)
        gate_weights = self.gate(concat)  # (batch_size, 2)
        
        # Weighted combination
        fused = gate_weights[:, 0:1] * audio_emb + gate_weights[:, 1:2] * text_emb
        
        # Fusion
        fused = self.fusion(fused)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits, gate_weights


class GIAFusion(nn.Module):
    """
    Gated Interaction Attention (GIA) fusion module.

    Combines bidirectional cross-attention between audio and text with
    modality-specific gating mechanisms that control how much cross-modal
    information is injected into each stream.
    """

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_heads: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim

        # Projection layers to a common hidden dimension
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Bidirectional cross-attention
        self.audio_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_to_audio_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # LayerNorm for residual connections
        self.audio_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)

        # Gating networks per modality.
        # Each gate receives [self_representation, cross_representation]
        # and outputs values in (0, 1) that weight the mixture.
        def make_gate():
            return nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )

        self.audio_gate = make_gate()
        self.text_gate = make_gate()

        # Fusion MLP (kept compatible with CrossAttentionFusion so that
        # the output dimension is hidden_dim // 2 and downstream heads
        # like the LSTM used in ablation studies can be reused).
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
        )

        # Default classification head (can be replaced by ablation heads)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 4, num_classes),
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio_features: Audio features (batch_size, audio_dim)
            text_features: Text features (batch_size, text_dim)

        Returns:
            Logits (batch_size, num_classes)
        """
        # Project modalities to common space
        audio_emb = self.audio_proj(audio_features)  # (B, H)
        text_emb = self.text_proj(text_features)     # (B, H)

        # Add sequence dimension for attention
        audio_seq = audio_emb.unsqueeze(1)  # (B, 1, H)
        text_seq = text_emb.unsqueeze(1)    # (B, 1, H)

        # Cross-attention: audio attends to text and vice versa
        audio_ctx, _ = self.audio_to_text_attn(
            query=audio_seq,
            key=text_seq,
            value=text_seq,
        )
        text_ctx, _ = self.text_to_audio_attn(
            query=text_seq,
            key=audio_seq,
            value=audio_seq,
        )

        # Residual + LayerNorm (as in the GIA block)
        audio_ctx = self.audio_norm(audio_ctx + audio_seq)
        text_ctx = self.text_norm(text_ctx + text_seq)

        # Remove sequence dimension
        audio_ctx = audio_ctx.squeeze(1)  # (B, H)
        text_ctx = text_ctx.squeeze(1)    # (B, H)

        # Compute modality-specific gates
        audio_gate_input = torch.cat([audio_emb, audio_ctx], dim=-1)
        text_gate_input = torch.cat([text_emb, text_ctx], dim=-1)

        g_audio = self.audio_gate(audio_gate_input)  # (B, H) in (0, 1)
        g_text = self.text_gate(text_gate_input)     # (B, H) in (0, 1)

        # Gated interaction:
        # H_A^{GIA} = g * H_cross + (1 - g) * H_self (same for text)
        audio_gia = g_audio * audio_ctx + (1.0 - g_audio) * audio_emb
        text_gia = g_text * text_ctx + (1.0 - g_text) * text_emb

        # Concatenate refined representations and pass through fusion MLP
        fused = torch.cat([audio_gia, text_gia], dim=-1)
        fused = self.fusion(fused)

        # Classification
        logits = self.classifier(fused)

        return logits


class TriModalCrossAttentionFusion(nn.Module):
    """
    Cross-attention-style fusion for three modalities:
    audio, text, and vision (e.g., ViT over Mel-spectrogram images).

    The three modality embeddings are projected to a common space,
    jointly refined via self-attention, and then collapsed to a
    single fused representation that is compatible with the
    ablation heads (output dim = hidden_dim // 2, like CrossAttentionFusion/GIAFusion).
    """

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        vision_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_heads: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim

        # Projections to common hidden space
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)

        # Joint self-attention over the 3 modality tokens
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Fusion MLP:
        # input: concatenation of the 3 refined tokens => 3 * hidden_dim
        # output: hidden_dim // 2 (to stay compatible with ablation heads)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
        )

        # Default classifier (can be replaced by LSTM/Transformer heads in ablations)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 4, num_classes),
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio_features: (batch_size, audio_dim)
            text_features: (batch_size, text_dim)
            vision_features: (batch_size, vision_dim)

        Returns:
            Logits (batch_size, num_classes)
        """
        # Project each modality to common hidden space
        audio_emb = self.audio_proj(audio_features)   # (B, H)
        text_emb = self.text_proj(text_features)      # (B, H)
        vision_emb = self.vision_proj(vision_features)  # (B, H)

        # Stack as a sequence of 3 tokens
        tokens = torch.stack([audio_emb, text_emb, vision_emb], dim=1)  # (B, 3, H)

        # Self-attention over the 3 tokens (cross-modal interactions)
        attn_out, _ = self.self_attn(tokens, tokens, tokens)  # (B, 3, H)
        tokens = self.attn_norm(tokens + attn_out)

        # Flatten tokens and pass through fusion MLP
        fused = tokens.reshape(tokens.size(0), -1)  # (B, 3 * H)
        fused = self.fusion(fused)                  # (B, H//2)

        # Classification (can be overridden in ablation models)
        logits = self.classifier(fused)
        return logits

