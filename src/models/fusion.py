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


class TriModalHierarchicalFusion(nn.Module):
    """
    Hierarchical fusion for three modalities: audio, text, and vision.
    
    Fusion progresses in multiple levels:
    1. First fuses audio + text
    2. Then fuses the result with vision
    Each level uses cross-attention.
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

        # Level 1: Audio-Text cross-attention
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
        self.level1_norm_audio = nn.LayerNorm(hidden_dim)
        self.level1_norm_text = nn.LayerNorm(hidden_dim)

        # Level 1 fusion: audio + text -> fused_audio_text
        self.level1_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Level 2: Fused audio-text with vision cross-attention
        self.fused_to_vision_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.vision_to_fused_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.level2_norm_fused = nn.LayerNorm(hidden_dim)
        self.level2_norm_vision = nn.LayerNorm(hidden_dim)

        # Final fusion MLP
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

        # Default classifier
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
        # Project to common space
        audio_emb = self.audio_proj(audio_features)   # (B, H)
        text_emb = self.text_proj(text_features)      # (B, H)
        vision_emb = self.vision_proj(vision_features)  # (B, H)

        # Level 1: Audio-Text fusion
        audio_seq = audio_emb.unsqueeze(1)  # (B, 1, H)
        text_seq = text_emb.unsqueeze(1)   # (B, 1, H)

        audio_attended, _ = self.audio_to_text_attn(
            query=audio_seq, key=text_seq, value=text_seq
        )
        text_attended, _ = self.text_to_audio_attn(
            query=text_seq, key=audio_seq, value=audio_seq
        )

        audio_attended = self.level1_norm_audio(audio_attended + audio_seq)
        text_attended = self.level1_norm_text(text_attended + text_seq)

        audio_attended = audio_attended.squeeze(1)  # (B, H)
        text_attended = text_attended.squeeze(1)   # (B, H)

        # Fuse audio and text
        fused_audio_text = self.level1_fusion(
            torch.cat([audio_attended, text_attended], dim=-1)
        )  # (B, H)

        # Level 2: Fused audio-text with vision
        fused_seq = fused_audio_text.unsqueeze(1)  # (B, 1, H)
        vision_seq = vision_emb.unsqueeze(1)       # (B, 1, H)

        fused_attended, _ = self.fused_to_vision_attn(
            query=fused_seq, key=vision_seq, value=vision_seq
        )
        vision_attended, _ = self.vision_to_fused_attn(
            query=vision_seq, key=fused_seq, value=fused_seq
        )

        fused_attended = self.level2_norm_fused(fused_attended + fused_seq)
        vision_attended = self.level2_norm_vision(vision_attended + vision_seq)

        fused_attended = fused_attended.squeeze(1)  # (B, H)
        vision_attended = vision_attended.squeeze(1)  # (B, H)

        # Final fusion
        final_fused = torch.cat([fused_attended, vision_attended], dim=-1)
        final_fused = self.fusion(final_fused)  # (B, H//2)

        # Classification
        logits = self.classifier(final_fused)
        return logits


class TriModalTensorFusion(nn.Module):
    """
    Low-Rank Tensor Fusion for three modalities.
    
    Captures complex interactions between modalities using tensor factorization
    for computational efficiency.
    """

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        vision_dim: int,
        hidden_dim: int,
        num_classes: int,
        rank: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        self.rank = rank

        # Projections to common hidden space
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)

        # Low-rank tensor fusion factors
        # For 3 modalities, we use factorized bilinear pooling
        # Output = sum over rank: (W_a^r @ audio) * (W_t^r @ text) * (W_v^r @ vision)
        self.audio_factors = nn.Parameter(torch.randn(rank, hidden_dim, hidden_dim))
        self.text_factors = nn.Parameter(torch.randn(rank, hidden_dim, hidden_dim))
        self.vision_factors = nn.Parameter(torch.randn(rank, hidden_dim, hidden_dim))
        self.fusion_bias = nn.Parameter(torch.zeros(hidden_dim))

        # Initialize factors
        nn.init.xavier_uniform_(self.audio_factors)
        nn.init.xavier_uniform_(self.text_factors)
        nn.init.xavier_uniform_(self.vision_factors)

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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

        # Default classifier
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
        # Project to common space
        audio_emb = self.audio_proj(audio_features)   # (B, H)
        text_emb = self.text_proj(text_features)      # (B, H)
        vision_emb = self.vision_proj(vision_features)  # (B, H)

        # Low-rank tensor fusion
        # For each rank r: (W_a^r @ audio) * (W_t^r @ text) * (W_v^r @ vision)
        fused = torch.zeros(audio_emb.size(0), self.hidden_dim, device=audio_emb.device)

        for r in range(self.rank):
            audio_r = torch.matmul(audio_emb, self.audio_factors[r])  # (B, H)
            text_r = torch.matmul(text_emb, self.text_factors[r])    # (B, H)
            vision_r = torch.matmul(vision_emb, self.vision_factors[r])  # (B, H)
            fused += audio_r * text_r * vision_r

        fused = fused + self.fusion_bias
        fused = self.fusion(fused)  # (B, H//2)

        # Classification
        logits = self.classifier(fused)
        return logits


class TriModalBilinearFusion(nn.Module):
    """
    Bilinear Pooling fusion for three modalities.
    
    Captures second-order interactions between modalities using
    factorized bilinear pooling for efficiency.
    """

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        vision_dim: int,
        hidden_dim: int,
        num_classes: int,
        rank: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        self.rank = rank

        # Projections to common hidden space
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)

        # Factorized bilinear pooling
        # For pairwise interactions: audio-text, audio-vision, text-vision
        # Each uses low-rank bilinear: (U @ x) * (V @ y)
        self.audio_text_U = nn.Linear(hidden_dim, rank, bias=False)
        self.audio_text_V = nn.Linear(hidden_dim, rank, bias=False)
        self.audio_vision_U = nn.Linear(hidden_dim, rank, bias=False)
        self.audio_vision_V = nn.Linear(hidden_dim, rank, bias=False)
        self.text_vision_U = nn.Linear(hidden_dim, rank, bias=False)
        self.text_vision_V = nn.Linear(hidden_dim, rank, bias=False)

        # Combine pairwise interactions
        self.pairwise_fusion = nn.Sequential(
            nn.Linear(rank * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Final fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),  # 3 pairwise + 1 direct
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

        # Default classifier
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
        # Project to common space
        audio_emb = self.audio_proj(audio_features)   # (B, H)
        text_emb = self.text_proj(text_features)      # (B, H)
        vision_emb = self.vision_proj(vision_features)  # (B, H)

        # Pairwise bilinear interactions
        # Audio-Text
        audio_text = (self.audio_text_U(audio_emb) * self.audio_text_V(text_emb))  # (B, rank)

        # Audio-Vision
        audio_vision = (self.audio_vision_U(audio_emb) * self.audio_vision_V(vision_emb))  # (B, rank)

        # Text-Vision
        text_vision = (self.text_vision_U(text_emb) * self.text_vision_V(vision_emb))  # (B, rank)

        # Combine pairwise interactions
        pairwise = torch.cat([audio_text, audio_vision, text_vision], dim=-1)  # (B, rank*3)
        pairwise_fused = self.pairwise_fusion(pairwise)  # (B, H)

        # Combine with direct features
        all_features = torch.cat([audio_emb, text_emb, vision_emb, pairwise_fused], dim=-1)  # (B, 4*H)
        final_fused = self.fusion(all_features)  # (B, H//2)

        # Classification
        logits = self.classifier(final_fused)
        return logits


class TriModalAdaptiveWeightedFusion(nn.Module):
    """
    Adaptive Weighted Fusion for three modalities.
    
    Learns dynamic weights for each modality using attention mechanism,
    then combines weighted features.
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

        # Attention mechanism to compute adaptive weights
        # Query: learnable context vector
        # Keys/Values: modality embeddings
        self.context_vector = nn.Parameter(torch.randn(1, 1, hidden_dim))
        nn.init.xavier_uniform_(self.context_vector)

        self.weight_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.weight_norm = nn.LayerNorm(hidden_dim)

        # Weight projection to get 3 weights (one per modality)
        self.weight_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1),
        )

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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

        # Default classifier
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
        # Project to common space
        audio_emb = self.audio_proj(audio_features)   # (B, H)
        text_emb = self.text_proj(text_features)      # (B, H)
        vision_emb = self.vision_proj(vision_features)  # (B, H)

        # Stack modalities as sequence
        modalities = torch.stack([audio_emb, text_emb, vision_emb], dim=1)  # (B, 3, H)

        # Compute adaptive weights using attention
        batch_size = audio_emb.size(0)
        context = self.context_vector.expand(batch_size, -1, -1)  # (B, 1, H)

        attended, _ = self.weight_attention(
            query=context,
            key=modalities,
            value=modalities,
        )
        attended = self.weight_norm(attended + context)
        attended = attended.squeeze(1)  # (B, H)

        # Get weights for each modality
        weights = self.weight_proj(attended)  # (B, 3)

        # Weighted combination
        weighted_audio = weights[:, 0:1] * audio_emb
        weighted_text = weights[:, 1:2] * text_emb
        weighted_vision = weights[:, 2:3] * vision_emb

        fused = weighted_audio + weighted_text + weighted_vision  # (B, H)
        fused = self.fusion(fused)  # (B, H//2)

        # Classification
        logits = self.classifier(fused)
        return logits


class TriModalEnhancedCrossAttentionFusion(nn.Module):
    """
    Enhanced Cross-Attention Fusion with multiple layers.
    
    Uses a stack of cross-attention layers with residual connections
    and layer normalization for deeper interaction modeling.
    """

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        vision_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Projections to common hidden space
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)

        # Stack of cross-attention layers
        self.attention_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            # Self-attention over 3 modalities
            self.attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Fusion MLP
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

        # Default classifier
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
        # Project to common space
        audio_emb = self.audio_proj(audio_features)   # (B, H)
        text_emb = self.text_proj(text_features)      # (B, H)
        vision_emb = self.vision_proj(vision_features)  # (B, H)

        # Stack as sequence
        tokens = torch.stack([audio_emb, text_emb, vision_emb], dim=1)  # (B, 3, H)

        # Apply multiple attention layers with residuals
        for attn, norm in zip(self.attention_layers, self.norms):
            attn_out, _ = attn(tokens, tokens, tokens)  # (B, 3, H)
            tokens = norm(tokens + attn_out)

        # Flatten and fuse
        fused = tokens.reshape(tokens.size(0), -1)  # (B, 3*H)
        fused = self.fusion(fused)  # (B, H//2)

        # Classification
        logits = self.classifier(fused)
        return logits


class TriModalGatedFusion(nn.Module):
    """
    Gated Fusion for three modalities.
    
    Extension of GIA to three modalities with gating mechanisms
    for adaptive information flow control.
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

        # Cross-attention between modalities
        self.audio_to_others_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_to_others_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.vision_to_others_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norms
        self.audio_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.vision_norm = nn.LayerNorm(hidden_dim)

        # Gating networks for each modality
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
        self.vision_gate = make_gate()

        # Fusion MLP
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

        # Default classifier
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
        # Project to common space
        audio_emb = self.audio_proj(audio_features)   # (B, H)
        text_emb = self.text_proj(text_features)      # (B, H)
        vision_emb = self.vision_proj(vision_features)  # (B, H)

        # Prepare sequences
        audio_seq = audio_emb.unsqueeze(1)  # (B, 1, H)
        text_seq = text_emb.unsqueeze(1)   # (B, 1, H)
        vision_seq = vision_emb.unsqueeze(1)  # (B, 1, H)

        # Stack other modalities for cross-attention
        others = torch.stack([text_emb, vision_emb], dim=1)  # (B, 2, H) for audio
        text_others = torch.stack([audio_emb, vision_emb], dim=1)  # (B, 2, H) for text
        vision_others = torch.stack([audio_emb, text_emb], dim=1)  # (B, 2, H) for vision

        # Cross-attention
        audio_ctx, _ = self.audio_to_others_attn(
            query=audio_seq, key=others, value=others
        )
        text_ctx, _ = self.text_to_others_attn(
            query=text_seq, key=text_others, value=text_others
        )
        vision_ctx, _ = self.vision_to_others_attn(
            query=vision_seq, key=vision_others, value=vision_others
        )

        # Residual + norm
        audio_ctx = self.audio_norm(audio_ctx + audio_seq)
        text_ctx = self.text_norm(text_ctx + text_seq)
        vision_ctx = self.vision_norm(vision_ctx + vision_seq)

        # Remove sequence dimension
        audio_ctx = audio_ctx.squeeze(1)  # (B, H)
        text_ctx = text_ctx.squeeze(1)    # (B, H)
        vision_ctx = vision_ctx.squeeze(1)  # (B, H)

        # Compute gates
        audio_gate_input = torch.cat([audio_emb, audio_ctx], dim=-1)
        text_gate_input = torch.cat([text_emb, text_ctx], dim=-1)
        vision_gate_input = torch.cat([vision_emb, vision_ctx], dim=-1)

        g_audio = self.audio_gate(audio_gate_input)
        g_text = self.text_gate(text_gate_input)
        g_vision = self.vision_gate(vision_gate_input)

        # Gated combination
        audio_gated = g_audio * audio_ctx + (1.0 - g_audio) * audio_emb
        text_gated = g_text * text_ctx + (1.0 - g_text) * text_emb
        vision_gated = g_vision * vision_ctx + (1.0 - g_vision) * vision_emb

        # Concatenate and fuse
        fused = torch.cat([audio_gated, text_gated, vision_gated], dim=-1)
        fused = self.fusion(fused)  # (B, H//2)

        # Classification
        logits = self.classifier(fused)
        return logits


class TriModalLateFusion(nn.Module):
    """
    Late Fusion for three modalities.
    
    Each modality processes independently through its own classifier,
    then final logits are combined via weighted voting or averaging.
    """

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        vision_dim: int,
        hidden_dim: int,
        num_classes: int,
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

        # Individual classifiers for each modality
        def make_classifier():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.8),
                nn.Linear(hidden_dim // 2, num_classes),
            )

        self.audio_classifier = make_classifier()
        self.text_classifier = make_classifier()
        self.vision_classifier = make_classifier()

        # Learnable weights for combining logits
        self.combination_weights = nn.Parameter(torch.ones(3) / 3.0)

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
        # Project to common space
        audio_emb = self.audio_proj(audio_features)   # (B, H)
        text_emb = self.text_proj(text_features)      # (B, H)
        vision_emb = self.vision_proj(vision_features)  # (B, H)

        # Individual classification
        audio_logits = self.audio_classifier(audio_emb)  # (B, num_classes)
        text_logits = self.text_classifier(text_emb)     # (B, num_classes)
        vision_logits = self.vision_classifier(vision_emb)  # (B, num_classes)

        # Normalize combination weights
        weights = F.softmax(self.combination_weights, dim=0)

        # Weighted combination of logits
        combined_logits = (
            weights[0] * audio_logits +
            weights[1] * text_logits +
            weights[2] * vision_logits
        )

        return combined_logits


class TriModalGLUFusion(nn.Module):
    """
    GLU (Gated Linear Unit) Fusion for three modalities.
    
    Applies GLU to each modality to filter and select the most relevant
    features before fusion. GLU formula: GLU(x) = (W1*x + b1) ⊙ σ(W2*x + b2)
    where ⊙ is element-wise product.
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

        # GLU for each modality
        # GLU(x) = Linear1(x) ⊙ σ(Linear2(x))
        def make_glu():
            return nn.ModuleDict({
                'linear1': nn.Linear(hidden_dim, hidden_dim),
                'linear2': nn.Linear(hidden_dim, hidden_dim),
            })

        self.audio_glu = make_glu()
        self.text_glu = make_glu()
        self.vision_glu = make_glu()

        # Cross-attention after GLU filtering
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # Fusion MLP
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

        # Default classifier
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
        # Project to common space
        audio_emb = self.audio_proj(audio_features)   # (B, H)
        text_emb = self.text_proj(text_features)      # (B, H)
        vision_emb = self.vision_proj(vision_features)  # (B, H)

        # Apply GLU to filter features
        # GLU(x) = Linear1(x) ⊙ σ(Linear2(x))
        audio_filtered = self.audio_glu['linear1'](audio_emb) * torch.sigmoid(
            self.audio_glu['linear2'](audio_emb)
        )
        text_filtered = self.text_glu['linear1'](text_emb) * torch.sigmoid(
            self.text_glu['linear2'](text_emb)
        )
        vision_filtered = self.vision_glu['linear1'](vision_emb) * torch.sigmoid(
            self.vision_glu['linear2'](vision_emb)
        )

        # Cross-attention over filtered features
        tokens = torch.stack([audio_filtered, text_filtered, vision_filtered], dim=1)  # (B, 3, H)
        attn_out, _ = self.cross_attn(tokens, tokens, tokens)  # (B, 3, H)
        tokens = self.attn_norm(tokens + attn_out)

        # Flatten and fuse
        fused = tokens.reshape(tokens.size(0), -1)  # (B, 3*H)
        fused = self.fusion(fused)  # (B, H//2)

        # Classification
        logits = self.classifier(fused)
        return logits

