"""Fusion modules for combining audio, text, and vision modalities.

This module provides various fusion strategies for multimodal emotion recognition.
ConflictAwareCrossAttentionFusion supports three conflict detection methods (set via
config `model.conflict_aware.conflict_detection_method`):

- prediction_divergence: Conflict when two or more modalities predict different
  classes (e.g. audio=angry, text=happy). Uses per-modality logits.
- feature_distance: Conflict when modality embeddings are far apart in representation
  space (mean pairwise cosine distance > conflict_threshold). Uses embeddings only.
- learned: A small MLP predicts conflict probability from concatenated embeddings;
  conflict when probability > conflict_threshold. Adds trainable parameters.
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
        
        # Projection layers to common dimension
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-attention: audio attends to text and vice versa
        self.audio_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.text_to_audio_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # LayerNorm for residual connections
        self.audio_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        
        # Fusion layers
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
        
        # Classification head
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
        
        # Add sequence dimension for attention
        audio_seq = audio_emb.unsqueeze(1)  # (B, 1, H)
        text_seq = text_emb.unsqueeze(1)   # (B, 1, H)
        
        # Cross-attention: audio attends to text and vice versa
        audio_ctx, _ = self.audio_to_text_attn(
            query=audio_seq,
            key=text_seq,
            value=text_seq
        )
        text_ctx, _ = self.text_to_audio_attn(
            query=text_seq,
            key=audio_seq,
            value=audio_seq
        )
        
        # Residual + LayerNorm
        audio_ctx = self.audio_norm(audio_ctx + audio_seq)
        text_ctx = self.text_norm(text_ctx + text_seq)
        
        # Remove sequence dimension
        audio_ctx = audio_ctx.squeeze(1)  # (B, H)
        text_ctx = text_ctx.squeeze(1)   # (B, H)
        
        # Concatenate and fuse
        fused = torch.cat([audio_ctx, text_ctx], dim=-1)
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
        
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        
        # Projection layers
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Fusion layers
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
        
        # Classification head
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
        
        # Compute gate
        gate_input = torch.cat([audio_emb, text_emb], dim=-1)
        gate_weights = self.gate(gate_input)
        
        # Gated fusion
        gated_audio = audio_emb * gate_weights
        gated_text = text_emb * (1 - gate_weights)
        
        # Concatenate and fuse
        fused = torch.cat([gated_audio, gated_text], dim=-1)
        fused = self.fusion(fused)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits


class GIAFusion(nn.Module):
    """
    Gated Interaction Attention (GIA) fusion module.

    Combines bidirectional cross-attention between audio and text with
    modality-specific gating mechanisms that control how much cross-modal
    information each modality incorporates.
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
        # and outputs a gating vector in (0, 1) to blend self and cross.
        def make_gate():
            return nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),  # Output in (0, 1)
            )

        self.audio_gate = make_gate()
        self.text_gate = make_gate()

        # Fusion MLP (kept compatible with CrossAttentionFusion so that
        # the output dimension is hidden_dim // 2 and downstream heads
        # can be reused)
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
        # Project to common space
        audio_emb = self.audio_proj(audio_features)  # (B, H)
        text_emb = self.text_proj(text_features)     # (B, H)

        # Add sequence dimension for attention
        audio_seq = audio_emb.unsqueeze(1)  # (B, 1, H)
        text_seq = text_emb.unsqueeze(1)   # (B, 1, H)

        # Cross-attention: audio attends to text and vice versa
        audio_ctx, _ = self.audio_to_text_attn(
            query=audio_seq,
            key=text_seq,
            value=text_seq
        )
        text_ctx, _ = self.text_to_audio_attn(
            query=text_seq,
            key=audio_seq,
            value=audio_seq
        )

        # Residual + LayerNorm (as in the GIA block)
        audio_ctx = self.audio_norm(audio_ctx + audio_seq)
        text_ctx = self.text_norm(text_ctx + text_seq)

        # Remove sequence dimension
        audio_ctx = audio_ctx.squeeze(1)  # (B, H)
        text_ctx = text_ctx.squeeze(1)   # (B, H)

        # Compute modality-specific gates
        audio_gate_input = torch.cat([audio_emb, audio_ctx], dim=-1)
        text_gate_input = torch.cat([text_emb, text_ctx], dim=-1)

        g_audio = self.audio_gate(audio_gate_input)  # (B, H) in (0, 1)
        g_text = self.text_gate(text_gate_input)     # (B, H) in (0, 1)

        # Gated interaction:
        # H_A^{GIA} = g * H_cross + (1 - g) * H_self (same for text)
        audio_gia = g_audio * audio_ctx + (1 - g_audio) * audio_emb
        text_gia = g_text * text_ctx + (1 - g_text) * text_emb

        # Concatenate refined representations and pass through fusion MLP
        fused = torch.cat([audio_gia, text_gia], dim=-1)
        fused = self.fusion(fused)

        # Classification
        logits = self.classifier(fused)

        return logits


class GatedInteractiveAttentionFusion(nn.Module):
    """
    Gated Interactive Attention Fusion (bimodal).

    Multi-layer bidirectional cross-attention between audio and text with
    modality-specific gating. Configurable via fusion_config (num_heads,
    num_interaction_layers, gate_activation, residual_connections, etc.).
    """

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_classes: int,
        fusion_config: Optional[dict] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        cfg = fusion_config or {}
        proj_dim = cfg.get("projection_dim", cfg.get("hidden_dim", hidden_dim))
        num_heads = cfg.get("num_heads", 8)
        num_layers = cfg.get("num_interaction_layers", 3)
        use_gated = cfg.get("use_gated_mechanism", True)
        gate_act = cfg.get("gate_activation", "sigmoid")
        bidirectional_gates = cfg.get("bidirectional_gates", True)
        use_residual = cfg.get("residual_connections", True)
        use_ln = cfg.get("layer_norm", True)
        attn_dropout = cfg.get("attention_dropout", 0.1)

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.hidden_dim = proj_dim
        self.num_layers = num_layers
        self.use_gated = use_gated
        self.use_residual = use_residual
        self.use_ln = use_ln

        self.audio_proj = nn.Linear(audio_dim, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)

        self.audio_norms = nn.ModuleList([nn.LayerNorm(proj_dim) for _ in range(num_layers)])
        self.text_norms = nn.ModuleList([nn.LayerNorm(proj_dim) for _ in range(num_layers)])

        self.audio_to_text_attns = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=proj_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.text_to_audio_attns = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=proj_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        if use_gated:
            act = nn.Sigmoid() if gate_act == "sigmoid" else nn.Tanh()
            def make_gate():
                layers = [
                    nn.Linear(proj_dim * 2, proj_dim),
                    nn.GELU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(proj_dim, proj_dim),
                    act,
                ]
                if use_ln:
                    layers.insert(1, nn.LayerNorm(proj_dim))
                return nn.Sequential(*layers)
            self.audio_gates = nn.ModuleList([make_gate() for _ in range(num_layers)])
            self.text_gates = nn.ModuleList([make_gate() for _ in range(num_layers)]) if bidirectional_gates else self.audio_gates
        else:
            self.audio_gates = None
            self.text_gates = None

        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.LayerNorm(proj_dim) if use_ln else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.LayerNorm(proj_dim // 2) if use_ln else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
        )
        fusion_out_dim = proj_dim // 2

        self.classifier = nn.Sequential(
            nn.Linear(fusion_out_dim, fusion_out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(fusion_out_dim // 2, num_classes),
        )

    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            audio_features: (batch_size, audio_dim) or (batch_size, seq_len, audio_dim)
            text_features: (batch_size, text_dim) or (batch_size, seq_len, text_dim)
        Returns:
            Logits (batch_size, num_classes)
        """
        if audio_features.dim() == 2:
            audio_emb = self.audio_proj(audio_features).unsqueeze(1)
            text_emb = self.text_proj(text_features).unsqueeze(1)
        else:
            audio_emb = self.audio_proj(audio_features)
            text_emb = self.text_proj(text_features)
        B = audio_emb.size(0)

        for i in range(self.num_layers):
            audio_ctx, _ = self.audio_to_text_attns[i](
                query=audio_emb, key=text_emb, value=text_emb
            )
            text_ctx, _ = self.text_to_audio_attns[i](
                query=text_emb, key=audio_emb, value=audio_emb
            )
            if self.use_residual:
                audio_ctx = audio_ctx + audio_emb
                text_ctx = text_ctx + text_emb
            if self.use_ln:
                audio_ctx = self.audio_norms[i](audio_ctx)
                text_ctx = self.text_norms[i](text_ctx)
            if self.use_gated and self.audio_gates is not None:
                a_cat = torch.cat([audio_emb, audio_ctx], dim=-1).squeeze(1)
                t_cat = torch.cat([text_emb, text_ctx], dim=-1).squeeze(1)
                g_a = self.audio_gates[i](a_cat).unsqueeze(1)
                gate_t = self.text_gates[i] if self.text_gates is not self.audio_gates else self.audio_gates[i]
                g_t = gate_t(t_cat).unsqueeze(1)
                audio_emb = g_a * audio_ctx + (1 - g_a) * audio_emb
                text_emb = g_t * text_ctx + (1 - g_t) * text_emb
            else:
                audio_emb = audio_ctx
                text_emb = text_ctx

        audio_emb = audio_emb.squeeze(1)
        text_emb = text_emb.squeeze(1)
        fused = torch.cat([audio_emb, text_emb], dim=-1)
        fused = self.fusion(fused)
        return self.classifier(fused)


class TriModalCrossAttentionFusion(nn.Module):
    """
    Cross-attention-style fusion for three modalities: audio, text, and vision.

    The three modality embeddings are projected to a common space,
    jointly refined via self-attention, and then collapsed to a
    single representation for classification.
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
        # output: hidden_dim // 2
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
        # Project to common space
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
        fused = self.fusion(fused)  # (B, H//2)

        # Classification (can be overridden in ablation models)
        logits = self.classifier(fused)

        return logits


class TriModalHierarchicalFusion(nn.Module):
    """
    Hierarchical fusion for three modalities: audio, text, and vision.

    First fuses audio and text, then fuses the result with vision.
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
            fused += audio_r * text_r * vision_r  # Element-wise product

        fused = fused + self.fusion_bias
        fused = self.fusion(fused)  # (B, H//2)

        # Classification
        logits = self.classifier(fused)
        return logits


class TriModalBilinearFusion(nn.Module):
    """
    Bilinear Pooling fusion for three modalities.
    
    Captures second-order interactions between pairs of modalities.
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
        self.audio_text_U = nn.Linear(hidden_dim, rank)
        self.audio_text_V = nn.Linear(hidden_dim, rank)
        self.audio_vision_U = nn.Linear(hidden_dim, rank)
        self.audio_vision_V = nn.Linear(hidden_dim, rank)
        self.text_vision_U = nn.Linear(hidden_dim, rank)
        self.text_vision_V = nn.Linear(hidden_dim, rank)

        # Combine pairwise interactions
        self.pairwise_fusion = nn.Sequential(
            nn.Linear(rank * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Final fusion MLP
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

        # Pairwise bilinear interactions
        # Audio-Text
        audio_text = (self.audio_text_U(audio_emb) * self.audio_text_V(text_emb))  # (B, rank)

        # Audio-Vision
        audio_vision = (self.audio_vision_U(audio_emb) * self.audio_vision_V(vision_emb))  # (B, rank)

        # Text-Vision
        text_vision = (self.text_vision_U(text_emb) * self.text_vision_V(vision_emb))  # (B, rank)

        # Combine pairwise interactions
        pairwise = torch.cat([audio_text, audio_vision, text_vision], dim=-1)  # (B, 3*rank)
        pairwise_fused = self.pairwise_fusion(pairwise)  # (B, H)

        # Final fusion
        fused = self.fusion(pairwise_fused)  # (B, H//2)

        # Classification
        logits = self.classifier(fused)
        return logits


class TriModalAdaptiveWeightedFusion(nn.Module):
    """
    Adaptive Weighted Fusion for three modalities.
    
    Learns scalar weights (via sigmoid) to dynamically assign importance
    to each modality.
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

        # Weight estimation network
        # Takes concatenated features and outputs 3 weights (one per modality)
        self.weight_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1),  # Normalize weights to sum to 1
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

        # Estimate adaptive weights
        all_features = torch.cat([audio_emb, text_emb, vision_emb], dim=-1)  # (B, 3*H)
        weights = self.weight_net(all_features)  # (B, 3)

        # Weighted combination
        weighted_audio = audio_emb * weights[:, 0:1]  # (B, H)
        weighted_text = text_emb * weights[:, 1:2]   # (B, H)
        weighted_vision = vision_emb * weights[:, 2:3]  # (B, H)

        # Sum weighted features
        fused = weighted_audio + weighted_text + weighted_vision  # (B, H)
        fused = self.fusion(fused)  # (B, H//2)

        # Classification
        logits = self.classifier(fused)
        return logits


class TriModalEnhancedCrossAttentionFusion(nn.Module):
    """
    Enhanced Cross-Attention Fusion for three modalities.
    
    Uses multiple layers of cross-attention for deeper interaction modeling.
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

        # Projections to common hidden space
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)

        # Multiple cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            ) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Final fusion
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

        # Stack modalities
        tokens = torch.stack([audio_emb, text_emb, vision_emb], dim=1)  # (B, 3, H)

        # Apply multiple cross-attention layers
        for attn, norm in zip(self.cross_attn_layers, self.layer_norms):
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
    Gated Multimodal Unit (GMU) for three modalities.
    
    Uses learnable gates to control the flow of information from each modality.
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

        # Gating networks for each modality
        # Each gate takes all modalities and outputs a gate value for that modality
        def make_gate():
            return nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim, 1),
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

        # Compute gates
        all_features = torch.cat([audio_emb, text_emb, vision_emb], dim=-1)  # (B, 3*H)
        audio_gate = self.audio_gate(all_features)  # (B, 1)
        text_gate = self.text_gate(all_features)   # (B, 1)
        vision_gate = self.vision_gate(all_features)  # (B, 1)

        # Apply gates
        gated_audio = audio_emb * audio_gate
        gated_text = text_emb * text_gate
        gated_vision = vision_emb * vision_gate

        # Concatenate and fuse
        fused = torch.cat([gated_audio, gated_text, gated_vision], dim=-1)  # (B, 3*H)
        fused = self.fusion(fused)  # (B, H//2)

        # Classification
        logits = self.classifier(fused)
        return logits


class TriModalLateFusion(nn.Module):
    """
    Late Fusion for three modalities.
    
    Processes each modality independently until the end, then combines
    logits/probabilities by averaging or concatenation.
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

        # GLU for each modality (ModuleDict so parameters register with the model)
        # GLU(x) = Linear1(x) ⊙ σ(Linear2(x))
        def make_glu():
            return nn.ModuleDict(
                {
                    "linear1": nn.Linear(hidden_dim, hidden_dim),
                    "linear2": nn.Linear(hidden_dim, hidden_dim),
                }
            )

        self.audio_glu = make_glu()
        self.text_glu = make_glu()
        self.vision_glu = make_glu()

        # Cross-attention over filtered features
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


class ConflictAwareCrossAttentionFusion(nn.Module):
    """
    Conflict-Aware Cross-Attention Fusion for three modalities.
    
    Detects when modalities contradict each other (e.g., happy text + angry audio)
    and assigns weights based on "confidence" instead of simply averaging.
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
        conflict_threshold: float = 0.3,
        # One of: "prediction_divergence" (logits disagree), "feature_distance" (embedding cosine distance), "learned" (MLP)
        conflict_detection_method: str = "prediction_divergence",
    ):
        super().__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        self.conflict_threshold = conflict_threshold
        self.conflict_detection_method = conflict_detection_method

        # Projections to common hidden space
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)

        # Individual classifiers for conflict detection
        # Initialize with small weights to prevent initial bias
        self.audio_classifier = nn.Linear(hidden_dim, num_classes)
        self.text_classifier = nn.Linear(hidden_dim, num_classes)
        self.vision_classifier = nn.Linear(hidden_dim, num_classes)
        
        # Initialize conflict detection classifiers with small weights
        for classifier in [self.audio_classifier, self.text_classifier, self.vision_classifier]:
            nn.init.normal_(classifier.weight, mean=0.0, std=0.01)
            if classifier.bias is not None:
                nn.init.normal_(classifier.bias, mean=0.0, std=0.01)

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            ) for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Confidence estimation network
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # Confidence scores for each modality
            nn.Sigmoid(),
        )

        # Learned conflict detector (only when conflict_detection_method == "learned")
        if conflict_detection_method == "learned":
            self.conflict_detector_net = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            for module in self.conflict_detector_net.modules():
                if isinstance(module, nn.Linear):
                    if module == list(self.conflict_detector_net.modules())[-2]:
                        nn.init.normal_(module.weight, mean=0.0, std=0.01)
                        if module.bias is not None:
                            nn.init.normal_(module.bias, mean=0.0, std=0.01)
                    else:
                        nn.init.xavier_uniform_(module.weight, gain=1.0)
                        if module.bias is not None:
                            nn.init.constant_(module.bias, 0.0)
        else:
            self.conflict_detector_net = None

        # Final fusion
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

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 4, num_classes),
        )
        
        # Initialize classifier with small weights to prevent class collapse
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classifier weights to prevent initial class collapse."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                # Last layer gets very small initialization
                if module == list(self.classifier.modules())[-1]:
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    if module.bias is not None:
                        nn.init.normal_(module.bias, mean=0.0, std=0.01)
                else:
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)

    def _detect_conflict_prediction_divergence(
        self,
        audio_logits: torch.Tensor,
        text_logits: torch.Tensor,
        vision_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Detect conflicts when two or more modalities predict different classes.

        Uses argmax of per-modality logits; conflict if at least two pairwise
        predictions disagree (e.g. audio=angry, text=happy).

        Args:
            audio_logits: Per-sample logits from audio branch. Shape (B, num_classes).
            text_logits: Per-sample logits from text branch. Shape (B, num_classes).
            vision_logits: Per-sample logits from vision branch. Shape (B, num_classes).

        Returns:
            Binary conflict scores. Shape (B,) with 1.0 if conflict, 0.0 otherwise.
        """
        audio_pred = torch.argmax(audio_logits, dim=-1)
        text_pred = torch.argmax(text_logits, dim=-1)
        vision_pred = torch.argmax(vision_logits, dim=-1)
        audio_text_disagree = (audio_pred != text_pred).float()
        audio_vision_disagree = (audio_pred != vision_pred).float()
        text_vision_disagree = (text_pred != vision_pred).float()
        # Conflict when at least two modality pairs disagree
        conflict = (audio_text_disagree + audio_vision_disagree + text_vision_disagree) >= 2.0
        return conflict.float()

    def _detect_conflict_feature_distance(
        self,
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor,
        vision_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Detect conflicts from embedding space: low similarity implies potential conflict.

        L2-normalizes embeddings, computes pairwise cosine similarity, converts to
        distance (1 - cos). Conflict when mean of the three pairwise distances
        exceeds conflict_threshold.

        Args:
            audio_emb: Audio embeddings. Shape (B, hidden_dim).
            text_emb: Text embeddings. Shape (B, hidden_dim).
            vision_emb: Vision embeddings. Shape (B, hidden_dim).

        Returns:
            Binary conflict scores. Shape (B,) with 1.0 if conflict, 0.0 otherwise.
        """
        a = F.normalize(audio_emb, p=2, dim=-1)
        t = F.normalize(text_emb, p=2, dim=-1)
        v = F.normalize(vision_emb, p=2, dim=-1)
        cos_at = (a * t).sum(dim=-1).clamp(-1.0, 1.0)
        cos_av = (a * v).sum(dim=-1).clamp(-1.0, 1.0)
        cos_tv = (t * v).sum(dim=-1).clamp(-1.0, 1.0)
        d_at = (1.0 - cos_at)
        d_av = (1.0 - cos_av)
        d_tv = (1.0 - cos_tv)
        mean_dist = (d_at + d_av + d_tv) / 3.0
        conflict = (mean_dist > self.conflict_threshold).float()
        return conflict

    def _detect_conflict_learned(
        self,
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor,
        vision_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Learned conflict detection: MLP predicts conflict probability from concatenated embeddings.

        conflict_detector_net is only created when conflict_detection_method == "learned"
        for checkpoint compatibility.

        Args:
            audio_emb: Audio embeddings. Shape (B, hidden_dim).
            text_emb: Text embeddings. Shape (B, hidden_dim).
            vision_emb: Vision embeddings. Shape (B, hidden_dim).

        Returns:
            Binary conflict scores. Shape (B,) with 1.0 if conflict, 0.0 otherwise.
        """
        x = torch.cat([audio_emb, text_emb, vision_emb], dim=-1)
        conflict_prob = self.conflict_detector_net(x).squeeze(-1)
        conflict = (conflict_prob > self.conflict_threshold).float()
        return conflict

    def _detect_conflict(
        self,
        audio_emb: torch.Tensor,
        text_emb: torch.Tensor,
        vision_emb: torch.Tensor,
        audio_logits: torch.Tensor,
        text_logits: torch.Tensor,
        vision_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch to the configured conflict detection method.

        Supported methods (set in config as model.conflict_aware.conflict_detection_method):
        - prediction_divergence: use when you want conflict = "modalities disagree on class".
        - feature_distance: use when you want conflict = "embeddings are far apart".
        - learned: use when you want the model to learn what counts as conflict.

        Args:
            audio_emb: Audio embeddings (B, hidden_dim). Used by feature_distance and learned.
            text_emb: Text embeddings (B, hidden_dim).
            vision_emb: Vision embeddings (B, hidden_dim).
            audio_logits: Audio classifier logits (B, num_classes). Used by prediction_divergence.
            text_logits: Text classifier logits (B, num_classes).
            vision_logits: Vision classifier logits (B, num_classes).

        Returns:
            Binary conflict mask. Shape (B,) with 1.0 if conflict, 0.0 otherwise.

        Raises:
            ValueError: If conflict_detection_method is not one of the three supported values.
        """
        if self.conflict_detection_method == "prediction_divergence":
            return self._detect_conflict_prediction_divergence(audio_logits, text_logits, vision_logits)
        elif self.conflict_detection_method == "feature_distance":
            return self._detect_conflict_feature_distance(audio_emb, text_emb, vision_emb)
        elif self.conflict_detection_method == "learned":
            return self._detect_conflict_learned(audio_emb, text_emb, vision_emb)
        else:
            raise ValueError(
                f"Unknown conflict_detection_method: {self.conflict_detection_method}. "
                "Use prediction_divergence, feature_distance, or learned."
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

        # Get individual predictions for conflict detection
        audio_logits = self.audio_classifier(audio_emb)  # (B, num_classes)
        text_logits = self.text_classifier(text_emb)     # (B, num_classes)
        vision_logits = self.vision_classifier(vision_emb)  # (B, num_classes)

        # Detect conflicts
        conflicts = self._detect_conflict(
            audio_emb, text_emb, vision_emb,
            audio_logits, text_logits, vision_logits,
        )  # (B,)

        # Stack modalities for cross-attention
        tokens = torch.stack([audio_emb, text_emb, vision_emb], dim=1)  # (B, 3, H)

        # Apply cross-attention layers
        for attn, norm in zip(self.cross_attn_layers, self.layer_norms):
            attn_out, _ = attn(tokens, tokens, tokens)  # (B, 3, H)
            tokens = norm(tokens + attn_out)

        # Estimate confidence for each modality
        tokens_flat = tokens.reshape(tokens.size(0), -1)  # (B, 3*H)
        confidence_scores = self.confidence_net(tokens_flat)  # (B, 3)

        # Apply conflict-aware weighting
        # When conflict detected, use confidence-based weighting
        # Otherwise, use uniform weighting
        base_weights = torch.ones_like(confidence_scores) / 3.0  # (B, 3)
        conflict_weights = confidence_scores / (confidence_scores.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Interpolate based on conflict detection
        conflict_mask = conflicts.unsqueeze(-1)  # (B, 1)
        final_weights = (1 - conflict_mask) * base_weights + conflict_mask * conflict_weights  # (B, 3)

        # Conflict-aware weights apply to tokens before the fusion MLP (not only unused sum)
        weighted_tokens = tokens * final_weights.unsqueeze(-1)  # (B, 3, H)
        tokens_concat = weighted_tokens.reshape(tokens.size(0), -1)  # (B, 3*H)
        final_fused = self.fusion(tokens_concat)  # (B, H//2)

        # Classification
        logits = self.classifier(final_fused)
        return logits
