"""
Audio fusion module with cross-attention.

Fuses Wav2Vec2 features with OpenSMILE features using cross-attention mechanism
and residual connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AudioCrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion for combining Wav2Vec2 and OpenSMILE features.
    
    Uses bidirectional cross-attention to allow both feature types to attend
    to each other, with residual connections for gradient flow.
    """
    
    def __init__(
        self,
        wav2vec2_dim: int,
        opensmile_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.3,
        use_residual: bool = True,
        residual_alpha: float = 0.2,
    ):
        """
        Args:
            wav2vec2_dim: Dimension of Wav2Vec2 features (typically 768)
            opensmile_dim: Dimension of OpenSMILE features (typically 88)
            hidden_dim: Common hidden dimension for fusion
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_residual: Whether to use residual connections
            residual_alpha: Weight for residual connection (0-1)
        """
        super().__init__()
        
        self.wav2vec2_dim = wav2vec2_dim
        self.opensmile_dim = opensmile_dim
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        self.residual_alpha = residual_alpha
        
        # Projections to common hidden dimension
        self.wav2vec2_proj = nn.Sequential(
            nn.Linear(wav2vec2_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        
        self.opensmile_proj = nn.Sequential(
            nn.Linear(opensmile_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        
        # Cross-attention: Wav2Vec2 attends to OpenSMILE
        self.wav2vec2_to_opensmile_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Cross-attention: OpenSMILE attends to Wav2Vec2
        self.opensmile_to_wav2vec2_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norms for residual connections
        self.wav2vec2_norm = nn.LayerNorm(hidden_dim)
        self.opensmile_norm = nn.LayerNorm(hidden_dim)
        
        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        wav2vec2_features: torch.Tensor,
        opensmile_features: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Fuse Wav2Vec2 and OpenSMILE features using cross-attention.
        
        Args:
            wav2vec2_features: Wav2Vec2 features (batch_size, wav2vec2_dim)
            opensmile_features: OpenSMILE features (batch_size, opensmile_dim)
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Fused features (batch_size, hidden_dim)
            Optional: Dictionary with attention weights
        """
        batch_size = wav2vec2_features.size(0)
        
        # Project to common dimension
        wav2vec2_proj = self.wav2vec2_proj(wav2vec2_features)  # (B, H)
        opensmile_proj = self.opensmile_proj(opensmile_features)  # (B, H)
        
        # Add sequence dimension for attention
        wav2vec2_seq = wav2vec2_proj.unsqueeze(1)  # (B, 1, H)
        opensmile_seq = opensmile_proj.unsqueeze(1)  # (B, 1, H)
        
        # Cross-attention: Wav2Vec2 attends to OpenSMILE
        wav2vec2_attended, wav2vec2_attn_weights = self.wav2vec2_to_opensmile_attn(
            query=wav2vec2_seq,
            key=opensmile_seq,
            value=opensmile_seq,
        )  # (B, 1, H)
        
        # Cross-attention: OpenSMILE attends to Wav2Vec2
        opensmile_attended, opensmile_attn_weights = self.opensmile_to_wav2vec2_attn(
            query=opensmile_seq,
            key=wav2vec2_seq,
            value=wav2vec2_seq,
        )  # (B, 1, H)
        
        # Residual connections with layer norm
        if self.use_residual:
            wav2vec2_attended = self.wav2vec2_norm(
                wav2vec2_attended + self.residual_alpha * wav2vec2_seq
            )
            opensmile_attended = self.opensmile_norm(
                opensmile_attended + self.residual_alpha * opensmile_seq
            )
        else:
            wav2vec2_attended = self.wav2vec2_norm(wav2vec2_attended)
            opensmile_attended = self.opensmile_norm(opensmile_attended)
        
        # Remove sequence dimension
        wav2vec2_attended = wav2vec2_attended.squeeze(1)  # (B, H)
        opensmile_attended = opensmile_attended.squeeze(1)  # (B, H)
        
        # Concatenate attended features
        fused = torch.cat([wav2vec2_attended, opensmile_attended], dim=-1)  # (B, 2*H)
        
        # Fusion MLP
        fused = self.fusion_mlp(fused)  # (B, H)
        
        # Final output projection with residual
        if self.use_residual:
            output = self.output_proj(fused) + self.residual_alpha * fused
        else:
            output = self.output_proj(fused)
        
        # Prepare return values
        if return_attention_weights:
            attn_dict = {
                'wav2vec2_to_opensmile': wav2vec2_attn_weights,
                'opensmile_to_wav2vec2': opensmile_attn_weights,
            }
            return output, attn_dict
        else:
            return output, None


class SimpleAudioFusion(nn.Module):
    """
    Simpler fusion module without cross-attention.
    
    Uses concatenation and MLP for faster computation.
    """
    
    def __init__(
        self,
        wav2vec2_dim: int,
        opensmile_dim: int,
        hidden_dim: int,
        dropout: float = 0.3,
        use_residual: bool = True,
    ):
        """
        Args:
            wav2vec2_dim: Dimension of Wav2Vec2 features
            opensmile_dim: Dimension of OpenSMILE features
            hidden_dim: Output hidden dimension
            dropout: Dropout rate
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.use_residual = use_residual
        
        # Projections
        self.wav2vec2_proj = nn.Linear(wav2vec2_dim, hidden_dim)
        self.opensmile_proj = nn.Linear(opensmile_dim, hidden_dim)
        
        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.8),
        )
    
    def forward(
        self,
        wav2vec2_features: torch.Tensor,
        opensmile_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse features using simple concatenation.
        
        Args:
            wav2vec2_features: (batch_size, wav2vec2_dim)
            opensmile_features: (batch_size, opensmile_dim)
            
        Returns:
            Fused features (batch_size, hidden_dim)
        """
        # Project to common dimension
        wav2vec2_proj = self.wav2vec2_proj(wav2vec2_features)
        opensmile_proj = self.opensmile_proj(opensmile_features)
        
        # Concatenate
        fused = torch.cat([wav2vec2_proj, opensmile_proj], dim=-1)
        
        # Fusion MLP
        output = self.fusion(fused)
        
        # Residual connection
        if self.use_residual:
            output = output + 0.2 * (wav2vec2_proj + opensmile_proj) / 2
        
        return output


