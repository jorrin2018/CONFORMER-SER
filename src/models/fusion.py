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

