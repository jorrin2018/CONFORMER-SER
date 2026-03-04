"""
Attention-based gates for feature selection.

Uses multi-head attention mechanism to learn which features are important
and applies soft selection based on attention weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AttentionBasedGates(nn.Module):
    """
    Attention-based gates for filtering and selecting important features.
    
    Uses multi-head self-attention to learn feature importance and applies
    soft gating based on attention weights.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        dropout: float = 0.3,
        use_layer_norm: bool = True,
        temperature: float = 1.0,
    ):
        """
        Args:
            input_dim: Dimension of input features
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
            temperature: Temperature for attention softmax (lower = sharper)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.temperature = temperature
        
        # Ensure input_dim is divisible by num_heads
        assert input_dim % num_heads == 0, \
            f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
        
        self.head_dim = input_dim // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        
        # Output projection
        self.out_proj = nn.Linear(input_dim, input_dim)
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)
        else:
            self.layer_norm = None
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Gate generation network
        # Takes attention-weighted features and generates gate values
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid(),  # Gate values in [0, 1]
        )
        
        # Initialize weights
        self._reset_parameters()
        
        # Initialize gate network weights
        for module in self.gate_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Smaller gain for gates
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0.0)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0.0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
    
    def forward(
        self,
        features: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply attention-based gating to features.
        
        Args:
            features: Input features (batch_size, feature_dim)
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Filtered features (batch_size, feature_dim)
            Optional: Attention weights (batch_size, num_heads, feature_dim, feature_dim)
        """
        batch_size = features.size(0)
        feature_dim = features.size(1)
        
        # Store residual
        residual = features
        
        # Apply layer norm if enabled
        if self.layer_norm is not None:
            features = self.layer_norm(features)
        
        # Project to Q, K, V
        q = self.q_proj(features)  # (B, D)
        k = self.k_proj(features)  # (B, D)
        v = self.v_proj(features)  # (B, D)
        
        # Reshape for multi-head attention
        # Treat each feature as a token
        # We'll use self-attention over features
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, d)
        k = k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, d)
        v = v.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, d)
        
        # Compute attention scores
        # Since we have single token, we compute attention over feature dimensions
        # We'll use a different approach: compute importance scores for each feature
        # by attending to a learnable context vector
        
        # Alternative: Compute feature importance via self-attention
        # Reshape features to treat each dimension as a token
        q = features.unsqueeze(1)  # (B, 1, D)
        k = features.unsqueeze(1)  # (B, 1, D)
        v = features.unsqueeze(1)  # (B, 1, D)
        
        # Project and reshape for multi-head
        q = self.q_proj(q).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, d)
        k = self.k_proj(k).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, d)
        v = self.v_proj(v).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, d)
        
        # Compute attention scores (scaled dot-product attention)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5 * self.temperature)  # (B, H, 1, 1)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (B, H, 1, d)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, 1, H, d)
        attn_output = attn_output.view(batch_size, feature_dim)  # (B, D)
        
        # Project output
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout_layer(attn_output)
        
        # Residual connection
        attn_output = attn_output + residual
        
        # Generate gates based on attention-weighted features
        gate_values = self.gate_net(attn_output)  # (B, D) in [0, 1]
        
        # Apply gates to original features (soft selection)
        gated_features = features * gate_values
        
        if return_attention_weights:
            # Return attention weights (averaged over heads)
            attn_weights_avg = attn_weights.mean(dim=1).squeeze(1)  # (B, 1, 1)
            return gated_features, attn_weights_avg
        else:
            return gated_features, None


class FeatureImportanceGates(nn.Module):
    """
    Simplified attention-based gates that directly compute feature importance.
    
    More efficient version that computes importance scores directly without
    full self-attention mechanism.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden dimension for importance network (default: input_dim)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim
        
        # Importance scoring network
        self.importance_net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(self.hidden_dim // 2, input_dim),
            nn.Sigmoid(),  # Importance scores in [0, 1]
        )
        
        # Feature transformation (optional)
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Smaller gain
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        features: torch.Tensor,
        return_gates: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply importance-based gating.
        
        Args:
            features: Input features (batch_size, feature_dim)
            return_gates: Whether to return gate values
            
        Returns:
            Filtered features (batch_size, feature_dim)
            Optional: Gate values (batch_size, feature_dim)
        """
        # Compute importance scores (gates)
        gate_values = self.importance_net(features)  # (B, D)
        
        # Transform features
        transformed = self.feature_transform(features)
        
        # Apply gates
        gated_features = transformed * gate_values
        
        if return_gates:
            return gated_features, gate_values
        else:
            return gated_features, None


