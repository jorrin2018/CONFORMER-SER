"""
Ablation Study 30: Dirichlet Evidence Learning
Base: Ablation 25 (Enhanced Cross-Attention)
Modificación: Output Layer & Loss para Uncertainty Estimation (Evidential Deep Learning)
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ablation_25_enhanced_cross_attention import Ablation25Model

class DirichletClassificationHead(nn.Module):
    """
    Classification head that outputs Dirichlet parameters (alphas) instead of logits.
    alpha = exp(logit) + 1
    """
    def __init__(
        self, 
        hidden_dim: int, 
        num_classes: int, 
        dropout: float = 0.3,
        num_layers: int = 2,
        bidirectional: bool = True
    ):
        super().__init__()
        
        # LSTM feature extractor (same as before)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        lstm_out_dim = hidden_dim if bidirectional else hidden_dim // 2
        
        # Final projection to alphas
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM
        # x shape: (batch, seq_len, hidden) or (batch, hidden) if already pooled
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        lstm_out, _ = self.lstm(x)
        # Take last state
        last_state = lstm_out[:, -1, :]
        
        # Logits
        logits = self.classifier(last_state)
        
        # Evidence activation: alpha = exp(logit) + 1
        # or Softplus + 1 for stability
        alphas = F.softplus(logits) + 1
        
        return alphas


class Ablation30Model(Ablation25Model):
    """
    Modelo trimodal con fusión cross-attention mejorada y salida Dirichlet.
    """

    def __init__(self, config: Dict):
        # Initialize as Ablation 25
        super().__init__(config)

        model_config = config.get("model", {})
        fusion_output_dim = self.hidden_dim // 2
        
        # Replace standard LSTM head with Dirichlet Head
        self.fusion.classifier = DirichletClassificationHead(
            hidden_dim=fusion_output_dim,
            num_classes=self.num_classes,
            dropout=model_config.get("dropout", 0.3),
            num_layers=model_config.get("lstm_num_layers", 2),
            bidirectional=model_config.get("lstm_bidirectional", True),
        )

