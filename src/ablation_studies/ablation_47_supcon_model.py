"""
Ablation Study 47: Conflict-Aware + Supervised Contrastive Loss (SupCon) pre-fusion.

Extends Ablation 34 by adding a supervised contrastive loss term on
audio/text embeddings before Conflict-Aware Cross-Attention.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .ablation_34_model import Ablation34Model
from src.utils.losses import SupConLoss


class Ablation47Model(Ablation34Model):
    """
    Ablation 34 + optional supervised contrastive loss on audio/text embeddings.

    Forward returns loss_contrastive when config.training.loss.use_contrastive_loss
    is true and labels are provided. The loss is computed on L2-normalized
    audio/text pooled features before Conflict-Aware fusion.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        loss_cfg = config.get("training", {}).get("loss", {})
        temp = float(loss_cfg.get("contrastive_temperature", 0.1))
        self._contrastive_cfg = loss_cfg
        self._supcon_loss = SupConLoss(temperature=temp, base_temperature=temp)

    def forward(
        self,
        audio: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = super().forward(audio, text_input_ids, text_attention_mask, labels)

        use_cl = self._contrastive_cfg.get("use_contrastive_loss", False)
        if not use_cl or labels is None:
            out["loss_contrastive"] = None
            return out

        audio_feat = out.get("audio_features")
        text_feat = out.get("text_features")
        if audio_feat is None or text_feat is None:
            out["loss_contrastive"] = None
            return out

        # L2-normalize per-modality embeddings
        audio_norm = F.normalize(audio_feat, p=2, dim=-1)
        text_norm = F.normalize(text_feat, p=2, dim=-1)

        # Build features tensor with 2 views: (B, 2, D)
        features = torch.stack([audio_norm, text_norm], dim=1)

        loss_contrastive = self._supcon_loss(features, labels)
        out["loss_contrastive"] = loss_contrastive
        return out

