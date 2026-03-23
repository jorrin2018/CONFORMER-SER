"""
Ablation Study 48: Conflict-Aware + Supervised Contrastive Loss (SupCon) - 3 views.

Extends Ablation 47 by adding vision embeddings to the contrastive loss.
SupCon is computed over audio + text + vision (3 views) instead of audio + text (2 views).
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .ablation_47_supcon_model import Ablation47Model


class Ablation48Model(Ablation47Model):
    """
    Ablation 47 + vision in SupCon: supervised contrastive loss on audio/text/vision.

    Forward returns loss_contrastive when config.training.loss.use_contrastive_loss
    is true and labels are provided. The loss is computed on L2-normalized
    audio/text/vision pooled features before Conflict-Aware fusion.
    """

    def forward(
        self,
        audio: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = super(Ablation47Model, self).forward(
            audio, text_input_ids, text_attention_mask, labels
        )

        use_cl = self._contrastive_cfg.get("use_contrastive_loss", False)
        if not use_cl or labels is None:
            out["loss_contrastive"] = None
            return out

        audio_feat = out.get("audio_features")
        text_feat = out.get("text_features")
        vision_feat = out.get("vision_features")

        if audio_feat is None or text_feat is None:
            out["loss_contrastive"] = None
            return out

        # L2-normalize per-modality embeddings
        audio_norm = F.normalize(audio_feat, p=2, dim=-1)
        text_norm = F.normalize(text_feat, p=2, dim=-1)

        if vision_feat is not None:
            # 3 views: audio + text + vision
            vision_norm = F.normalize(vision_feat, p=2, dim=-1)
            features = torch.stack([audio_norm, text_norm, vision_norm], dim=1)  # (B, 3, D)
        else:
            # Fallback to 2 views (e.g. bimodal path)
            features = torch.stack([audio_norm, text_norm], dim=1)  # (B, 2, D)

        loss_contrastive = self._supcon_loss(features, labels)
        out["loss_contrastive"] = loss_contrastive
        return out
