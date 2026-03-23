"""
Ablation Study 43: Conflict-Aware + Focal Loss + Supervised Contrastive Loss.

Extends Ablation 34 with:
- Focal Loss (config) for hard-class focus (e.g. Neutral).
- Supervised contrastive loss on modality embeddings (audio + text) before fusion
  to pull same-emotion samples together and push others apart.
- Config-driven: use_contrastive_loss, contrastive_weight, contrastive_temperature
  in training.loss; scheduler on val_ua.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ablation_34_model import Ablation34Model


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised Contrastive Loss (Khosla et al.).
    embeddings: (B, D), L2-normalized.
    labels: (B,) class indices.
    """
    B = embeddings.size(0)
    if B < 2:
        return embeddings.new_zeros(1).squeeze(0)

    # (B, B) pairwise logits
    logits = torch.mm(embeddings, embeddings.t()) / temperature

    # Mask self
    mask_self = torch.eye(B, dtype=torch.bool, device=embeddings.device)
    logits = logits.masked_fill(mask_self, float("-inf"))

    # For each anchor i, positives are j != i with labels[j] == labels[i]
    labels_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # (B, B)
    labels_eq = labels_eq.masked_fill(mask_self, 0.0)

    # Log-sum-exp over all j != i for denominator
    log_sum_exp = torch.logsumexp(logits, dim=1)  # (B,)

    # For each i, loss_i = -log( sum_{p in P(i)} exp(logits[i,p]) / sum_j exp(logits[i,j]) )
    exp_logits = torch.exp(logits)
    sum_pos = (exp_logits * labels_eq).sum(dim=1)  # (B,)
    n_pos = labels_eq.sum(dim=1).clamp(min=1e-9)  # (B,)
    log_sum_pos = torch.log(sum_pos.clamp(min=1e-9) + 1e-9)
    loss_per_anchor = log_sum_exp - log_sum_pos

    valid = (n_pos > 0.5).float()
    if valid.sum() < 1e-9:
        return embeddings.new_zeros(1).squeeze(0)
    loss = (loss_per_anchor * valid).sum() / valid.sum().clamp(min=1.0)
    return loss


class Ablation43Model(Ablation34Model):
    """
    Ablation 34 + optional supervised contrastive loss on modality embeddings.
    Forward returns loss_contrastive when config has use_contrastive_loss and labels are provided.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self._contrastive_cfg = config.get("training", {}).get("loss", {})

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

        # Single embedding per sample: mean of audio and text (same emotion)
        combined = (audio_feat + text_feat) * 0.5
        combined = F.normalize(combined, p=2, dim=-1)

        temp = float(self._contrastive_cfg.get("contrastive_temperature", 0.07))
        loss_contrastive = supervised_contrastive_loss(combined, labels, temperature=temp)
        out["loss_contrastive"] = loss_contrastive

        return out
