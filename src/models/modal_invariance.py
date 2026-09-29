"""
Modal Invariance Constraint (MIC) using Symmetric KL Divergence.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalInvarianceConstraint(nn.Module):
    """
    Computes Symmetric KL Divergence between audio and text distributions.

    Args:
        temperature: Softmax temperature for smoothing.
        eps: Small constant to avoid log(0).
    """

    def __init__(self, temperature: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Args:
            audio_features: (batch_size, dim)
            text_features: (batch_size, dim)
            temperature: Optional override for temperature.

        Returns:
            Scalar symmetric KL divergence loss.
        """
        temp = temperature if temperature is not None else self.temperature

        audio_logits = audio_features / max(temp, self.eps)
        text_logits = text_features / max(temp, self.eps)

        log_p = F.log_softmax(audio_logits, dim=-1)
        log_q = F.log_softmax(text_logits, dim=-1)
        p = log_p.exp()
        q = log_q.exp()

        kl_pq = F.kl_div(log_p, q, reduction="batchmean")
        kl_qp = F.kl_div(log_q, p, reduction="batchmean")

        skl = 0.5 * (kl_pq + kl_qp)
        return skl


