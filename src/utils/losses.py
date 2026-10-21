"""
Custom loss functions for emotion recognition.

Loss config contract (``training.loss`` in YAML):
    - ``type: cross_entropy``: uses ``nn.CrossEntropyLoss`` with optional ``class_weights``
      and ``label_smoothing``. The ``gamma`` field is **not** passed to PyTorch CE and has
      no effect; it exists for YAML symmetry with focal/composite recipes.
    - ``type: focal``: ``gamma`` is the focal focusing parameter (see ``FocalLoss``).
    - ``type: composite_loss``: ``gamma`` applies only to sub-components of type focal;
      see ``loss_config['components']``.

Training scripts call :func:`log_loss_contract_traceability` so ignored fields are visible
in logs (post-45 optimization spec).
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def log_loss_contract_traceability(
    logger: logging.Logger,
    loss_type: str,
    gamma: float,
    label_smoothing: float,
) -> None:
    """Log how ``gamma`` and ``label_smoothing`` interact with ``loss_type`` (startup traceability).

    For ``cross_entropy``, logs that ``gamma`` is unused and warns if ``gamma`` differs from
    the factory default (2.0), which often indicates a mistaken expectation of focal behavior.

    Args:
        logger: Training logger.
        loss_type: ``training.loss.type`` from config.
        gamma: ``training.loss.gamma`` from config.
        label_smoothing: ``training.loss.label_smoothing`` from config.
    """
    if loss_type == "cross_entropy":
        logger.info(
            "Loss contract: type=cross_entropy uses nn.CrossEntropyLoss(weight=class_weights, "
            "label_smoothing=%s). training.loss.gamma=%s is ignored for cross_entropy "
            "(gamma applies to focal and to focal_loss components inside composite_loss).",
            label_smoothing,
            gamma,
        )
        if gamma != 2.0:
            logger.warning(
                "training.loss.gamma=%s is set but loss.type is cross_entropy — gamma has no effect. "
                "Use loss.type: focal (or composite_loss with focal_loss) to apply focal gamma.",
                gamma,
            )
    elif loss_type == "focal":
        logger.info(
            "Loss contract: type=focal uses FocalLoss with gamma=%s, label_smoothing=%s.",
            gamma,
            label_smoothing,
        )
    elif loss_type == "composite_loss":
        logger.info(
            "Loss contract: type=composite_loss — focal gamma comes from each "
            "loss.components.*.gamma in YAML; top-level training.loss.gamma=%s is not passed "
            "into CompositeLoss (set per-component keys explicitly).",
            gamma,
        )
    else:
        logger.info(
            "Loss contract: type=%s — see create_loss_function; gamma=%s label_smoothing=%s.",
            loss_type,
            gamma,
            label_smoothing,
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Paper: https://arxiv.org/abs/1708.02002
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Args:
            alpha: Weighting factor for each class (1D tensor). If None, all classes have equal weight.
            gamma: Focusing parameter. Higher gamma focuses more on hard examples.
            reduction: 'none', 'mean', or 'sum'
            label_smoothing: Label smoothing factor (0.0 to 1.0)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Focal loss value
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            weight=self.alpha,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CrossEntropyLossAutoCast(nn.CrossEntropyLoss):
    """CrossEntropyLoss that auto-casts weight to match input dtype.

    Fixes RuntimeError when weight (float32) doesn't match logits dtype
    (bfloat16) under BF16 mixed precision.
    """

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.weight is not None and inputs.dtype != self.weight.dtype:
            self.weight = self.weight.to(inputs.dtype)
        return super().forward(inputs, targets)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.
    """
    
    def __init__(self, smoothing: float = 0.1):
        """
        Args:
            smoothing: Label smoothing factor (0.0 to 1.0)
        """
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Label smoothed cross entropy loss
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        num_classes = inputs.size(-1)
        
        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))



class DirichletEvidenceLoss(nn.Module):
    """
    Evidence Loss based on Dirichlet Distribution.
    
    Based on: "Evidential Deep Learning to Quantify Classification Uncertainty" (Sensoy et al., 2018)
    
    Minimizes the Bayes risk for the sum of squares error (Type 2 MLE).
    Includes KL divergence regularization to penalize misleading evidence.
    """
    
    def __init__(self, num_classes: int, annealing_step: int = 10):
        """
        Args:
            num_classes: Number of output classes
            annealing_step: Step at which KL regularization is fully applied
        """
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.epoch_num = 0  # To be updated during training
        
    def forward(self, alphas: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            alphas: Dirichlet parameters (batch_size, num_classes). Must be > 1.
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Evidence loss value
        """
        # One-hot encoding
        y = F.one_hot(targets, self.num_classes).to(alphas.device)
        
        # Calculate Dirichlet strength
        S = torch.sum(alphas, dim=1, keepdim=True)
        
        # Evidence and Probabilities
        # evidence = alphas - 1
        p = alphas / S
        
        # A: Sum of Squares Error Loss (Bayes Risk)
        # Loss = integral ||y - p||^2 dDir(p|alpha)
        #      = sum( (y_ij - alpha_ij/S_i)^2 + alpha_ij(S_i - alpha_ij) / (S_i^2 * (S_i + 1)) )
        
        err = (y - p) ** 2
        var = p * (1 - p) / (S + 1)
        loss_ace = torch.sum(err + var, dim=1)
        
        # B: KL Divergence Regularization
        # Penalizes evidence for incorrect classes
        # KL[ Dir(alpha) || Dir(1,1...1) ] where alpha_tilde = y + (1-y)*alpha
        
        alpha_tilde = y + (1 - y) * alphas
        
        # KL divergence between Dir(alpha_tilde) and Dir(1) (Uniform)
        # Simplified version for efficiency
        
        lgamma_alpha_tilde = torch.lgamma(alpha_tilde)
        lgamma_S_tilde = torch.lgamma(torch.sum(alpha_tilde, dim=1, keepdim=True))
        lgamma_1 = torch.lgamma(torch.tensor(1.0))
        lgamma_K = torch.lgamma(torch.tensor(float(self.num_classes)))
        
        # KL(Dir(alpha) || Dir(1)) formula:
        # log(Gamma(sum(alpha))) - log(Gamma(K)) - sum(log(Gamma(alpha))) + sum((alpha-1)*(digamma(alpha)-digamma(sum(alpha))))
        
        # Using the simplified approach from paper which minimizes evidence on incorrect classes
        # KL_reg = KL(Dir(alpha_tilde) || Dir(<1,1,1>))
        
        kl = lgamma_S_tilde - lgamma_K - torch.sum(lgamma_alpha_tilde, dim=1, keepdim=True) \
             + torch.sum(
                 (alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(torch.sum(alpha_tilde, dim=1, keepdim=True))),
                 dim=1, keepdim=True
             )
        
        # Annealing coefficient
        annealing_coef = min(1.0, max(0.0, self.epoch_num / self.annealing_step))
        
        loss = torch.mean(loss_ace + annealing_coef * kl.squeeze())
        return loss

    def set_epoch(self, epoch):
        self.epoch_num = epoch


class CompositeLoss(nn.Module):
    """
    Weighted sum of multiple loss components (e.g. cross_entropy + focal).
    """

    def __init__(self, components: list):
        """
        Args:
            components: List of (loss_module, weight) or dict with 'module' and 'weight'.
        """
        super().__init__()
        self.components = nn.ModuleList()
        self.weights = []
        for c in components:
            if isinstance(c, (list, tuple)):
                mod, w = c[0], c[1]
            else:
                mod, w = c["module"], c["weight"]
            self.components.append(mod)
            self.weights.append(w)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total = 0.0
        for mod, w in zip(self.components, self.weights):
            total = total + w * mod(inputs, targets)
        return total


def create_loss_function(
    loss_type: str = 'cross_entropy',
    class_weights: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
    device: str = 'cuda',
    num_classes: int = 4,
    loss_config: Optional[dict] = None,
) -> nn.Module:
    """
    Factory function to create loss function.

    Args:
        loss_type: Type of loss ('cross_entropy', 'focal', 'label_smoothing', 'composite_loss', ...)
        gamma: Focal focusing parameter; **only used** when ``loss_type=='focal'`` or inside
            ``composite_loss`` focal sub-modules. For plain ``cross_entropy``, this argument is
            ignored (callers should still pass config value and run
            :func:`log_loss_contract_traceability` for traceability).
        loss_config: Full loss config dict; required for composite_loss (must have 'components').
    """
    if loss_type == 'cross_entropy':
        if class_weights is not None:
            class_weights = class_weights.to(device)
        if label_smoothing > 0.0:
            return CrossEntropyLossAutoCast(weight=class_weights, label_smoothing=label_smoothing)
        return CrossEntropyLossAutoCast(weight=class_weights)
    
    elif loss_type == 'focal':
        if alpha is not None:
            alpha = alpha.to(device)
        return FocalLoss(alpha=alpha, gamma=gamma, label_smoothing=label_smoothing)
    
    elif loss_type == 'label_smoothing':
        if class_weights is not None:
            class_weights = class_weights.to(device)
            # For label smoothing with weights, we need to combine approaches
            # Here we use LabelSmoothingCrossEntropy and apply weights manually
            return LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        return LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    
    elif loss_type == 'dirichlet':
        # Evidence Loss
        return DirichletEvidenceLoss(num_classes=num_classes)
    
    elif loss_type == 'dirichlet_neutral':
        # Dirichlet Loss with neutral regularization
        from src.ablation_studies.ablation_36c_model import DirichletNeutralRegularizedLoss
        neutral_class_idx = 3  # Assuming neutral is the 4th class (0: angry, 1: happy, 2: sad, 3: neutral)
        neutral_weight = 2.0
        return DirichletNeutralRegularizedLoss(
            num_classes=num_classes,
            neutral_class_idx=neutral_class_idx,
            neutral_weight=neutral_weight
        )

    elif loss_type == 'composite_loss':
        if not loss_config or 'components' not in loss_config:
            raise ValueError("composite_loss requires loss_config with 'components'")
        comp_list = []
        for name, cfg in loss_config['components'].items():
            if not isinstance(cfg, dict):
                continue
            weight = cfg.get('weight', 1.0)
            if name == 'cross_entropy':
                ls = cfg.get('label_smoothing', 0.0)
                if class_weights is not None:
                    class_weights = class_weights.to(device)
                sub = nn.CrossEntropyLoss(
                    weight=class_weights,
                    label_smoothing=ls
                )
            elif name == 'focal_loss' and cfg.get('enabled', True):
                a = alpha
                if a is not None:
                    a = a.to(device)
                sub = FocalLoss(
                    alpha=a,
                    gamma=cfg.get('gamma', 2.0),
                    label_smoothing=cfg.get('label_smoothing', 0.0)
                )
            else:
                continue
            comp_list.append((sub, weight))
        if not comp_list:
            raise ValueError("composite_loss components must include at least cross_entropy or focal_loss")
        return CompositeLoss(comp_list)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for multimodal features.

    Expects multiple "views" (e.g. audio and text) per sample and pulls
    together embeddings from the same class while pushing apart different classes.
    """

    def __init__(self, temperature: float = 0.1, base_temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (batch_size, n_views, hidden_dim)
            labels:   Tensor of shape (batch_size,)
        """
        device = features.device
        bsz = features.size(0)
        n_views = features.size(1)

        if bsz <= 1:
            return features.new_zeros(())

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # (B, B)

        # Flatten views: (B * V, D)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # Similarity matrix scaled by temperature
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T),
            self.temperature,
        )

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask self-contrast
        logits_mask = torch.ones_like(anchor_dot_contrast, device=device)
        logits_mask = logits_mask.fill_(1.0)
        indices = torch.arange(bsz * n_views, device=device)
        logits_mask.scatter_(1, indices.view(-1, 1), 0.0)

        # Repeat mask for views
        mask = mask.repeat(n_views, n_views)
        mask = mask * logits_mask

        # Compute log-probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-20)

        # Mean of log-likelihood over positive pairs
        mask_sum = mask.sum(1)
        # Avoid division by zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum + 1e-20)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(n_views, bsz).mean()

        return loss
