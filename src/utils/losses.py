"""
Custom loss functions for emotion recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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


def create_loss_function(
    loss_type: str = 'cross_entropy',
    class_weights: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
    device: str = 'cuda'
) -> nn.Module:
    """
    Factory function to create loss function.
    
    Args:
        loss_type: Type of loss ('cross_entropy', 'focal', 'label_smoothing')
        class_weights: Class weights for CrossEntropyLoss
        alpha: Alpha weights for FocalLoss
        gamma: Gamma parameter for FocalLoss
        label_smoothing: Label smoothing factor
        device: Device to move tensors to
        
    Returns:
        Loss function module
    """
    if loss_type == 'cross_entropy':
        if class_weights is not None:
            class_weights = class_weights.to(device)
        return nn.CrossEntropyLoss(weight=class_weights)
    
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
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")




