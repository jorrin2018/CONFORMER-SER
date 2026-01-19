"""
Ablation Study 36C: Conflict-Aware + Dirichlet Hybrid with Neutral Regularization
Base: Ablation 34 (Conflict-Aware Cross-Attention)
Modificación: Dirichlet Classification Head + Dirichlet Loss con regularización adicional para neutral
Objetivo: Combinar Dirichlet con enfoque específico para mejorar neutral
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ablation_34_model import Ablation34Model
from .ablation_30_dirichlet import DirichletClassificationHead
from src.utils.losses import DirichletEvidenceLoss


class DirichletNeutralRegularizedLoss(DirichletEvidenceLoss):
    """
    Dirichlet Evidence Loss con regularización adicional para la clase neutral.
    Aumenta la penalización cuando neutral es mal clasificada.
    """
    
    def __init__(self, num_classes: int, annealing_step: int = 10, neutral_class_idx: int = 3, neutral_weight: float = 2.0):
        """
        Args:
            num_classes: Number of output classes
            annealing_step: Step at which KL regularization is fully applied
            neutral_class_idx: Index of neutral class (default 3 for angry, happy, sad, neutral)
            neutral_weight: Weight multiplier for neutral class errors
        """
        super().__init__(num_classes, annealing_step)
        self.neutral_class_idx = neutral_class_idx
        self.neutral_weight = neutral_weight
    
    def forward(self, alphas: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            alphas: Dirichlet parameters (batch_size, num_classes). Must be > 1.
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Evidence loss value with additional neutral regularization
        """
        # Get base Dirichlet loss
        base_loss = super().forward(alphas, targets)
        
        # One-hot encoding
        y = F.one_hot(targets, self.num_classes).to(alphas.device)
        
        # Calculate Dirichlet strength
        S = torch.sum(alphas, dim=1, keepdim=True)
        p = alphas / S
        
        # Additional penalty for neutral class misclassification
        # Focus on cases where neutral is the true class or predicted class
        neutral_mask = (targets == self.neutral_class_idx).float()
        
        # Error for neutral class specifically
        neutral_error = (y[:, self.neutral_class_idx] - p[:, self.neutral_class_idx]) ** 2
        
        # Weighted neutral error
        neutral_penalty = self.neutral_weight * neutral_mask * neutral_error
        
        # Also penalize when neutral should be predicted but isn't
        neutral_predicted = (torch.argmax(p, dim=1) == self.neutral_class_idx).float()
        neutral_missed = neutral_mask * (1 - neutral_predicted)
        missed_penalty = self.neutral_weight * 0.5 * neutral_missed
        
        # Combine losses
        total_loss = base_loss + torch.mean(neutral_penalty + missed_penalty)
        
        return total_loss


class Ablation36CModel(Ablation34Model):
    """
    Modelo trimodal con Conflict-Aware Cross-Attention + Dirichlet con regularización para neutral:
    - Audio (Wav2Vec2)
    - Texto (DistilBERT) con SpeechCue
    - Visual: Mel-espectrograma -> ViT (congelado)
    
    Fusión: Conflict-Aware Cross-Attention (detecta contradicciones y usa confianza)
    Clasificación: Dirichlet Classification Head (outputs alphas for uncertainty estimation)
    Loss: Dirichlet Evidence Loss con regularización adicional para neutral
    """

    def __init__(self, config: Dict):
        # Initialize as Ablation 34 (Conflict-Aware Fusion)
        super().__init__(config)

        model_config = config.get("model", {})
        fusion_output_dim = self.hidden_dim // 2
        
        # Replace LSTM classification head with Dirichlet Head
        self.fusion.classifier = DirichletClassificationHead(
            hidden_dim=fusion_output_dim,
            num_classes=self.num_classes,
            dropout=model_config.get("dropout", 0.5),  # Increased dropout
            num_layers=model_config.get("lstm_num_layers", 2),
            bidirectional=model_config.get("lstm_bidirectional", True),
        )
    
    def forward(
        self,
        audio: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Dirichlet output handling.
        The classifier outputs alphas instead of logits.
        """
        # Get features from parent model
        audio_outputs = self.audio_encoder(audio)
        audio_features = audio_outputs.last_hidden_state
        audio_pooled = self.pool_audio_features(audio_features)
        
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]
        
        vit_images = self._waveform_to_vit_image(audio)
        vit_images = vit_images.to(audio.device)
        vision_outputs = self.vision_encoder(pixel_values=vit_images)
        if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
            vision_features = vision_outputs.pooler_output
        else:
            vision_features = vision_outputs.last_hidden_state[:, 0, :]
        
        # Fusion returns alphas (Dirichlet parameters)
        alphas = self.fusion(audio_pooled, text_features, vision_features)
        
        # Convert alphas to probabilities and logits for compatibility
        S = torch.sum(alphas, dim=1, keepdim=True)
        probs = alphas / S
        logits = torch.log(probs + 1e-9)  # Convert to logits for predictions
        
        predictions = torch.argmax(probs, dim=-1)
        
        return {
            "logits": logits,  # For compatibility with metrics
            "alphas": alphas,  # For Dirichlet loss
            "predictions": predictions,
            "loss": None,  # Loss computed in training loop
            "audio_features": audio_pooled,
            "text_features": text_features,
            "vision_features": vision_features,
        }
