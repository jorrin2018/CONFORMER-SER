"""
Ablation Study 36A: Conflict-Aware + Dirichlet (Pure)
Base: Ablation 34 (Conflict-Aware Cross-Attention)
Modificación: Dirichlet Classification Head + Dirichlet Evidence Loss
Objetivo: Combinar mejor generalización (Conflict-Aware) con mejor validación (Dirichlet)
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ablation_34_model import Ablation34Model
from .ablation_30_dirichlet import DirichletClassificationHead


class Ablation36AModel(Ablation34Model):
    """
    Modelo trimodal con Conflict-Aware Cross-Attention + Dirichlet Evidence Learning:
    - Audio (Wav2Vec2)
    - Texto (DistilBERT) con SpeechCue
    - Visual: Mel-espectrograma -> ViT (congelado)
    
    Fusión: Conflict-Aware Cross-Attention (detecta contradicciones y usa confianza)
    Clasificación: Dirichlet Classification Head (outputs alphas for uncertainty estimation)
    Loss: Dirichlet Evidence Loss
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
