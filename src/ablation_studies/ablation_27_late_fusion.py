"""
Ablation Study 27: Late Fusion

Same architecture as Ablation 13c but uses TriModalLateFusion
where each modality processes independently and final logits are combined.
"""

from typing import Dict

from .ablation_13_model import Ablation13Model
from src.models.fusion import TriModalLateFusion


class Ablation27Model(Ablation13Model):
    """
    Modelo trimodal con fusión tardía:
    - Audio (Wav2Vec2)
    - Texto (DistilBERT) con SpeechCue
    - Visual: Mel-espectrograma -> ViT (congelado)
    
    Fusión: Late Fusion (combinación de decisiones finales)
    """

    def __init__(self, config: Dict):
        # Initialize as Ablation 13 (backbone + ViT + SpeechCue)
        super().__init__(config)

        model_config = config.get("model", {})
        audio_hidden_size = self.audio_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size
        vision_hidden_size = self.vision_encoder.config.hidden_size

        # Replace fusion with late fusion
        self.fusion = TriModalLateFusion(
            audio_dim=audio_hidden_size,
            text_dim=text_hidden_size,
            vision_dim=vision_hidden_size,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            dropout=model_config.get("dropout", 0.3),
        )

        # Note: Late fusion already includes classifiers, so we don't replace it

