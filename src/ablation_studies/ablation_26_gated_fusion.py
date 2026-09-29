"""
Ablation Study 26: Gated Fusion

Same architecture as Ablation 13c but uses TriModalGatedFusion
with gating mechanisms for adaptive information flow control.
"""

from typing import Dict

from .ablation_13_model import Ablation13Model
from src.models.fusion import TriModalGatedFusion


class Ablation26Model(Ablation13Model):
    """
    Modelo trimodal con fusión gated:
    - Audio (Wav2Vec2)
    - Texto (DistilBERT) con SpeechCue
    - Visual: Mel-espectrograma -> ViT (congelado)
    
    Fusión: Gated Fusion (mecanismos de puerta adaptativos)
    """

    def __init__(self, config: Dict):
        # Initialize as Ablation 13 (backbone + ViT + SpeechCue)
        super().__init__(config)

        model_config = config.get("model", {})
        audio_hidden_size = self.audio_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size
        vision_hidden_size = self.vision_encoder.config.hidden_size

        # Replace fusion with gated fusion
        self.fusion = TriModalGatedFusion(
            audio_dim=audio_hidden_size,
            text_dim=text_hidden_size,
            vision_dim=vision_hidden_size,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_heads=model_config.get("fusion_num_heads", 4),
            dropout=model_config.get("dropout", 0.3),
        )

        # Keep the same LSTM classification head
        from .classification_heads import LSTMClassificationHead
        fusion_output_dim = self.hidden_dim // 2
        lstm_num_layers = model_config.get("lstm_num_layers", 2)
        lstm_bidirectional = model_config.get("lstm_bidirectional", True)

        self.fusion.classifier = LSTMClassificationHead(
            hidden_dim=fusion_output_dim,
            num_classes=self.num_classes,
            dropout=model_config.get("dropout", 0.3),
            num_layers=lstm_num_layers,
            bidirectional=lstm_bidirectional,
        )

