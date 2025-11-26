"""
Ablation Study 23: Bilinear Fusion

Same architecture as Ablation 13c but uses TriModalBilinearFusion
for bilinear pooling to capture second-order interactions.
"""

from typing import Dict

from .ablation_13_model import Ablation13Model
from src.models.fusion import TriModalBilinearFusion


class Ablation23Model(Ablation13Model):
    """
    Modelo trimodal con fusión bilineal:
    - Audio (Wav2Vec2)
    - Texto (DistilBERT) con SpeechCue
    - Visual: Mel-espectrograma -> ViT (congelado)
    
    Fusión: Bilinear Pooling (factorized)
    """

    def __init__(self, config: Dict):
        # Initialize as Ablation 13 (backbone + ViT + SpeechCue)
        super().__init__(config)

        model_config = config.get("model", {})
        audio_hidden_size = self.audio_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size
        vision_hidden_size = self.vision_encoder.config.hidden_size

        # Replace fusion with bilinear fusion
        self.fusion = TriModalBilinearFusion(
            audio_dim=audio_hidden_size,
            text_dim=text_hidden_size,
            vision_dim=vision_hidden_size,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            rank=model_config.get("bilinear_rank", 16),
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

