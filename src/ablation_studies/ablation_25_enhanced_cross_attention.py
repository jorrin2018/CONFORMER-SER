"""
Ablation Study 25: Enhanced Cross-Attention Fusion

Same architecture as Ablation 13c but uses TriModalEnhancedCrossAttentionFusion
with multiple layers of cross-attention for deeper interaction modeling.
"""

from typing import Dict

from .ablation_13_model import Ablation13Model
from src.models.fusion import TriModalEnhancedCrossAttentionFusion


class Ablation25Model(Ablation13Model):
    """
    Modelo trimodal con fusión cross-attention mejorada:
    - Audio (Wav2Vec2)
    - Texto (DistilBERT) con SpeechCue
    - Visual: Mel-espectrograma -> ViT (congelado)
    
    Fusión: Enhanced Cross-Attention (múltiples capas)
    """

    def __init__(self, config: Dict):
        # Initialize as Ablation 13 (backbone + ViT + SpeechCue)
        super().__init__(config)

        model_config = config.get("model", {})
        audio_hidden_size = self.audio_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size
        vision_hidden_size = self.vision_encoder.config.hidden_size

        # Replace fusion with enhanced cross-attention fusion
        self.fusion = TriModalEnhancedCrossAttentionFusion(
            audio_dim=audio_hidden_size,
            text_dim=text_hidden_size,
            vision_dim=vision_hidden_size,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_heads=model_config.get("fusion_num_heads", 4),
            num_layers=model_config.get("fusion_num_layers", 3),
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

