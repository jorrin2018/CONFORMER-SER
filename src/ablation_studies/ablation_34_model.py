"""
Ablation Study 34: Conflict-Aware Cross-Attention Fusion

Mecanismo de atención que detecta explícitamente cuando dos modalidades
se contradicen (ej. Texto feliz + Audio enojado) y asigna pesos basados
en "confianza" en lugar de simplemente promediar.
"""

from typing import Dict

from .ablation_25_enhanced_cross_attention import Ablation25Model
from src.models.fusion import ConflictAwareCrossAttentionFusion


class Ablation34Model(Ablation25Model):
    """
    Modelo trimodal con Conflict-Aware Cross-Attention:
    - Audio (Wav2Vec2)
    - Texto (DistilBERT) con SpeechCue
    - Visual: Mel-espectrograma -> ViT (congelado)
    
    Fusión: Conflict-Aware Cross-Attention (detecta contradicciones y usa confianza)
    Basado en Ablation 25 (Enhanced Cross-Attention).
    """

    def __init__(self, config: Dict):
        # Initialize as Ablation 25 (Enhanced Cross-Attention base)
        super().__init__(config)

        model_config = config.get("model", {})
        conflict_config = model_config.get("conflict_aware", {})
        
        audio_hidden_size = self.audio_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size
        vision_hidden_size = self.vision_encoder.config.hidden_size

        # Replace fusion with conflict-aware cross-attention fusion
        self.fusion = ConflictAwareCrossAttentionFusion(
            audio_dim=audio_hidden_size,
            text_dim=text_hidden_size,
            vision_dim=vision_hidden_size,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_heads=model_config.get("fusion_num_heads", 4),
            num_layers=model_config.get("fusion_num_layers", 3),
            dropout=model_config.get("dropout", 0.3),
            conflict_threshold=conflict_config.get("conflict_threshold", 0.3),
            conflict_detection_method=conflict_config.get("conflict_detection_method", "prediction_divergence"),
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

