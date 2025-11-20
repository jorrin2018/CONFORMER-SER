"""
Ablation Study 09: Unfreeze encoders + GIA fusion + LSTM classification head.

Extends Ablation 05 by replacing the cross-attention fusion with the
GIA (Gated Interaction Attention) fusion block.
"""

from src.models.conformer import ConformerMultimodalModel
from .classification_heads import LSTMClassificationHead


class Ablation09Model(ConformerMultimodalModel):
    """
    Model with partially unfrozen encoders, GIA fusion, and LSTM-based
    classification head.
    """

    def __init__(self, config: dict):
        """
        Initialize model with unfrozen encoder layers, GIA fusion, and LSTM head.

        Args:
            config: Configuration dictionary with model parameters.
        """
        model_config = config.get("model", {})

        # Default to GIA fusion if not explicitly specified
        fusion_type = model_config.get("fusion_type", "gia")

        super().__init__(
            audio_encoder_name=model_config.get(
                "audio_encoder", "facebook/wav2vec2-base"
            ),
            text_encoder_name=model_config.get(
                "text_encoder", "distilbert-base-uncased"
            ),
            num_classes=model_config.get("num_classes", 7),
            hidden_dim=model_config.get("hidden_dim", 256),
            dropout=model_config.get("dropout", 0.3),
            fusion_type=fusion_type,
            fusion_num_heads=model_config.get("fusion_num_heads", 4),
            freeze_audio_encoder=model_config.get("freeze_audio_encoder", False),
            audio_unfreeze_last_n_layers=model_config.get(
                "audio_unfreeze_last_n_layers", 4
            ),
            freeze_text_encoder=model_config.get("freeze_text_encoder", False),
            text_unfreeze_last_n_layers=model_config.get(
                "text_unfreeze_last_n_layers", 2
            ),
        )

        # The fusion module outputs features of size hidden_dim // 2.
        fusion_output_dim = self.hidden_dim // 2
        lstm_num_layers = model_config.get("lstm_num_layers", 2)
        lstm_bidirectional = model_config.get("lstm_bidirectional", True)

        # Replace classifier with an LSTM-based head (same as Ablation 05).
        self.fusion.classifier = LSTMClassificationHead(
            hidden_dim=fusion_output_dim,
            num_classes=self.num_classes,
            dropout=model_config.get("dropout", 0.3),
            num_layers=lstm_num_layers,
            bidirectional=lstm_bidirectional,
        )


