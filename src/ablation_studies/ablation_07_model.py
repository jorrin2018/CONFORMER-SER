"""
Ablation Study 07: Unfreeze middle and high layers of encoders + LSTM classification head.
"""

import torch.nn as nn
from src.models.conformer import ConformerMultimodalModel
from .classification_heads import LSTMClassificationHead


class Ablation07Model(ConformerMultimodalModel):
    """
    Model with unfrozen middle and high encoder layers and LSTM-based classification head.
    
    Similar to Ablation 05, but unfreezes more layers:
    - Audio: last 6 layers (instead of 4) - middle and high layers
    - Text: last 4 layers (instead of 2) - middle and high layers
    - LSTM classification head (same as Ablation 05)
    """
    
    def __init__(self, config: dict):
        """
        Initialize model with unfrozen middle/high encoder layers and LSTM head.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        # Extract model config
        model_config = config.get('model', {})
        
        # Call parent constructor with more unfrozen layers
        super().__init__(
            audio_encoder_name=model_config.get('audio_encoder', 'facebook/wav2vec2-base'),
            text_encoder_name=model_config.get('text_encoder', 'distilbert-base-uncased'),
            num_classes=model_config.get('num_classes', 7),
            hidden_dim=model_config.get('hidden_dim', 256),
            dropout=model_config.get('dropout', 0.3),
            fusion_type=model_config.get('fusion_type', 'cross_attention'),
            freeze_audio_encoder=model_config.get('freeze_audio_encoder', False),
            audio_unfreeze_last_n_layers=model_config.get('audio_unfreeze_last_n_layers', 6),  # More layers
            freeze_text_encoder=model_config.get('freeze_text_encoder', False),
            text_unfreeze_last_n_layers=model_config.get('text_unfreeze_last_n_layers', 4)  # More layers
        )
        
        # Replace classifier with LSTM head (same as Ablation 05)
        fusion_output_dim = self.hidden_dim // 2
        lstm_num_layers = model_config.get('lstm_num_layers', 2)
        lstm_bidirectional = model_config.get('lstm_bidirectional', True)
        
        self.fusion.classifier = LSTMClassificationHead(
            hidden_dim=fusion_output_dim,
            num_classes=self.num_classes,
            dropout=model_config.get('dropout', 0.3),
            num_layers=lstm_num_layers,
            bidirectional=lstm_bidirectional
        )






