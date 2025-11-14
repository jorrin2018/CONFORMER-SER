"""
Ablation Study 02: LSTM classification head.
"""

import torch.nn as nn
from src.models.conformer import ConformerMultimodalModel
from .classification_heads import LSTMClassificationHead


class Ablation02Model(ConformerMultimodalModel):
    """
    Model with LSTM-based classification head.
    
    Reemplaza la cabeza de clasificación estándar con una red LSTM
    bidireccional para procesar las características fusionadas.
    """
    
    def __init__(self, config: dict):
        """
        Initialize model with LSTM classification head.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        # Extract model config
        model_config = config.get('model', {})
        
        # Call parent constructor (encoders remain frozen as base)
        super().__init__(
            audio_encoder_name=model_config.get('audio_encoder', 'facebook/wav2vec2-base'),
            text_encoder_name=model_config.get('text_encoder', 'distilbert-base-uncased'),
            num_classes=model_config.get('num_classes', 7),
            hidden_dim=model_config.get('hidden_dim', 256),
            dropout=model_config.get('dropout', 0.3),
            fusion_type=model_config.get('fusion_type', 'cross_attention'),
            freeze_audio_encoder=model_config.get('freeze_audio_encoder', True),
            audio_unfreeze_last_n_layers=model_config.get('audio_unfreeze_last_n_layers', 0),
            freeze_text_encoder=model_config.get('freeze_text_encoder', True),
            text_unfreeze_last_n_layers=model_config.get('text_unfreeze_last_n_layers', 0)
        )
        
        # Replace classifier with LSTM head
        # The fusion module outputs features of size hidden_dim // 2
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

