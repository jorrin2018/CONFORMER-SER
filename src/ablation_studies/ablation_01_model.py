"""
Ablation Study 01: Unfreeze encoders (audio and text).
"""

import torch.nn as nn
from src.models.conformer import ConformerMultimodalModel


class Ablation01Model(ConformerMultimodalModel):
    """
    Model with unfrozen encoders for fine-tuning.
    
    Descongela las últimas N capas de los encoders de audio y texto
    para permitir fine-tuning mientras mantiene el resto congelado.
    """
    
    def __init__(self, config: dict):
        """
        Initialize model with unfrozen encoder layers.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        # Extract model config
        model_config = config.get('model', {})
        
        # Call parent constructor with unfrozen encoders
        super().__init__(
            audio_encoder_name=model_config.get('audio_encoder', 'facebook/wav2vec2-base'),
            text_encoder_name=model_config.get('text_encoder', 'distilbert-base-uncased'),
            num_classes=model_config.get('num_classes', 7),
            hidden_dim=model_config.get('hidden_dim', 256),
            dropout=model_config.get('dropout', 0.3),
            fusion_type=model_config.get('fusion_type', 'cross_attention'),
            freeze_audio_encoder=model_config.get('freeze_audio_encoder', False),
            audio_unfreeze_last_n_layers=model_config.get('audio_unfreeze_last_n_layers', 4),
            freeze_text_encoder=model_config.get('freeze_text_encoder', False),
            text_unfreeze_last_n_layers=model_config.get('text_unfreeze_last_n_layers', 2)
        )

