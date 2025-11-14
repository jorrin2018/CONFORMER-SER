"""
Ablation Study 03: Transformer classification head.
"""

import torch.nn as nn
from src.models.conformer import ConformerMultimodalModel
from .classification_heads import TransformerClassificationHead


class Ablation03Model(ConformerMultimodalModel):
    """
    Model with Transformer-based classification head.
    
    Reemplaza la cabeza de clasificación estándar con un pequeño
    transformer encoder para procesar las características fusionadas.
    """
    
    def __init__(self, config: dict):
        """
        Initialize model with Transformer classification head.
        
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
        
        # Replace classifier with Transformer head
        # The fusion module outputs features of size hidden_dim // 2
        fusion_output_dim = self.hidden_dim // 2
        transformer_num_layers = model_config.get('transformer_num_layers', 1)
        transformer_num_heads = model_config.get('transformer_num_heads', 8)
        
        self.fusion.classifier = TransformerClassificationHead(
            hidden_dim=fusion_output_dim,
            num_classes=self.num_classes,
            dropout=model_config.get('dropout', 0.3),
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads
        )

