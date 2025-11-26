"""
Ablation Study 06: Unfreeze encoders + Transformer classification head.
"""

import torch.nn as nn
from src.models.conformer import ConformerMultimodalModel
from .classification_heads import TransformerClassificationHead


class Ablation06Model(ConformerMultimodalModel):
    """
    Model with unfrozen encoders and Transformer-based classification head.
    
    Combina el fine-tuning de encoders (Ablation 01) con una cabeza
    de clasificación Transformer (Ablation 03) para evaluar el impacto conjunto.
    """
    
    def __init__(self, config: dict):
        """
        Initialize model with unfrozen encoder layers and Transformer head.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        # Extract model config
        model_config = config.get('model', {})
        
        # Call parent constructor with unfrozen encoders (like Ablation 01)
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
        
        # Replace classifier with Transformer head (like Ablation 03)
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







