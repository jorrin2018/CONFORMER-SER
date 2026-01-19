"""
Ablation Study 31: Temporal Alignment Learning (Adaptive DTW)

Módulo insertado antes de la fusión que aprende a "estirar" o "comprimir"
temporalmente los features de audio/texto para que coincidan evento-a-evento
antes de fusionarlos. Resuelve la asincronía real entre modalidades.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ablation_25_enhanced_cross_attention import Ablation25Model
from src.models.fusion import TriModalEnhancedCrossAttentionFusion
from src.models.temporal_alignment import TemporalAlignmentModule


class Ablation31Model(Ablation25Model):
    """
    Modelo trimodal con Temporal Alignment Learning:
    - Audio (Wav2Vec2)
    - Texto (DistilBERT) con SpeechCue
    - Visual: Mel-espectrograma -> ViT (congelado)
    
    Modificación: Módulo de alineación temporal antes de la fusión.
    Aprende a alinear secuencias de audio y texto temporalmente usando DTW adaptativo.
    """

    def __init__(self, config: Dict):
        # Initialize as Ablation 25 (Enhanced Cross-Attention base)
        super().__init__(config)

        model_config = config.get("model", {})
        alignment_config = model_config.get("temporal_alignment", {})
        
        audio_hidden_size = self.audio_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size
        vision_hidden_size = self.vision_encoder.config.hidden_size

        # Temporal alignment module
        self.temporal_alignment = TemporalAlignmentModule(
            audio_dim=audio_hidden_size,
            text_dim=text_hidden_size,
            alignment_dim=alignment_config.get("alignment_dim", 128),
            num_layers=alignment_config.get("alignment_layers", 2),
            max_warp_factor=alignment_config.get("max_warp_factor", 1.5),
            dropout=model_config.get("dropout", 0.3),
        )

        # Store alignment loss weight from config
        self.alignment_loss_weight = config.get("training", {}).get("loss", {}).get(
            "alignment_loss_weight", 0.1
        )

        # Use enhanced cross-attention as base (temporal alignment applied before this)
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

    def forward(
        self,
        audio: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with temporal alignment.
        
        Args:
            audio: Audio waveform (batch_size, n_samples)
            text_input_ids: Text input IDs (batch_size, seq_len)
            text_attention_mask: Text attention mask (batch_size, seq_len)
            labels: Ground truth labels (batch_size,)
        
        Returns:
            Dictionary containing logits, loss, predictions, and alignment_loss
        """
        # Extract audio features (temporal sequence)
        audio_outputs = self.audio_encoder(audio)
        audio_features_seq = audio_outputs.last_hidden_state  # (B, audio_seq_len, audio_dim)

        # Extract text features (temporal sequence)
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
        )
        text_features_seq = text_outputs.last_hidden_state  # (B, text_seq_len, text_dim)

        # Apply temporal alignment
        aligned_audio_seq, aligned_text_seq, alignment_loss = self.temporal_alignment(
            audio_features_seq, text_features_seq
        )  # (B, audio_seq_len, dim)

        # Pool aligned sequences to fixed-size vectors
        # Use mean pooling (can be changed to attention pooling)
        audio_pooled = torch.mean(aligned_audio_seq, dim=1)  # (B, audio_dim)
        text_pooled = torch.mean(aligned_text_seq, dim=1)    # (B, text_dim)

        # Extract vision features (same as Ablation 13)
        vit_images = self._waveform_to_vit_image(audio)  # (B, 3, H, W)
        vit_images = vit_images.to(audio.device)

        vision_outputs = self.vision_encoder(pixel_values=vit_images)
        if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
            vision_features = vision_outputs.pooler_output  # (B, H_v)
        else:
            vision_features = vision_outputs.last_hidden_state[:, 0, :]  # CLS

        # Fusion trimodal
        logits = self.fusion(audio_pooled, text_pooled, vision_features)

        # Compute classification loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(logits, labels)
            
            # Combine classification loss with alignment regularization loss
            loss = classification_loss + self.alignment_loss_weight * alignment_loss

        predictions = torch.argmax(logits, dim=-1)

        return {
            "logits": logits,
            "predictions": predictions,
            "loss": loss,
            "alignment_loss": alignment_loss,
            "audio_features": audio_pooled,
            "text_features": text_pooled,
            "vision_features": vision_features,
        }

