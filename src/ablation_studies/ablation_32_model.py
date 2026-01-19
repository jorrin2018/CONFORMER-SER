"""
Ablation Study 32: Temporal-Aligned Multimodal Bucketing

Divide el audio en "buckets" (ventanas) de ~400ms. Fusiona modalidades dentro
de cada bucket y luego usa una LSTM para conectar la secuencia de buckets.
Mejora la precisión local evitando que el ruido de un segmento largo contamine
la decisión emocional de una frase corta.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ablation_25_enhanced_cross_attention import Ablation25Model
from src.models.fusion import TriModalCrossAttentionFusion
from src.models.temporal_bucketing import divide_into_buckets, pool_bucket_features


class TemporalBucketingFusion(nn.Module):
    """
    Fusion module that processes modalities in temporal buckets.
    """
    
    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        vision_dim: int,
        hidden_dim: int,
        num_classes: int,
        bucket_fusion_type: str = "cross_attention",
        bucket_lstm_layers: int = 2,
        bucket_lstm_bidirectional: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Fusion within each bucket
        if bucket_fusion_type == "cross_attention":
            self.bucket_fusion = TriModalCrossAttentionFusion(
                audio_dim=audio_dim,
                text_dim=text_dim,
                vision_dim=vision_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown bucket_fusion_type: {bucket_fusion_type}")
        
        # LSTM to connect bucket sequence
        lstm_input_dim = hidden_dim // 2  # Output dim of bucket fusion
        self.bucket_lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=bucket_lstm_layers,
            bidirectional=bucket_lstm_bidirectional,
            dropout=dropout if bucket_lstm_layers > 1 else 0,
            batch_first=True,
        )
        
        lstm_output_dim = (hidden_dim // 2) * (2 if bucket_lstm_bidirectional else 1)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio_features: (batch_size, num_buckets, audio_dim) or (batch_size, audio_dim)
            text_features: (batch_size, num_buckets, text_dim) or (batch_size, text_dim)
            vision_features: (batch_size, num_buckets, vision_dim) or (batch_size, vision_dim)
        
        Returns:
            Logits (batch_size, num_classes)
        """
        # Handle both single bucket and multiple buckets
        if audio_features.dim() == 2:
            # Single bucket case - add sequence dimension
            audio_features = audio_features.unsqueeze(1)
            text_features = text_features.unsqueeze(1)
            vision_features = vision_features.unsqueeze(1)
        
        batch_size, num_buckets = audio_features.shape[:2]
        
        # Process each bucket
        bucket_outputs = []
        for i in range(num_buckets):
            bucket_audio = audio_features[:, i, :]  # (B, audio_dim)
            bucket_text = text_features[:, i, :]    # (B, text_dim)
            bucket_vision = vision_features[:, i, :]  # (B, vision_dim)
            
            # Fuse within bucket (temporarily replace classifier to get features)
            bucket_fused = self.bucket_fusion.fusion(
                torch.cat([
                    self.bucket_fusion.audio_proj(bucket_audio),
                    self.bucket_fusion.text_proj(bucket_text),
                    self.bucket_fusion.vision_proj(bucket_vision),
                ], dim=-1)
            )  # (B, hidden_dim // 2)
            
            bucket_outputs.append(bucket_fused)
        
        # Stack bucket outputs
        bucket_sequence = torch.stack(bucket_outputs, dim=1)  # (B, num_buckets, hidden_dim // 2)
        
        # LSTM over bucket sequence
        lstm_out, _ = self.bucket_lstm(bucket_sequence)  # (B, num_buckets, lstm_output_dim)
        
        # Use last timestep or mean pooling
        lstm_final = lstm_out[:, -1, :]  # (B, lstm_output_dim)
        
        # Classification
        logits = self.classifier(lstm_final)
        return logits


class Ablation32Model(Ablation25Model):
    """
    Modelo trimodal con Temporal Bucketing:
    - Audio (Wav2Vec2)
    - Texto (DistilBERT) con SpeechCue
    - Visual: Mel-espectrograma -> ViT (congelado)
    
    Modificación: Procesa modalidades en buckets temporales y conecta con LSTM.
    Divide el audio en ventanas temporales (~400ms), procesa cada bucket por separado,
    y luego conecta la secuencia de buckets con una LSTM.
    Basado en Ablation 25 (Enhanced Cross-Attention).
    """

    def __init__(self, config: Dict):
        # Initialize as Ablation 25 (Enhanced Cross-Attention base)
        super().__init__(config)

        model_config = config.get("model", {})
        bucketing_config = model_config.get("temporal_bucketing", {})
        dataset_config = config.get("dataset", {})
        
        audio_hidden_size = self.audio_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size
        vision_hidden_size = self.vision_encoder.config.hidden_size

        # Store bucketing parameters
        self.bucket_duration_ms = bucketing_config.get("bucket_duration_ms", 400)
        self.bucket_overlap_ms = bucketing_config.get("bucket_overlap_ms", 50)
        self.sample_rate = dataset_config.get("sample_rate", 16000)

        # Replace fusion with temporal bucketing fusion
        self.fusion = TemporalBucketingFusion(
            audio_dim=audio_hidden_size,
            text_dim=text_hidden_size,
            vision_dim=vision_hidden_size,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            bucket_fusion_type=bucketing_config.get("bucket_fusion_type", "cross_attention"),
            bucket_lstm_layers=bucketing_config.get("bucket_lstm_layers", 2),
            bucket_lstm_bidirectional=bucketing_config.get("bucket_lstm_bidirectional", True),
            dropout=model_config.get("dropout", 0.3),
        )

    def forward(
        self,
        audio: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with temporal bucketing.
        
        Args:
            audio: Audio waveform (batch_size, n_samples)
            text_input_ids: Text input IDs (batch_size, seq_len)
            text_attention_mask: Text attention mask (batch_size, seq_len)
            labels: Ground truth labels (batch_size,)
        
        Returns:
            Dictionary containing logits, loss, predictions
        """
        batch_size = audio.shape[0]
        
        # Divide audio into buckets (at waveform level)
        audio_buckets, num_buckets = divide_into_buckets(
            audio,
            bucket_duration_ms=self.bucket_duration_ms,
            sample_rate=self.sample_rate,
            overlap_ms=self.bucket_overlap_ms,
        )  # (B, num_buckets, bucket_samples)
        
        # Reshape for batch processing: (B * num_buckets, bucket_samples)
        audio_buckets_flat = audio_buckets.reshape(-1, audio_buckets.shape[-1])
        
        # Process all audio buckets through encoder
        audio_outputs = self.audio_encoder(audio_buckets_flat)
        audio_features_seq = audio_outputs.last_hidden_state  # (B * num_buckets, seq_len, audio_dim)
        
        # Pool features within each bucket (use mean pooling)
        audio_features_pooled = torch.mean(audio_features_seq, dim=1)  # (B * num_buckets, audio_dim)
        
        # Reshape back: (B, num_buckets, audio_dim)
        audio_features_buckets = audio_features_pooled.reshape(batch_size, num_buckets, -1)
        
        # Process text - for simplicity, replicate text features for each bucket
        # (In a more sophisticated implementation, we could align text with audio buckets)
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]  # CLS (B, text_dim)
        
        # Replicate text features for each bucket
        text_features_buckets = text_features.unsqueeze(1).expand(-1, num_buckets, -1)  # (B, num_buckets, text_dim)
        
        # Process vision - create images for each audio bucket
        # Reshape audio buckets for vision processing
        vision_features_list = []
        for i in range(num_buckets):
            audio_bucket = audio_buckets[:, i, :]  # (B, bucket_samples)
            vit_images = self._waveform_to_vit_image(audio_bucket)  # (B, 3, H, W)
            vit_images = vit_images.to(audio.device)
            
            vision_outputs = self.vision_encoder(pixel_values=vit_images)
            if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
                vision_features = vision_outputs.pooler_output  # (B, H_v)
            else:
                vision_features = vision_outputs.last_hidden_state[:, 0, :]  # CLS
            
            vision_features_list.append(vision_features)
        
        # Stack vision features: (B, num_buckets, vision_dim)
        vision_features_buckets = torch.stack(vision_features_list, dim=1)
        
        # Fusion with bucketing
        logits = self.fusion(
            audio_features_buckets,
            text_features_buckets,
            vision_features_buckets,
        )
        
        # Compute loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        predictions = torch.argmax(logits, dim=-1)
        
        return {
            "logits": logits,
            "predictions": predictions,
            "loss": loss,
            "audio_features": audio_features_buckets,
            "text_features": text_features_buckets,
            "vision_features": vision_features_buckets,
        }

