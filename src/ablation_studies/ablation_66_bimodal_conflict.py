"""
Ablation Study 66: Bimodal Conflict-Aware Cross-Attention (WavLM + DistilBERT, no ViT).

Audit: does the vision branch (ViT on log-mel spectrogram) add value when the
audio encoder is already WavLM? This model strips out the ViT entirely and uses
a 2-modality version of Conflict-Aware fusion.

Model: WavLM + DistilBERT -> BimodalConflictAwareFusion -> MultiScaleTemporal head.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn

from src.models.conformer import ConformerMultimodalModel
from src.models.fusion import BimodalConflictAwareFusion
from .classification_heads import MultiScaleTemporalClassificationHead


class Ablation66Model(ConformerMultimodalModel):
    """
    Bimodal Conflict-Aware model (WavLM + DistilBERT, no ViT).

    Extends ConformerMultimodalModel but replaces the fusion with
    BimodalConflictAwareFusion and adds the MultiScaleTemporal head.
    """

    def __init__(self, config: Dict):
        model_config = config.get("model", {})
        fusion_type = model_config.get("fusion_type", "conflict_aware_cross_attention")

        # Initialize parent (loads audio + text encoders)
        super().__init__(
            audio_encoder_name=model_config.get("audio_encoder", "facebook/wav2vec2-base"),
            text_encoder_name=model_config.get("text_encoder", "distilbert-base-uncased"),
            num_classes=model_config.get("num_classes", 4),
            hidden_dim=model_config.get("hidden_dim", 512),
            dropout=model_config.get("dropout", 0.3),
            fusion_type=fusion_type,
            fusion_num_heads=model_config.get("fusion_num_heads", 8),
            freeze_audio_encoder=model_config.get("freeze_audio_encoder", False),
            audio_unfreeze_last_n_layers=model_config.get("audio_unfreeze_last_n_layers", 0),
            freeze_text_encoder=model_config.get("freeze_text_encoder", False),
            text_unfreeze_last_n_layers=model_config.get("text_unfreeze_last_n_layers", 0),
            audio_pooling=model_config.get("audio_pooling", "mean"),
        )

        # Replace fusion with bimodal Conflict-Aware
        conflict_config = model_config.get("conflict_aware", {})
        audio_hidden_size = self.audio_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size

        self.fusion = BimodalConflictAwareFusion(
            audio_dim=audio_hidden_size,
            text_dim=text_hidden_size,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_heads=model_config.get("fusion_num_heads", 4),
            num_layers=model_config.get("fusion_num_layers", 3),
            dropout=model_config.get("dropout", 0.3),
            conflict_threshold=conflict_config.get("conflict_threshold", 0.3),
            conflict_detection_method=conflict_config.get("conflict_detection_method", "prediction_divergence"),
        )

        # Replace classifier head with MultiScaleTemporal
        head_config = model_config.get("classification_head", {})
        dropout = model_config.get("dropout", 0.3)
        self.fusion.classifier = MultiScaleTemporalClassificationHead(
            hidden_dim=self.hidden_dim // 2,
            num_classes=self.num_classes,
            dropout=dropout,
            use_temporal_pooling=head_config.get("use_temporal_pooling", True),
            pooling_types=head_config.get("pooling_types", ["mean", "max", "attention"]),
            use_lstm=head_config.get("use_lstm", True),
            lstm_num_layers=head_config.get("lstm_num_layers", 2),
            lstm_bidirectional=head_config.get("lstm_bidirectional", True),
            lstm_hidden=head_config.get("lstm_hidden", 512),
            lstm_dropout=head_config.get("lstm_dropout", 0.3),
            use_self_attention=head_config.get("use_self_attention", True),
            attention_heads=head_config.get("attention_heads", 4),
            final_dropout=head_config.get("final_dropout", 0.4),
        )

        # Explicitly mark no vision to avoid ForwardRef warnings
        self.vision_encoder = None
        self.mel_transform = None

    def forward(
        self,
        audio: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Audio features
        audio_outputs = self.audio_encoder(audio)
        audio_features = audio_outputs.last_hidden_state
        audio_pooled = self.pool_audio_features(audio_features)

        # Text features
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]

        # Bimodal Conflict-Aware fusion
        logits = self.fusion(audio_pooled, text_features)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        predictions = torch.argmax(logits, dim=-1)
        return {
            "logits": logits,
            "predictions": predictions,
            "loss": loss,
            "audio_features": audio_pooled,
            "text_features": text_features,
        }
