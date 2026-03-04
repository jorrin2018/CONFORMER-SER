"""
Ablation Study 39: ACN + MIC + BCAF over Conflict-Aware Cross-Attention.

Adds:
- Audio Context Network (ACN) before pooling
- Modal Invariance Constraint (MIC) using SKL
- Bidirectional Connection Attention Fusion (BCAF)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ablation_34_model import Ablation34Model
from .classification_heads import LSTMClassificationHead
from src.models.audio_context_network import AudioContextNetwork
from src.models.modal_invariance import ModalInvarianceConstraint
from src.models.bcaf_fusion import BidirectionalConnectionAttentionFusion
from src.models.fusion import ConflictAwareCrossAttentionFusion


class Ablation39Model(Ablation34Model):
    """
    Ablation 39: Conflict-Aware Cross-Attention + ACN + MIC + BCAF.
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        model_config = config.get("model", {})
        acn_config = model_config.get("audio_context_network", {})
        mic_config = model_config.get("modal_invariance", {})
        bcaf_config = model_config.get("bcaf", {})
        conflict_config = model_config.get("conflict_aware", {})

        self.use_acn = acn_config.get("enabled", True)
        self.use_mic = mic_config.get("enabled", True)
        self.use_bcaf = bcaf_config.get("enabled", True)

        audio_hidden_size = self.audio_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size
        vision_hidden_size = self.vision_encoder.config.hidden_size

        acn_out_dim = int(acn_config.get("conv1d_out_channels", self.hidden_dim))
        acn_kernel = int(acn_config.get("conv1d_kernel_size", 3))
        acn_layers = int(acn_config.get("transformer_num_layers", 2))
        acn_heads = int(acn_config.get("transformer_num_heads", 8))
        acn_dropout = float(acn_config.get("dropout", 0.1))

        if self.use_acn:
            self.acn = AudioContextNetwork(
                input_dim=audio_hidden_size,
                output_dim=acn_out_dim,
                conv_kernel_size=acn_kernel,
                num_transformer_layers=acn_layers,
                num_heads=acn_heads,
                dropout=acn_dropout,
            )
        else:
            self.acn = None
            acn_out_dim = audio_hidden_size

        # Replace fusion to match ACN output dimension for audio
        self.fusion = ConflictAwareCrossAttentionFusion(
            audio_dim=acn_out_dim,
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

        mic_latent_dim = int(mic_config.get("latent_dim", self.hidden_dim))
        self.mic_temperature = float(mic_config.get("temperature", 1.0))
        self.mic_loss_weight = float(mic_config.get("loss_weight", 0.1))
        if self.use_mic:
            self.mic = ModalInvarianceConstraint(temperature=self.mic_temperature)
            self.mic_audio_proj = nn.Linear(acn_out_dim, mic_latent_dim)
            self.mic_text_proj = nn.Linear(text_hidden_size, mic_latent_dim)
        else:
            self.mic = None
            self.mic_audio_proj = None
            self.mic_text_proj = None

        self.bcaf_loss_weight = float(bcaf_config.get("loss_weight", 0.1))
        if self.use_bcaf:
            self.bcaf = BidirectionalConnectionAttentionFusion(
                audio_dim=acn_out_dim,
                text_dim=text_hidden_size,
                vision_dim=vision_hidden_size,
                latent_dim=int(bcaf_config.get("latent_dim", 256)),
                hidden_dim=int(bcaf_config.get("hidden_dim", self.hidden_dim)),
                dropout=float(bcaf_config.get("dropout", 0.1)),
                reconstruction_weight=float(bcaf_config.get("reconstruction_weight", 1.0)),
                alignment_weight=float(bcaf_config.get("alignment_weight", 0.5)),
            )
        else:
            self.bcaf = None

    def forward(
        self,
        audio: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with ACN, MIC, and BCAF.
        """
        # Audio: Wav2Vec2 -> ACN -> Pool
        audio_outputs = self.audio_encoder(audio)
        audio_features = audio_outputs.last_hidden_state  # (B, S, H_a)
        if self.acn is not None:
            audio_features = self.acn(audio_features)
        audio_pooled = self.pool_audio_features(audio_features)

        # Text
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]

        # Vision
        vit_images = self._waveform_to_vit_image(audio)
        vit_images = vit_images.to(audio.device)
        vision_outputs = self.vision_encoder(pixel_values=vit_images)
        if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
            vision_features = vision_outputs.pooler_output
        else:
            vision_features = vision_outputs.last_hidden_state[:, 0, :]

        bcaf_loss = torch.tensor(0.0, device=audio.device)
        bcaf_audio = audio_pooled
        bcaf_text = text_features
        bcaf_vision = vision_features
        if self.bcaf is not None:
            bcaf_outputs, bcaf_loss = self.bcaf(
                audio_pooled, text_features, vision_features
            )
            bcaf_audio = bcaf_outputs["audio"]
            bcaf_text = bcaf_outputs["text"]
            bcaf_vision = bcaf_outputs["vision"]

        # Fusion
        logits = self.fusion(bcaf_audio, bcaf_text, bcaf_vision)

        mic_loss = torch.tensor(0.0, device=audio.device)
        if self.mic is not None:
            audio_mic = self.mic_audio_proj(audio_pooled)
            text_mic = self.mic_text_proj(text_features)
            mic_loss = self.mic(audio_mic, text_mic)

        loss = None
        if labels is not None:
            classification_loss = F.cross_entropy(logits, labels)
            loss = (
                classification_loss +
                self.mic_loss_weight * mic_loss +
                self.bcaf_loss_weight * bcaf_loss
            )

        predictions = torch.argmax(logits, dim=-1)

        return {
            "logits": logits,
            "predictions": predictions,
            "loss": loss,
            "audio_features": audio_pooled,
            "text_features": text_features,
            "vision_features": vision_features,
            "mic_loss": mic_loss,
            "bcaf_loss": bcaf_loss,
        }


