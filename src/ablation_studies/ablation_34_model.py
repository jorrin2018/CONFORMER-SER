"""
Ablation Study 34: Conflict-Aware Cross-Attention (trimodal) or
WavLM + RoBERTa + Gated Interactive Attention (bimodal phase1).

- Trimodal: Conflict-Aware Cross-Attention (audio + text + vision).
- Bimodal (fusion_type gated_interactive_attention): WavLM + RoBERTa,
  GatedInteractiveAttentionFusion + MultiScaleTemporal head, optional conflict-aware.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from .ablation_25_enhanced_cross_attention import Ablation25Model
from src.models.fusion import ConflictAwareCrossAttentionFusion, GatedInteractiveAttentionFusion
from .classification_heads import LSTMClassificationHead, MultiScaleTemporalClassificationHead


class Ablation34Model(Ablation25Model):
    """
    - Si fusion_type != "gated_interactive_attention": modelo trimodal con
      Conflict-Aware Cross-Attention (hereda Ablation25).
    - Si fusion_type == "gated_interactive_attention": modelo bimodal
      (WavLM + RoBERTa) con GatedInteractiveAttentionFusion +
      MultiScaleTemporalClassificationHead y opcional conflict-aware.
    """

    def __init__(self, config: Dict):
        model_config = config.get("model", {})
        fusion_type = model_config.get("fusion_type", "")

        if fusion_type == "gated_interactive_attention":
            # Bimodal path: WavLM + RoBERTa, no vision
            self._init_bimodal_gated(config)
            return

        # Trimodal path: Conflict-Aware (same as before)
        super().__init__(config)
        conflict_config = model_config.get("conflict_aware", {})
        audio_hidden_size = self.audio_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size
        vision_hidden_size = self.vision_encoder.config.hidden_size

        self.fusion = ConflictAwareCrossAttentionFusion(
            audio_dim=audio_hidden_size,
            text_dim=text_hidden_size,
            vision_dim=vision_hidden_size,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_heads=model_config.get("fusion_num_heads", 4),
            num_layers=model_config.get("fusion_num_layers", 3),
            dropout=model_config.get("dropout", 0.3),
            conflict_threshold=conflict_config.get("conflict_threshold", 0.3),
            conflict_detection_method=conflict_config.get(
                "conflict_detection_method", "prediction_divergence"
            ),
        )
        fusion_output_dim = self.hidden_dim // 2
        self.fusion.classifier = LSTMClassificationHead(
            hidden_dim=fusion_output_dim,
            num_classes=self.num_classes,
            dropout=model_config.get("dropout", 0.3),
            num_layers=model_config.get("lstm_num_layers", 2),
            bidirectional=model_config.get("lstm_bidirectional", True),
        )

    def _init_bimodal_gated(self, config: Dict):
        """Build bimodal model: WavLM + RoBERTa + GatedInteractiveAttention + MultiScaleTemporal head."""
        model_config = config.get("model", {})
        dataset_config = config.get("dataset", {})

        self.num_classes = model_config.get("num_classes", 4)
        self.hidden_dim = model_config.get("hidden_dim", 768)
        dropout = model_config.get("dropout", 0.3)
        audio_encoder_name = model_config.get("audio_encoder", "microsoft/wavlm-large")
        text_encoder_name = model_config.get("text_encoder", "roberta-base")
        audio_pooling = model_config.get("audio_pooling", "mean")

        nn.Module.__init__(self)
        self.fusion_type = "gated_interactive_attention"
        self.audio_pooling = audio_pooling

        # Audio encoder (WavLM / Wav2Vec2 via AutoModel)
        try:
            self.audio_encoder = AutoModel.from_pretrained(audio_encoder_name)
            audio_hidden_size = self.audio_encoder.config.hidden_size
        except Exception as e:
            print(f"Warning: Could not load {audio_encoder_name}: {e}. Using config.")
            cfg = AutoConfig.from_pretrained(audio_encoder_name)
            self.audio_encoder = AutoModel.from_config(cfg)
            audio_hidden_size = cfg.hidden_size

        freeze_audio = model_config.get("freeze_audio_encoder", False)
        if freeze_audio:
            for p in self.audio_encoder.parameters():
                p.requires_grad = False
        unfreeze_audio_n = model_config.get("audio_unfreeze_last_n_layers", 0)
        if unfreeze_audio_n and hasattr(self.audio_encoder, "encoder") and hasattr(self.audio_encoder.encoder, "layers"):
            for layer in self.audio_encoder.encoder.layers[-unfreeze_audio_n:]:
                for p in layer.parameters():
                    p.requires_grad = True

        # Text encoder
        try:
            self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
            text_hidden_size = self.text_encoder.config.hidden_size
        except Exception as e:
            print(f"Warning: Could not load {text_encoder_name}: {e}. Using config.")
            cfg = AutoConfig.from_pretrained(text_encoder_name)
            self.text_encoder = AutoModel.from_config(cfg)
            text_hidden_size = cfg.hidden_size

        freeze_text = model_config.get("freeze_text_encoder", False)
        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        unfreeze_text_n = model_config.get("text_unfreeze_last_n_layers", 0)
        if unfreeze_text_n and hasattr(self.text_encoder, "encoder") and hasattr(self.text_encoder.encoder, "layer"):
            for layer in self.text_encoder.encoder.layer[-unfreeze_text_n:]:
                for p in layer.parameters():
                    p.requires_grad = True

        if audio_pooling == "attention":
            self.audio_attention = nn.Sequential(
                nn.Linear(audio_hidden_size, 1),
                nn.Softmax(dim=1),
            )
        else:
            self.audio_attention = None

        fusion_config = model_config.get("fusion_config", {})
        proj_dim = fusion_config.get("projection_dim", self.hidden_dim)
        self.fusion = GatedInteractiveAttentionFusion(
            audio_dim=audio_hidden_size,
            text_dim=text_hidden_size,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            fusion_config=fusion_config,
            dropout=dropout,
        )
        fusion_out_dim = proj_dim // 2

        head_config = model_config.get("classification_head", {})
        self.fusion.classifier = MultiScaleTemporalClassificationHead(
            hidden_dim=fusion_out_dim,
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

        conflict_config = model_config.get("conflict_aware", {})
        self._conflict_aware_enabled = conflict_config.get("enabled", False)
        if self._conflict_aware_enabled:
            self._audio_aux = nn.Linear(audio_hidden_size, self.num_classes)
            self._text_aux = nn.Linear(text_hidden_size, self.num_classes)
            self._conflict_threshold = conflict_config.get("conflict_threshold", 0.25)
            for m in [self._audio_aux, self._text_aux]:
                nn.init.normal_(m.weight, 0.0, 0.01)
                if m.bias is not None:
                    nn.init.normal_(m.bias, 0.0, 0.01)
        else:
            self._audio_aux = None
            self._text_aux = None

        # No vision in bimodal
        self.vision_encoder = None
        self.mel_transform = None

    def pool_audio_features(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.audio_encoder is None or not hasattr(self, "audio_pooling"):
            return features.mean(dim=1)
        if self.audio_pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand(features.size())
                s = (features * mask).sum(dim=1)
                n = mask.sum(dim=1).clamp(min=1e-9)
                return s / n
            return features.mean(dim=1)
        if self.audio_pooling == "max":
            return features.max(dim=1).values
        if self.audio_pooling == "attention" and self.audio_attention is not None:
            w = self.audio_attention(features)
            if attention_mask is not None:
                w = w.masked_fill(attention_mask.unsqueeze(-1) == 0, float("-inf"))
            w = torch.softmax(w, dim=1)
            return (features * w).sum(dim=1)
        return features.mean(dim=1)

    def forward(
        self,
        audio: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.fusion_type == "gated_interactive_attention":
            return self._forward_bimodal(audio, text_input_ids, text_attention_mask, labels)

        return super().forward(audio, text_input_ids, text_attention_mask, labels)

    def _forward_bimodal(
        self,
        audio: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        audio_out = self.audio_encoder(audio)
        audio_features = audio_out.last_hidden_state
        audio_pooled = self.pool_audio_features(audio_features)

        text_out = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
        )
        text_features = text_out.last_hidden_state[:, 0, :]

        logits = self.fusion(audio_pooled, text_features)

        if self._conflict_aware_enabled and self._audio_aux is not None:
            audio_logits = self._audio_aux(audio_pooled)
            text_logits = self._text_aux(text_features)
            pred_a = audio_logits.argmax(dim=-1)
            pred_t = text_logits.argmax(dim=-1)
            conflict = (pred_a != pred_t).float()
            pa = torch.softmax(audio_logits, dim=-1).max(dim=-1).values
            pt = torch.softmax(text_logits, dim=-1).max(dim=-1).values
            w_a = (pa / (pa + pt + 1e-9)).unsqueeze(-1)
            w_t = (pt / (pa + pt + 1e-9)).unsqueeze(-1)
            blend = w_a * audio_logits + w_t * text_logits
            logits = (1 - conflict).unsqueeze(-1) * logits + conflict.unsqueeze(-1) * blend

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        predictions = logits.argmax(dim=-1)
        return {
            "logits": logits,
            "predictions": predictions,
            "loss": loss,
            "audio_features": audio_pooled,
            "text_features": text_features,
        }
