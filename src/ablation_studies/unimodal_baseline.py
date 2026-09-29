"""
Unimodal baselines under the CAMF training regime.

Reviewer 3 asked for the obvious reference point: how far does a single encoder
get, fine-tuned the same way? Without it there is no way to tell whether the
tri-stream fusion contributes anything at all.

These are NOT off-the-shelf baselines. They reuse the same encoder, the same
partial fine-tuning policy, the same projection width, the same MultiScaleTemporal
head, the same optimiser, schedule, label smoothing and folds as study 45. The
only thing that changes is how many streams feed the head -- which is what an
ablation row is supposed to isolate.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from src.models.conformer import (apply_finetuning_policy, build_layer_weights,
                                  weighted_hidden_states)
from .classification_heads import MultiScaleTemporalClassificationHead


class SimpleClassificationHead(nn.Module):
    """Dropout + una capa densa sobre el vector agrupado.

    Por que existe: la cabeza MultiScaleTemporal recibe un vector (B, D), no una
    secuencia -- el audio ya se promedio sobre el tiempo en el forward. Sobre una
    secuencia de longitud 1, sus tres 'pooling types' devuelven tres casi-copias del
    mismo vector ('mean' y 'max' son identicos; 'attention' con una sola clave es una
    proyeccion lineal), y ese (B, 3, D) de duplicados alimenta un BiLSTM de 2 capas:
    9,988,868 parametros, el 19 % de los entrenables, modelando la dinamica temporal
    de tres copias de un vector.

    Esta cabeza es la de Pepino et al. 2021: agrupar en el tiempo y clasificar.
    """

    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(features))

_HEAD_KEYS = (
    "use_temporal_pooling", "pooling_types", "use_lstm", "lstm_num_layers",
    "lstm_bidirectional", "lstm_hidden", "lstm_dropout", "use_self_attention",
    "attention_heads", "final_dropout",
)


class UnimodalBaselineModel(nn.Module):
    """One encoder + the CAMF projection and MST head.

    `model.modality` selects the stream: "audio" (Wav2Vec2/WavLM) or "text"
    (DistilBERT/RoBERTa). The other encoders are never built, so these runs are
    substantially cheaper than the trimodal ones.
    """

    def __init__(self, config: Dict):
        super().__init__()
        mc = config.get("model", {})
        self.modality = mc.get("modality", "audio")
        self.num_classes = mc.get("num_classes", 4)
        self.hidden_dim = mc.get("hidden_dim", 512)
        self.audio_pooling = mc.get("audio_pooling", "mean")
        dropout = mc.get("dropout", 0.3)

        # IMPORTANTE: el nombre del submodulo TIENE que ser audio_encoder / text_encoder.
        # create_optimizer() en src/train.py agrupa los parametros por subcadena del
        # nombre ('audio_encoder' -> lr_audio, 'text_encoder' -> lr_text, el resto ->
        # lr_fusion). Con self.encoder todo el encoder habria caido en el grupo de
        # fusion y habria entrenado a 5e-6 en vez de 3e-5: la linea base saldria
        # artificialmente mala y la conclusion sobre la fusion, invertida.
        if self.modality == "audio":
            name = mc.get("audio_encoder", "facebook/wav2vec2-base")
            self.audio_encoder = AutoModel.from_pretrained(name)
            apply_finetuning_policy(
                self.audio_encoder,
                freeze_all=mc.get("freeze_audio_encoder", False),
                unfreeze_last_n=mc.get("audio_unfreeze_last_n_layers", 0),
                name=f"audio/{name}", freeze_feature_extractor=True,
            )
        elif self.modality == "text":
            name = mc.get("text_encoder", "distilbert-base-uncased")
            self.text_encoder = AutoModel.from_pretrained(name)
            tokenizer_len = config.get("text", {}).get("_tokenizer_len")
            if tokenizer_len:
                emb = self.text_encoder.get_input_embeddings()
                current = emb.weight.shape[0]
                if tokenizer_len != current:
                    self.text_encoder.resize_token_embeddings(tokenizer_len)
                    emb = self.text_encoder.get_input_embeddings()
                    emb.weight.requires_grad = True
                    mask = torch.zeros_like(emb.weight)
                    mask[current:] = 1.0
                    emb.weight.register_hook(lambda g, m=mask: g * m)
                    dim = emb.weight.shape[1]
                    self.speechcue_new_embeddings = (tokenizer_len - current) * dim
                    self.speechcue_frozen_embeddings = current * dim
            apply_finetuning_policy(
                self.text_encoder,
                freeze_all=mc.get("freeze_text_encoder", False),
                unfreeze_last_n=mc.get("text_unfreeze_last_n_layers", 0),
                name=f"text/{name}", freeze_feature_extractor=False,
            )
        else:
            raise ValueError(
                f"model.modality must be 'audio' or 'text', got {self.modality!r}"
            )

        # Suma ponderada de capas ocultas (Pepino et al.): las capas intermedias de
        # wav2vec2 llevan mas informacion paralinguistica que la ultima. El nombre del
        # parametro es deliberado: create_optimizer() agrupa por subcadena y
        # 'audio_encoder_layer_logits' cae en el grupo de audio (3e-5).
        self._weighted_hidden_states = (
            self.modality == "audio"
            and bool(mc.get("audio_weighted_hidden_states", False))
        )
        self.audio_encoder_layer_logits = (
            build_layer_weights(self.audio_encoder) if self._weighted_hidden_states else None
        )

        enc_dim = self.encoder.config.hidden_size  # via the property below
        out_dim = self.hidden_dim // 2          # same width the fusion feeds the head
        self.project = nn.Sequential(
            nn.Linear(enc_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Un solo factor por fila de ablacion: 'multi_scale_temporal' es lo corrido
        # hasta ahora; 'simple' es la misma tuberia con la cabeza de Pepino et al.
        head_type = mc.get("classification_head_type", "multi_scale_temporal")
        if head_type == "simple":
            self.classifier = SimpleClassificationHead(
                hidden_dim=out_dim, num_classes=self.num_classes, dropout=dropout
            )
        elif head_type == "multi_scale_temporal":
            head_cfg = {k: v for k, v in (mc.get("classification_head") or {}).items()
                        if k in _HEAD_KEYS}
            self.classifier = MultiScaleTemporalClassificationHead(
                hidden_dim=out_dim, num_classes=self.num_classes, dropout=dropout, **head_cfg
            )
        else:
            raise ValueError(
                f"model.classification_head_type no soportado en la linea base: "
                f"{head_type!r} (usa 'simple' o 'multi_scale_temporal')"
            )

    @property
    def encoder(self) -> nn.Module:
        """Read-only alias. A property, not an attribute: assigning the module under a
        second name would register it twice and duplicate it in named_parameters()."""
        return self.audio_encoder if self.modality == "audio" else self.text_encoder

    def forward(
        self,
        audio: torch.Tensor,
        text_input_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.modality == "audio":
            if self._weighted_hidden_states:
                feats = weighted_hidden_states(
                    self.audio_encoder, audio, self.audio_encoder_layer_logits)
            else:
                feats = self.audio_encoder(audio).last_hidden_state   # (B, S, H)
            pooled = feats.mean(dim=1) if self.audio_pooling == "mean" else feats.max(dim=1).values
        else:
            out = self.text_encoder(input_ids=text_input_ids,
                                    attention_mask=text_attention_mask)
            pooled = out.last_hidden_state[:, 0, :]            # [CLS]

        logits = self.classifier(self.project(pooled))
        return {"logits": logits, "predictions": logits.argmax(dim=-1), "loss": None}
