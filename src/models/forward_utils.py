"""Llamada unica al forward multimodal, con o sin longitudes de audio.

Los modelos trimodales pueden pedir `audio_lengths` (muestras reales de cada enunciado
antes del relleno a max_length) para promediar solo frames reales o calcular el
log-mel sin el relleno. Lo declaran con el atributo `uses_audio_lengths = True`.

El resto de modelos del repo tiene forward(audio, ids, mask, labels) y no acepta el
argumento; por eso no se pasa a ciegas. Y si un modelo lo pide y el batch no lo trae,
se revienta: un modelo que promedia sobre el relleno porque falto un tensor es el tipo
de degradacion silenciosa que ya costo dos estudios en este repo.
"""

from typing import Mapping, Optional

import torch


def model_uses_audio_lengths(model) -> bool:
    inner = getattr(model, "module", model)  # DataParallel / DDP
    return bool(getattr(inner, "uses_audio_lengths", False))


def model_uses_expert_gate(model) -> bool:
    inner = getattr(model, "module", model)
    return bool(getattr(inner, "uses_expert_gate", False))


def model_uses_classic_audio(model) -> bool:
    inner = getattr(model, "module", model)
    return bool(getattr(inner, "uses_classic_audio", False))


def multimodal_forward(model, audio: torch.Tensor, text_input_ids: torch.Tensor,
                       text_attention_mask: torch.Tensor,
                       labels: Optional[torch.Tensor] = None,
                       batch: Optional[Mapping] = None):
    kwargs = {}
    if model_uses_audio_lengths(model):
        lengths = None if batch is None else batch.get("audio_lengths")
        if lengths is None:
            raise RuntimeError(
                "El modelo usa audio_lengths (pooling enmascarado / log-mel sin relleno) "
                "pero el batch no trae 'audio_lengths'. Usa src.data.collate_fn.")
        kwargs["audio_lengths"] = lengths.to(audio.device)
    if model_uses_expert_gate(model):
        g = None if batch is None else batch.get("expert_g")
        if g is None:
            raise RuntimeError(
                "El modelo toma la compuerta de expertos congelados "
                "(conflict_source: experts) pero el batch no trae 'expert_g'. Los "
                "dataloaders tienen que salir de src.train.create_dataloaders.")
        kwargs["expert_g"] = g.to(audio.device)
    if model_uses_classic_audio(model):
        feats = None if batch is None else batch.get("audio_feats")
        if feats is None:
            raise RuntimeError(
                "El modelo usa el encoder clasico de audio (descriptores por frame) pero "
                "el batch no trae 'audio_feats'. Los dataloaders tienen que salir de "
                "src.train.create_dataloaders con model.clasico en el config.")
        kwargs["audio_feats"] = feats.to(audio.device)
        kwargs["audio_feats_lengths"] = batch["audio_feats_lengths"].to(audio.device)
    return model(audio, text_input_ids, text_attention_mask, labels, **kwargs)
