"""Transformador transmodal bidireccional de MulT (Tsai et al., ACL 2019), estudios 124 y 126.

QUE ES "LA PRIMERA PARTE" DE MulT. MulT tiene dos etapas: (1) transformadores
TRANSMODALES, donde la secuencia de una modalidad (objetivo) atiende a la de otra
(fuente), en las dos direcciones; (2) un transformador de autoatencion sobre la
concatenacion de esas salidas para predecir. Aqui se implementa la etapa (1) para
audio y texto:

    texto -> audio : cada frame de wav2vec2 (consulta) atiende a los tokens de DistilBERT
    audio -> texto : cada token (consulta) atiende a los frames de audio

y la salida entra a la fusion de CAMF (dos tokens) en lugar de la etapa (2).

LO QUE LO DISTINGUE DE LA CO-ATENCION DEL ESTUDIO 101 (y es fiel al articulo):
  * La FUENTE es siempre la secuencia de bajo nivel de la otra modalidad (Z^[0]), no la
    salida de la capa anterior: "each crossmodal block adapts directly from the low-level
    feature sequence" (Sec. 3.2). Solo el objetivo se actualiza capa a capa.
  * Pre-LayerNorm (LN antes de la atencion y del FFN), con la LN de la fuente propia de
    cada capa.
  * Proyeccion temporal Conv1d de cada secuencia a una dimension comun d, escalada por
    sqrt(d), mas posiciones sinusoidales (Sec. 3.1).
  * Varias capas (MulT usa 4-5).

No requiere alineacion audio-palabra: la atencion aprende el emparejamiento, que es el
punto de MulT para secuencias no alineadas.

Los parametros viven bajo `crossmodal` en el modelo, asi que van al grupo de
optimizacion `learning_rate_crossmodal` (capas nuevas desde cero).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def posiciones_sinusoidales(T: int, d: int, device, dtype) -> torch.Tensor:
    pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: d // 2])
    return pe.to(dtype)


class BloqueTransmodal(nn.Module):
    """Objetivo x atiende a la fuente fija z0. Pre-LN, residual, FFN (Ec. 3-4 de MulT)."""

    def __init__(self, d: int, heads: int, dropout: float, attn_dropout: float, ffn_mult: int):
        super().__init__()
        self.ln_x = nn.LayerNorm(d)
        self.ln_fuente = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=attn_dropout, batch_first=True)
        self.ln_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, fuente, fuente_pad: Optional[torch.Tensor]):
        q = self.ln_x(x)
        kv = self.ln_fuente(fuente)
        a, _ = self.attn(q, kv, kv, key_padding_mask=fuente_pad, need_weights=False)
        x = x + self.drop(a)
        return x + self.drop(self.ffn(self.ln_ffn(x)))


class MulTTransmodal(nn.Module):
    def __init__(self, audio_dim: int, text_dim: int, d_model: int = 256, num_heads: int = 8,
                 num_layers: int = 4, dropout: float = 0.1, attn_dropout: float = 0.1,
                 embed_dropout: float = 0.1, ffn_mult: int = 4, kernel: int = 1):
        super().__init__()
        if d_model % num_heads:
            raise ValueError(f"d_model={d_model} no es divisible entre num_heads={num_heads}")
        self.d = d_model
        pad = (kernel - 1) // 2
        self.proj_audio = nn.Conv1d(audio_dim, d_model, kernel, padding=pad, bias=False)
        self.proj_texto = nn.Conv1d(text_dim, d_model, kernel, padding=pad, bias=False)
        self.embed_drop = nn.Dropout(embed_dropout)
        bloque = lambda: BloqueTransmodal(d_model, num_heads, dropout, attn_dropout, ffn_mult)
        self.texto_a_audio = nn.ModuleList([bloque() for _ in range(num_layers)])  # objetivo: audio
        self.audio_a_texto = nn.ModuleList([bloque() for _ in range(num_layers)])  # objetivo: texto
        self.ln_audio = nn.LayerNorm(d_model)
        self.ln_texto = nn.LayerNorm(d_model)

    def _embeber(self, x, conv):
        z = conv(x.transpose(1, 2)).transpose(1, 2)                 # (B, T, d)
        z = math.sqrt(self.d) * z + posiciones_sinusoidales(z.size(1), self.d, z.device, z.dtype)
        return self.embed_drop(z)

    def forward(self, audio_seq: torch.Tensor, text_seq: torch.Tensor,
                text_attention_mask: torch.Tensor,
                audio_frame_mask: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """audio_seq (B, S_a, D_a), text_seq (B, S_t, D_t) -> (B, S_a, d), (B, S_t, d)."""
        za0 = self._embeber(audio_seq, self.proj_audio)
        zt0 = self._embeber(text_seq, self.proj_texto)
        t_pad = text_attention_mask == 0
        a_pad = None if audio_frame_mask is None else audio_frame_mask == 0
        za, zt = za0, zt0
        for blq in self.texto_a_audio:
            za = blq(za, zt0, t_pad)
        for blq in self.audio_a_texto:
            zt = blq(zt, za0, a_pad)
        return self.ln_audio(za), self.ln_texto(zt)
