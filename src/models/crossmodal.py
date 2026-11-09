"""Co-atencion audio <-> texto a nivel de secuencia (estudio 101).

POR QUE EXISTE. En el modelo del articulo (estudio 74) la "cross-attention" de
ConflictAwareCrossAttentionFusion opera sobre TRES vectores ya agrupados: el promedio
de los frames de wav2vec2, el [CLS] de DistilBERT y el [CLS] del ViT
(`torch.stack([audio_emb, text_emb, vision_emb])`). Es self-attention entre tres
tokens: ningun frame de audio ve ninguna palabra, ni al reves. La interaccion entre
modalidades ocurre solo despues de resumir cada una en un vector.

Este modulo anade la pieza que falta, ANTES del pooling: cada frame de audio atiende a
los tokens del texto y cada token del texto atiende a los frames de audio, en paralelo
(co-atencion bidireccional, como Zhao et al., Interspeech 2022, wav2vec 2.0 <-> BERT).
La salida conserva las dimensiones de entrada, asi que el pooling, la fusion y la cabeza
del 74 quedan intactos: el UNICO cambio es que las secuencias llegan enriquecidas con la
otra modalidad.

No necesita marcas de tiempo por palabra: la atencion aprende el emparejamiento
frame-palabra (es el argumento de MulT, Tsai et al. 2019, para secuencias no alineadas).
Por eso funciona igual en IEMOCAP, MELD, CREMA-D y con texto de ASR.

Detalle de la mascara de audio: en el regimen del 74 el audio llega rellenado a 8 s y el
pooling promedia tambien el relleno. Para que el 101 cambie un solo factor, por defecto
NO se enmascaran los frames de relleno (la atencion puede aprender a ignorarlos). Si el
modelo ya calcula la mascara de frames reales (audio_masked_pooling), se usa.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class CrossModalLayer(nn.Module):
    """Una direccion: `q` atiende a `kv`. Post-LN con residual y FFN."""

    def __init__(self, q_dim: int, kv_dim: int, num_heads: int, dropout: float,
                 ffn_mult: int):
        super().__init__()
        if q_dim % num_heads:
            raise ValueError(f"q_dim={q_dim} no es divisible entre num_heads={num_heads}")
        self.attn = nn.MultiheadAttention(q_dim, num_heads, dropout=dropout, kdim=kv_dim,
                                          vdim=kv_dim, batch_first=True)
        self.norm1 = nn.LayerNorm(q_dim)
        self.ffn = nn.Sequential(
            nn.Linear(q_dim, ffn_mult * q_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * q_dim, q_dim),
        )
        self.norm2 = nn.LayerNorm(q_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                kv_pad: Optional[torch.Tensor]) -> torch.Tensor:
        a, _ = self.attn(q, kv, kv, key_padding_mask=kv_pad, need_weights=False)
        q = self.norm1(q + self.drop(a))
        return self.norm2(q + self.drop(self.ffn(q)))


class CrossModalCoAttention(nn.Module):
    """Co-atencion bidireccional audio <-> texto, `num_layers` capas.

    En cada capa las dos direcciones leen las secuencias de la capa anterior (no se
    encadenan dentro de la capa), asi que el resultado no depende del orden.

    Los parametros viven bajo el nombre `crossmodal` del modelo: create_ablation_optimizer
    los manda al grupo `learning_rate_crossmodal` si el config lo define. Son capas
    nuevas, desde cero: con el lr de la fusion (5e-6) apenas se moverian y se leeria un
    falso nulo, que es exactamente lo que paso con los estudios 68/69 (REWRITE_NOTES).
    """

    def __init__(self, audio_dim: int, text_dim: int, num_heads: int = 8,
                 num_layers: int = 1, dropout: float = 0.1, ffn_mult: int = 2):
        super().__init__()
        self.audio_layers = nn.ModuleList([
            CrossModalLayer(audio_dim, text_dim, num_heads, dropout, ffn_mult)
            for _ in range(num_layers)])
        self.text_layers = nn.ModuleList([
            CrossModalLayer(text_dim, audio_dim, num_heads, dropout, ffn_mult)
            for _ in range(num_layers)])

    def forward(self, audio_seq: torch.Tensor, text_seq: torch.Tensor,
                text_attention_mask: torch.Tensor,
                audio_frame_mask: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        audio_seq: (B, S_a, D_a)   text_seq: (B, S_t, D_t)
        text_attention_mask: (B, S_t), 1 = token real
        audio_frame_mask: (B, S_a) opcional, 1 = frame real
        Devuelve las dos secuencias con las mismas formas.
        """
        t_pad = text_attention_mask == 0
        a_pad = None if audio_frame_mask is None else audio_frame_mask == 0
        for a_layer, t_layer in zip(self.audio_layers, self.text_layers):
            nuevo_a = a_layer(audio_seq, text_seq, t_pad)
            nuevo_t = t_layer(text_seq, audio_seq, a_pad)
            audio_seq, text_seq = nuevo_a, nuevo_t
        return audio_seq, text_seq
