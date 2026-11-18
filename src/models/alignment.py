"""Alineacion contrastiva audio-texto (estudios 120-122).

POR QUE. Los trabajos bimodales con mayor UA en IEMOCAP (MGCMA 2024: 80.24) alinean las
dos ramas con perdidas contrastivas. En su propia ablacion, sus tres modulos de
alineacion juntos valen +2.6 pp (77.62 -> 80.24). En CAMF ya medimos la parte que NO es
contrastiva: la co-atencion frame <-> token antes del pooling (101) dio -0.04 pp. Lo que
falta probar es la perdida.

QUE HACE. Dos cabezas de proyeccion pequenas llevan el vector de audio y el de texto de
cada enunciado (los mismos que entran a la fusion) a un espacio comun de `dim`
dimensiones, normalizado. Sobre ese espacio, una de dos perdidas:

  memoria     InfoNCE simetrica audio <-> texto (CLIP / MGCMA a nivel de instancia): el
              par del mismo enunciado es positivo y todo lo demas es negativo. Los
              negativos salen del batch Y de una cola FIFO de los ultimos `memoria`
              vectores de batches anteriores (cross-batch memory, Wang et al. 2020).

  prototipos  Contrastiva supervisada contra prototipos de clase de la OTRA modalidad:
              el audio de un enunciado 'happy' tiene que parecerse al prototipo de
              'happy' del texto mas que a los otros tres, y viceversa. Los prototipos
              son medias moviles exponenciales por clase.

POR QUE NO LA CONTRASTIVA DE BATCH A SECAS. Con batch 4 (lo que cabe en la RTX 3070)
cada ancla tiene como mucho 3 negativos, y acumular gradiente no lo arregla: los
negativos son los del MISMO forward (docs/REWRITE_NOTES.md, seccion 10). La cola y los
prototipos no dependen del tamano del batch.

La perdida se devuelve en outputs['loss_contrastive'] y train_epoch la suma con el peso
training.loss.contrastive_weight (las dos ramas, con y sin precision mixta). Solo existe
en entrenamiento: validacion, test y la seleccion del modelo no cambian.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

TIPOS = ("memoria", "prototipos")


def _proyector(entrada: int, dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(entrada, entrada // 2), nn.GELU(), nn.Linear(entrada // 2, dim))


class AlineacionContrastiva(nn.Module):
    def __init__(self, audio_dim: int, text_dim: int, tipo: str = "memoria", dim: int = 128,
                 temperatura: float = 0.07, memoria: int = 512, num_classes: int = 4,
                 momento_prototipos: float = 0.9):
        super().__init__()
        if tipo not in TIPOS:
            raise ValueError(f"model.alineacion.tipo={tipo!r}; usa uno de {TIPOS}")
        self.tipo = tipo
        self.temperatura = float(temperatura)
        self.proy_audio = _proyector(audio_dim, dim)
        self.proy_texto = _proyector(text_dim, dim)
        if tipo == "memoria":
            if memoria < 1:
                raise ValueError("model.alineacion.memoria tiene que ser >= 1")
            self.register_buffer("cola_audio", torch.zeros(memoria, dim))
            self.register_buffer("cola_texto", torch.zeros(memoria, dim))
            self.register_buffer("puntero", torch.zeros((), dtype=torch.long))
            self.register_buffer("llenos", torch.zeros((), dtype=torch.long))
        else:
            self.momento = float(momento_prototipos)
            self.register_buffer("proto_audio", torch.zeros(num_classes, dim))
            self.register_buffer("proto_texto", torch.zeros(num_classes, dim))
            self.register_buffer("proto_listo", torch.zeros(num_classes, dtype=torch.bool))

    # ------------------------------------------------------------------ memoria
    @torch.no_grad()
    def _encolar(self, za: torch.Tensor, zt: torch.Tensor) -> None:
        K = self.cola_audio.shape[0]
        for a, t in zip(za, zt):
            i = int(self.puntero)
            self.cola_audio[i] = a
            self.cola_texto[i] = t
            self.puntero.fill_((i + 1) % K)
        self.llenos.fill_(min(K, int(self.llenos) + za.shape[0]))

    def _perdida_memoria(self, za, zt):
        n = int(self.llenos)
        B = za.shape[0]
        objetivo = torch.arange(B, device=za.device)
        cand_t = torch.cat([zt, self.cola_texto[:n].to(zt.dtype)], 0)
        cand_a = torch.cat([za, self.cola_audio[:n].to(za.dtype)], 0)
        perd = 0.5 * (F.cross_entropy(za @ cand_t.t() / self.temperatura, objetivo)
                      + F.cross_entropy(zt @ cand_a.t() / self.temperatura, objetivo))
        self._encolar(za.detach().float(), zt.detach().float())
        return perd

    # --------------------------------------------------------------- prototipos
    @torch.no_grad()
    def _actualizar_prototipos(self, za, zt, labels):
        for c in labels.unique():
            m = labels == c
            ma, mt = F.normalize(za[m].mean(0), dim=-1), F.normalize(zt[m].mean(0), dim=-1)
            if bool(self.proto_listo[c]):
                ma = self.momento * self.proto_audio[c] + (1 - self.momento) * ma
                mt = self.momento * self.proto_texto[c] + (1 - self.momento) * mt
            self.proto_audio[c] = F.normalize(ma, dim=-1)
            self.proto_texto[c] = F.normalize(mt, dim=-1)
            self.proto_listo[c] = True

    def _perdida_prototipos(self, za, zt, labels):
        if labels is None:
            raise RuntimeError("La alineacion por prototipos necesita las etiquetas en el forward")
        zad, ztd = za.detach().float(), zt.detach().float()
        # Clases que aparecen por primera vez: su prototipo arranca en la media del batch.
        nuevas = [c for c in labels.unique() if not bool(self.proto_listo[c])]
        if nuevas:
            self._actualizar_prototipos(zad, ztd, labels)
        mascara = (~self.proto_listo).to(za.device)

        def lado(z, protos):
            logits = z @ protos.to(z.dtype).t() / self.temperatura
            return F.cross_entropy(logits.masked_fill(mascara, float("-inf")), labels)

        perd = 0.5 * (lado(za, self.proto_texto) + lado(zt, self.proto_audio))
        if not nuevas:
            self._actualizar_prototipos(zad, ztd, labels)
        return perd

    # ------------------------------------------------------------------ forward
    def forward(self, audio_vec: torch.Tensor, texto_vec: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        za = F.normalize(self.proy_audio(audio_vec), dim=-1)
        zt = F.normalize(self.proy_texto(texto_vec), dim=-1)
        if self.tipo == "memoria":
            return self._perdida_memoria(za, zt)
        return self._perdida_prototipos(za, zt, labels)
