"""Cache de descriptores acusticos por utterance para los encoders clasicos (111-119).

El cache lo escribe scripts/extraer_clasicas.py una sola vez por corpus:

    cache/clasicas/<corpus>_<fuente>.pt    {'meta': {...}, 'feats': {utterance_id: (T, F) float16}}
    cache/clasicas/<corpus>_<fuente>.json  los mismos metadatos, para no cargar el .pt solo
                                           para saber F (el modelo lo necesita al construirse)

En el entrenamiento:

  * Normalizacion z por descriptor con media y desviacion del TRAIN de cada fold
    (`ajustar_normalizacion`). Nada del test entra en esas estadisticas.
  * Recorte alineado con el de la forma de onda: si base_dataset elige un segmento
    aleatorio de max_length segundos, los descriptores salen del MISMO segmento, asi
    que el prompt de prosodia y el ViT (115) ven lo mismo que el BiLSTM.
  * Relleno con ceros hasta max_length * fps frames y longitud real aparte, para que
    el encoder solo agrupe frames reales.

Lo que NO se aplica a los descriptores: el aumento sobre la forma de onda (ruido,
velocidad, enmascarado temporal). Se calcularon una vez sobre el audio limpio.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch


class ClassicAudioStore:
    def __init__(self, ruta: str):
        ruta = Path(ruta)
        if not ruta.is_file():
            raise FileNotFoundError(
                f"No existe el cache de descriptores {ruta}. Generalo con:\n"
                f"  python scripts\\extraer_clasicas.py --study <N>")
        d = torch.load(ruta, map_location="cpu", weights_only=False)
        self.ruta = str(ruta)
        self.meta: Dict = d["meta"]
        self.feats: Dict[str, torch.Tensor] = d["feats"]
        self.fps = float(self.meta.get("fps", 100.0))
        self.dim = int(self.meta["dim"])
        self.media: Optional[torch.Tensor] = None
        self.desv: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.feats)

    def faltantes(self, ids: Iterable[str]):
        return [u for u in ids if u not in self.feats]

    def ajustar_normalizacion(self, ids_train: Iterable[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Media y desviacion por descriptor sobre los frames de las utterances de TRAIN."""
        s = torch.zeros(self.dim, dtype=torch.float64)
        s2 = torch.zeros(self.dim, dtype=torch.float64)
        n = 0
        for u in ids_train:
            x = self.feats[u].to(torch.float64)
            s += x.sum(0)
            s2 += (x * x).sum(0)
            n += x.shape[0]
        if n == 0:
            raise ValueError("ajustar_normalizacion: ningun frame de entrenamiento")
        media = s / n
        desv = (s2 / n - media * media).clamp(min=0).sqrt()
        desv[desv < 1e-6] = 1.0  # descriptores constantes: no se escalan
        self.media, self.desv = media.float(), desv.float()
        return self.media, self.desv

    def segmento(self, utt: str, inicio_s: float, max_s: float) -> Tuple[torch.Tensor, int]:
        """(max_s * fps, F) normalizado y rellenado con ceros, y la longitud real."""
        if self.media is None:
            raise RuntimeError("ClassicAudioStore sin normalizacion: llama a ajustar_normalizacion")
        x = self.feats[utt].float()
        n_max = max(1, int(round(max_s * self.fps)))
        i0 = min(int(round(inicio_s * self.fps)), max(x.shape[0] - 1, 0))
        x = x[i0:i0 + n_max]
        x = (x - self.media) / self.desv
        largo = int(x.shape[0])
        if largo == 0:  # utterance mas corta que un frame: un frame de ceros
            x = torch.zeros(1, self.dim)
            largo = 1
        if largo < n_max:
            x = torch.cat([x, torch.zeros(n_max - largo, self.dim)], 0)
        return x, largo


def guardar_cache(ruta: str, feats: Dict[str, torch.Tensor], meta: Dict) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".tmp")
    torch.save({"meta": meta, "feats": feats}, tmp)
    tmp.replace(ruta)
    ruta.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
