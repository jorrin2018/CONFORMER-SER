"""Señal de conflicto g a partir de expertos congelados (paso 2 de docs/EXPERTOS_CONGELADOS.md).

g(u) = 1 si los tres expertos (audio, texto, imagen) no predicen la misma clase para la
utterance u. Las predicciones salen de cache/experts/fold_S<k>.json, que escribe
scripts/precompute_experts.py:

  - utterances de TRAIN del fold externo: prediccion FUERA DE MUESTRA (cross-fitting,
    'probs_oof'), del modelo interno que dejo fuera su sesion;
  - el resto (validacion y test del fold externo): el experto completo (71 / 73 /
    logistica), que tampoco las vio.

Las dos fuentes son fuera de muestra para cada utterance, asi que la regla vale aunque
otra semilla reparta train/validacion de otra forma: lo unico que se exige es que el
fold (la sesion de test) sea el mismo.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

EXPERTS = ("audio", "text", "vision")


def load_expert_gate(cache_dir, test_session: int, require_oof: bool = True) -> Dict[str, float]:
    path = Path(cache_dir) / f"fold_S{int(test_session)}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Falta {path}. Corre: python scripts/precompute_experts.py --crossfit "
            f"--folds {int(test_session)}")
    d = json.loads(path.read_text(encoding="utf-8"))
    if int(d["meta"]["fold"]) != int(test_session):
        raise ValueError(f"{path} es del fold {d['meta']['fold']}, no del {test_session}")
    oof = d.get("probs_oof")
    if require_oof and not oof:
        raise ValueError(
            f"{path} no tiene predicciones fuera de muestra ('probs_oof'). Sin ellas los "
            f"expertos ya vieron las utterances de train y la compuerta casi no se activa "
            f"al entrenar (37 % frente a 66 % en test). Corre precompute_experts.py --crossfit.")
    gate = {}
    for u, s in d["split"].items():
        src = oof if (oof and s == "train") else d["probs"]
        preds = {int(np.argmax(src[e][u])) for e in EXPERTS}
        gate[u] = 0.0 if len(preds) == 1 else 1.0
    return gate


def check_coverage(gate: Dict[str, float], uids: Iterable[str], name: str) -> float:
    """Revienta si falta alguna utterance; devuelve la tasa de g en ese conjunto."""
    uids = list(uids)
    falta = [u for u in uids if u not in gate]
    if falta:
        raise KeyError(f"{len(falta)} utterances de {name} sin prediccion de los expertos "
                       f"(p. ej. {falta[:3]}): el cache no corresponde a este fold/corpus")
    return float(np.mean([gate[u] for u in uids])) if uids else float("nan")
