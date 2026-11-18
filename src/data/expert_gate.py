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


# ------------------------------------------------------------------ estudios 125-126
#: Orden de los 14 rasgos que recibe el detector con cross-fitting (bimodal_fusion.py).
RASGOS = ([f"logp_audio_{i}" for i in range(4)] + [f"logp_text_{i}" for i in range(4)]
          + ["entropia_audio", "entropia_text", "pmax_audio", "pmax_text", "coinciden",
             "log1p_palabras"])


def _palabras(texto: str) -> int:
    import re
    t = re.sub(r"\[[^\]]*\]", " ", (texto or "").lower())
    return len(re.sub(r"[^a-z' ]", " ", t).split())


def load_expert_features(cache_dir, test_session: int, textos: Dict[str, str]):
    """Rasgos de los expertos de audio y texto, FUERA DE MUESTRA, por utterance.

    Misma regla de fuente que load_expert_gate: utterances que el experto uso para
    entrenar ('train' en SU reparto) -> 'probs_oof' (cross-fitting); el resto ->
    'probs' (el experto completo, que no las vio). Asi vale aunque CAMF reparta
    train/validacion con otra semilla.

    Devuelve (rasgos, meta): rasgos[u] = lista de 14 floats, normalizada (z) con la media
    y la desviacion de las utterances 'train' del experto; meta con esas estadisticas.
    `textos`: utterance_id -> transcripcion (para el numero de palabras).
    """
    path = Path(cache_dir) / f"fold_S{int(test_session)}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Falta {path}. Corre: python scripts/precompute_experts.py --crossfit "
            f"--folds {int(test_session)}")
    d = json.loads(path.read_text(encoding="utf-8"))
    if int(d["meta"]["fold"]) != int(test_session):
        raise ValueError(f"{path} es del fold {d['meta']['fold']}, no del {test_session}")
    oof = d.get("probs_oof")
    if not oof:
        raise ValueError(f"{path} no tiene 'probs_oof': corre precompute_experts.py --crossfit")
    crudos = {}
    for u, s in d["split"].items():
        src = oof if s == "train" else d["probs"]
        pa = np.clip(np.asarray(src["audio"][u], dtype=np.float64), 1e-6, 1.0)
        pt = np.clip(np.asarray(src["text"][u], dtype=np.float64), 1e-6, 1.0)
        n = _palabras(textos[u]) if u in textos else 0
        crudos[u] = np.concatenate([
            np.log(pa), np.log(pt),
            [-(pa * np.log(pa)).sum(), -(pt * np.log(pt)).sum(), pa.max(), pt.max(),
             float(pa.argmax() == pt.argmax()), np.log1p(n)]])
    entren = np.stack([crudos[u] for u, s in d["split"].items() if s == "train"])
    media, desv = entren.mean(0), entren.std(0)
    desv[desv < 1e-8] = 1.0
    rasgos = {u: ((x - media) / desv).astype(np.float32).tolist() for u, x in crudos.items()}
    meta = {"fold": int(test_session), "rasgos": RASGOS, "media": media.tolist(),
            "desv": desv.tolist(), "n_train": int(len(entren))}
    return rasgos, meta
