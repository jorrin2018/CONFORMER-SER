#!/usr/bin/env python
"""
diagnose_conflict_gate.py -- Por que la compuerta de conflicto no aporta.

El estudio 74 midio que quitarla da +0.86 +- 1.14 pp (t(4)=1.68, IC 95 % incluye 0):
no se detecta beneficio. Este script responde POR QUE, sin reentrenar nada, usando los
best_model.pt del estudio 72 (el que lleva la compuerta activa).

La compuerta hace dos cosas separables, y el nulo puede venir de cualquiera:

  A. DETECTA conflicto: al menos dos de tres ramas discrepan en su argmax.
  B. REPONDERA por confianza en esas muestras, en vez de usar el peso uniforme 1/3.

Dos preguntas, dos respuestas posibles:

  1. ¿HAY ALGO QUE GANAR?  En las muestras de conflicto, se compara la exactitud del
     voto uniforme contra un ORACULO que siempre elige la rama correcta. Ese oraculo
     es el techo de CUALQUIER esquema de reponderacion. Si el margen es pequeño, no
     hay nada que ganar y ninguna variante de la compuerta lo va a arreglar.

  2. ¿LA CONFIANZA INFORMA?  Se mide si la confidence_net predice que rama acierta:
     exactitud de elegir la rama de maxima confianza, frente al azar y frente al
     oraculo. Si no supera al azar, TODAS las variantes basadas en confianza estan
     condenadas -- incluidas feature_distance y learned, que solo cambian A, no B.

Uso:
    python scripts/diagnose_conflict_gate.py --study 72
    python scripts/diagnose_conflict_gate.py --study 72 --folds 1 2

VEREDICTO OBSOLETO -- leer esto antes que la seccion 3 de la salida (22/09/2026).

La seccion "3. VEREDICTO" aplica una regla fija escrita cuando el detector estaba roto
y no sabiamos si habia techo que ganar. Dice "merece la pena correr las variantes,
correrlas LAS TRES". YA NO APLICA: la linea esta cerrada en cuatro configuraciones.

  72  detector roto                          74.71 +- 4.13
  74  sin compuerta                          75.57 +- 3.06   <-- el modelo del articulo
  75  detector entrenado (aux_loss_weight)   75.39 +- 3.38
  76  + confianza calibrada (conf_loss)      75.18 +- 2.85

Ninguna bate al 74. Y dentro del 76, con la confianza entrenada explicitamente para
predecir que rama acierta, seleccionar por confianza SIGUE sin batir al promedio
uniforme: -0.54 +- 3.02 pp pareado por fold (t(4) = -0.40). El techo de oraculo de
~22 pp es real y estable, la confianza informa +20 pp sobre el azar, y aun asi ninguna
regla de seleccion basada en ella supera a promediar. El script sigue siendo util para
MEDIR; su recomendacion final, no.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_ablation import create_ablation_model, load_ablation_config
from src.ablation_studies import load_base_config
from src.train import create_dataloaders

CLASES = ["angry", "happy", "sad", "neutral"]


def _fusion_de(model):
    f = getattr(model, "fusion", None)
    if f is None or not hasattr(f, "last_diag"):
        raise RuntimeError("Este modelo no expone fusion.last_diag; "
                           "¿es un estudio con ConflictAwareCrossAttentionFusion?")
    return f


def recolectar(study, fold, device, logger):
    """Pasa el test del fold por el mejor checkpoint y recoge el diagnostico."""
    base = ROOT / "checkpoints" / f"ablation_{study}" / f"fold_S{fold}_test"
    runs = [d for d in sorted(base.iterdir())
            if (d / "best_model.pt").exists() and (d / "final_results.json").exists()]
    if not runs:
        return None
    run = max(runs, key=lambda d: (d / "final_results.json").stat().st_mtime)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    cfg["compute"]["device"] = device

    _, _, test_loader, _ = create_dataloaders(cfg, logger, audio_only=False)
    model = create_ablation_model(study, cfg, device, logger).to(device).eval()
    ck = torch.load(run / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    fus = _fusion_de(model)

    A, T, V, C, K, Y = [], [], [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            audio = batch["audio"].to(device)
            model(audio,
                  batch["text_input_ids"].to(device),
                  batch["text_attention_mask"].to(device))
            d = fus.last_diag
            A.append(d["audio_logits"].cpu()); T.append(d["text_logits"].cpu())
            V.append(d["vision_logits"].cpu()); C.append(d["conflict"].cpu())
            K.append(d["confidence"].cpu());    Y.append(batch["labels"].cpu())
    cat = lambda xs: torch.cat(xs).numpy()
    return dict(a=cat(A), t=cat(T), v=cat(V), conf=cat(C), k=cat(K), y=cat(Y),
                run=run.name)


def analizar(d):
    a, t, v, conf, k, y = d["a"], d["t"], d["v"], d["conf"].astype(bool), d["k"], d["y"]
    pred = np.stack([a.argmax(1), t.argmax(1), v.argmax(1)], axis=1)   # (N, 3)
    acierta = (pred == y[:, None])                                     # (N, 3)

    def bloque(mask, nombre):
        n = int(mask.sum())
        if n == 0:
            return None
        ac = acierta[mask]
        # voto uniforme = suma de probabilidades con peso 1/3
        probs = (torch.softmax(torch.tensor(a[mask]), -1) +
                 torch.softmax(torch.tensor(t[mask]), -1) +
                 torch.softmax(torch.tensor(v[mask]), -1)).numpy() / 3.0
        unif = (probs.argmax(1) == y[mask]).mean()
        oraculo = ac.any(axis=1).mean()          # techo: alguna rama acierta
        # elegir la rama de maxima confianza estimada
        elegida = k[mask].argmax(1)
        por_conf = ac[np.arange(n), elegida].mean()
        azar = ac.mean()                          # elegir una rama al azar
        return dict(n=n, unif=unif, oraculo=oraculo, por_conf=por_conf, azar=azar,
                    nombre=nombre,
                    ramas=[ac[:, i].mean() for i in range(3)])
    return bloque(conf, "CONFLICTO"), bloque(~conf, "sin conflicto"), bloque(
        np.ones_like(conf), "todo el test"), conf, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", type=int, default=72)
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger("diag")
    logger.info = logger.warning = lambda *a, **k: None

    filas, tasas = [], []
    for fold in args.folds:
        d = recolectar(args.study, fold, args.device, logger)
        if d is None:
            print(f"  S{fold}: sin checkpoint, se salta")
            continue
        confl, sin, todo, mask, y = analizar(d)
        tasas.append(mask.mean())
        filas.append((fold, confl, sin, todo, mask, y))
        print(f"  S{fold} listo ({d['run']})")

    if not filas:
        print("No hay nada que analizar.")
        return 1

    print("\n" + "=" * 74)
    print("  1. ¿CUANTO DISPARA LA COMPUERTA?")
    print("=" * 74)
    print(f"  tasa de conflicto: {100*np.mean(tasas):.1f} % de las utterances de test "
          f"(min {100*min(tasas):.1f}, max {100*max(tasas):.1f})")

    print("\n" + "=" * 74)
    print("  2. ¿HAY ALGO QUE GANAR EN LAS MUESTRAS DE CONFLICTO?")
    print("=" * 74)
    print(f"  {'fold':<6}{'n':>7}{'uniforme':>11}{'por confianza':>15}"
          f"{'rama al azar':>14}{'ORACULO':>10}")
    print("  " + "-" * 70)
    agg = {kk: [] for kk in ("unif", "por_conf", "azar", "oraculo")}
    for fold, confl, *_ in filas:
        if confl is None:
            continue
        for kk in agg:
            agg[kk].append(confl[kk])
        print(f"  S{fold:<5}{confl['n']:>7}{100*confl['unif']:>10.1f}%"
              f"{100*confl['por_conf']:>14.1f}%{100*confl['azar']:>13.1f}%"
              f"{100*confl['oraculo']:>9.1f}%")
    print("  " + "-" * 70)
    m = {kk: 100*np.mean(v) for kk, v in agg.items()}
    print(f"  {'media':<6}{'':>7}{m['unif']:>10.1f}%{m['por_conf']:>14.1f}%"
          f"{m['azar']:>13.1f}%{m['oraculo']:>9.1f}%")

    techo = m["oraculo"] - m["unif"]
    gana_conf = m["por_conf"] - m["azar"]
    print(f"\n  TECHO disponible (oraculo - uniforme): {techo:+.1f} pp")
    print(f"  Lo que aporta la confianza (por confianza - azar): {gana_conf:+.1f} pp")

    print("\n" + "=" * 74)
    print("  3. VEREDICTO")
    print("=" * 74)
    if techo < 3.0:
        print("  TECHO BAJO. En las muestras que la compuerta marca como conflicto casi")
        print("  nunca hay una rama correcta que rescatar: incluso un oraculo perfecto")
        print(f"  ganaria solo {techo:.1f} pp sobre el voto uniforme. El problema esta en")
        print("  la DETECCION, no en la reponderacion. Ninguna variante que solo cambie")
        print("  como se pondera (feature_distance, learned) puede arreglarlo.")
    elif gana_conf < 2.0:
        print(f"  TECHO ALTO ({techo:+.1f} pp) pero la CONFIANZA NO INFORMA: elegir por")
        print(f"  confianza ({m['por_conf']:.1f} %) no mejora sobre elegir al azar")
        print(f"  ({m['azar']:.1f} %). El fallo es de CALIBRACION, no de formulacion.")
        print("  Las tres variantes comparten la misma confidence_net, asi que las tres")
        print("  estan condenadas. La linea prometedora es entrenar la confianza con un")
        print("  objetivo de calibracion, no probar otra regla de deteccion.")
    else:
        print(f"  TECHO ALTO ({techo:+.1f} pp) y la confianza SI informa ({gana_conf:+.1f} pp")
        print("  sobre el azar). El mecanismo tiene fundamento y esta mal explotado:")
        print("  merece la pena correr las variantes. Correrlas LAS TRES, declarandolo")
        print("  de antemano y reportandolas todas.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
