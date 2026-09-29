#!/usr/bin/env python
"""
report_optimizer.py -- Que tasa de aprendizaje recibe REALMENTE cada parte del modelo.

Existe porque create_ablation_optimizer() tiene dos ramas y la de 'tasa unica' se
aplicaba a estudios cuyos encoders NO estan congelados: los estudios 44, 45 y 66-69
entrenaron TODO -- encoders incluidos -- a learning_rate_fusion = 5e-6, mientras sus
configs declaraban 3e-5 para audio y 2e-5 para texto. Correr esto ANTES de gastar GPU.

    python scripts/report_optimizer.py --study 71
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_ablation import create_ablation_model, load_ablation_config
from src.ablation_studies import load_base_config
from src.ablation_studies.utils import create_ablation_optimizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", type=int, required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg_path = args.config
    if cfg_path is None:
        import glob
        hits = sorted(glob.glob(str(ROOT / f"config/ablation/ablation_{args.study}_*.yaml")))
        if not hits:
            print(f"No encuentro el config del estudio {args.study}")
            return 1
        cfg_path = hits[0]
    print(f"\n  config: {cfg_path}")

    config = load_ablation_config(cfg_path, load_base_config())
    config["compute"]["device"] = "cpu"
    import logging
    model = create_ablation_model(args.study, config, 'cpu', logging.getLogger('report'))
    opt = create_ablation_optimizer(model, config, args.study)

    # A que grupo fue a parar cada parametro
    grupo_de = {}
    for gi, g in enumerate(opt.param_groups):
        for p in g["params"]:
            grupo_de[id(p)] = gi

    resumen = defaultdict(lambda: defaultdict(int))
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        gi = grupo_de.get(id(p))
        # Clasificar por el MODULO, no por subcadena suelta: fusion.vision_proj y
        # fusion.level2_norm_vision llevan "vision" en el nombre pero son parametros
        # de fusion, no del ViT. Etiquetarlos como "encoder de vision" hacia parecer
        # que el ViT estaba descongelado cuando no lo esta.
        if name.endswith("audio_encoder_layer_logits") or name == "audio_encoder_layer_logits":
            parte = "logits de capa (audio)"
        elif name.startswith("audio_encoder.") or ".audio_encoder." in name:
            parte = "encoder de audio"
        elif name.startswith("text_encoder.") or ".text_encoder." in name:
            parte = "encoder de texto"
        elif name.startswith("vision_encoder.") or ".vision_encoder." in name:
            parte = "encoder de vision (ViT)"
        else:
            parte = "fusion / cabeza"
        resumen[parte][gi] += p.numel()

    tr = config["training"]
    print(f"  declarado en el config: audio {tr.get('learning_rate_audio')} | "
          f"texto {tr.get('learning_rate_text')} | fusion {tr.get('learning_rate_fusion')}")
    print(f"  use_param_groups: {tr.get('use_param_groups', False)}")
    print(f"\n  grupos del optimizador: {len(opt.param_groups)}")
    for gi, g in enumerate(opt.param_groups):
        print(f"    grupo {gi}: lr = {g['lr']:g}")

    print(f"\n  {'parte del modelo':<22}{'grupo':>7}{'lr real':>12}{'parametros':>14}")
    print("  " + "-" * 55)
    problema = False
    for parte, grupos in sorted(resumen.items()):
        for gi, n in sorted(grupos.items()):
            lr = opt.param_groups[gi]["lr"] if gi is not None else float("nan")
            print(f"  {parte:<22}{gi if gi is not None else '?':>7}{lr:>12g}{n:>14,}")
            if parte.startswith("encoder") and "vision" not in parte:
                esperado = tr.get("learning_rate_audio" if "audio" in parte else "learning_rate_text")
                if esperado is not None and abs(lr - float(esperado)) > 1e-12:
                    problema = True

    # Congelado vs entrenable por encoder: lo que de verdad hay que mirar para saber
    # si el ViT esta congelado como dice el articulo.
    print(f"\n  {'encoder':<26}{'entrenables':>14}{'totales':>14}{'%':>8}")
    print("  " + "-" * 62)
    for attr, etiqueta in (("audio_encoder", "audio"), ("text_encoder", "texto"),
                           ("vision_encoder", "vision (ViT)")):
        mod = getattr(model, attr, None)
        if mod is None:
            continue
        tot = sum(q.numel() for q in mod.parameters())
        ent = sum(q.numel() for q in mod.parameters() if q.requires_grad)
        print(f"  {etiqueta:<26}{ent:>14,}{tot:>14,}{100 * ent / tot if tot else 0:>7.1f}%")

    if problema:
        print("\n  AVISO: algun encoder entrenable NO esta recibiendo la tasa que declara")
        print("  el config. Pon training.use_param_groups: true si eso no es lo que quieres.")
    else:
        print("\n  Cada parte recibe la tasa que declara el config.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
