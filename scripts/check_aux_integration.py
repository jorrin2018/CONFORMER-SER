#!/usr/bin/env python
"""
check_aux_integration.py -- Que los terminos extra llegan DE VERDAD al entrenamiento.

Por que existe este test y no basta con check_aux_loss.py: aquel probaba el modulo de
fusion en AISLAMIENTO. Confirmaba que el mecanismo funciona, y nunca comprobaba que
estuviera conectado en la ruta real. El 21/09/2026 el estudio 75 se corrio entero -- los
cinco folds -- con el termino auxiliar puesto en la rama de precision mixta de
train_epoch, que con mixed_precision: false NUNCA se ejecuta. Los tres clasificadores
acabaron con los mismos pesos a SEIS DECIMALES que sin el arreglo, y el "resultado"
no media nada.

Este test monta el modelo real con el config real, da un paso de optimizacion por las
mismas funciones que usa el entrenamiento, y comprueba dos cosas distintas:

  aux_loss_weight  -> los tres clasificadores por rama SE MUEVEN con el termino y NO
                      se mueven sin el (sin el reciben gradiente exactamente cero,
                      porque solo los consume un argmax).

  conf_loss_weight -> el termino de calibracion produce gradiente NO NULO sobre la
                      confidence_net. Aqui no sirve el test de movimiento: la
                      confidence_net ya recibe senal por la perdida final de
                      clasificacion, asi que se mueve igual con el termino y sin el.
                      Lo que hay que probar es que el termino nuevo la alcanza, y eso
                      se mide derivando SOLO ese termino.

    python scripts/check_aux_integration.py --study 75
    python scripts/check_aux_integration.py --study 76
"""
from __future__ import annotations

import argparse
import copy
import glob
import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_ablation import create_ablation_model, load_ablation_config
from src.ablation_studies import load_base_config
from src.train import _aux_branch_loss, _conf_calibration_loss

AUX = ("fusion.audio_classifier.weight", "fusion.text_classifier.weight",
       "fusion.vision_classifier.weight")


def _lote(device, B=2):
    torch.manual_seed(1234)
    return (torch.randn(B, 16000 * 3, device=device),
            torch.randint(0, 1000, (B, 32), device=device),
            torch.ones(B, 32, dtype=torch.long, device=device),
            torch.randint(0, 4, (B,), device=device))


def un_paso(cfg, study, device, logger, con_aux: bool):
    """Un paso de optimizacion por la ruta real. Devuelve que se aplico y cuanto se movio."""
    if not con_aux:
        cfg = copy.deepcopy(cfg)
        cfg["model"]["conflict_aware"]["aux_loss_weight"] = 0.0
    torch.manual_seed(0)
    model = create_ablation_model(study, cfg, device, logger).to(device).train()
    antes = {k: v.detach().clone() for k, v in model.named_parameters() if k in AUX}

    audio, ids, mask, y = _lote(device)
    out = model(audio, ids, mask, y)
    loss = torch.nn.functional.cross_entropy(out["logits"], y)

    term, w = _aux_branch_loss(out, y, cfg)
    aux_aplicada = term is not None
    if aux_aplicada:
        loss = loss + w * term

    cterm, cw = _conf_calibration_loss(out, y, cfg)
    conf_aplicada = cterm is not None

    # Gradiente del termino de calibracion SOLO, sobre la confidence_net.
    conf_params = [p for n, p in model.named_parameters()
                   if "confidence_net" in n and p.requires_grad]
    grad_conf = 0.0
    if conf_aplicada and conf_params:
        grads = torch.autograd.grad(cterm, conf_params, retain_graph=True,
                                    allow_unused=True)
        grad_conf = max(float(g.abs().max()) for g in grads if g is not None)
    if conf_aplicada:
        loss = loss + cw * cterm

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    opt.zero_grad(); loss.backward(); opt.step()

    actuales = dict(model.named_parameters())
    movidos = {k: float((actuales[k].detach() - antes[k]).abs().max()) for k in AUX}
    return {
        "aux_aplicada": aux_aplicada,
        "conf_aplicada": conf_aplicada,
        "grad_conf": grad_conf,
        "movidos": movidos,
        "ramas": out.get("branch_logits") is not None,
        "confianza": out.get("branch_confidence") is not None,
        "n_conf_params": len(conf_params),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", type=int, default=76)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger("chk"); logger.info = logger.warning = lambda *a, **k: None

    cfg_path = sorted(glob.glob(str(ROOT / f"config/ablation/ablation_{args.study}_*.yaml")))[0]
    cfg = load_ablation_config(cfg_path, load_base_config())
    cfg["compute"]["device"] = args.device
    ca = cfg.get("model", {}).get("conflict_aware", {})
    w_aux = float(ca.get("aux_loss_weight", 0.0) or 0.0)
    w_conf = float(ca.get("conf_loss_weight", 0.0) or 0.0)

    print(f"\n  config: {Path(cfg_path).name}")
    print(f"  aux_loss_weight declarado : {w_aux}")
    print(f"  conf_loss_weight declarado: {w_conf}")

    r_si = un_paso(cfg, args.study, args.device, logger, True)
    r_no = un_paso(cfg, args.study, args.device, logger, False)

    print(f"\n  el modelo devuelve branch_logits    : {r_si['ramas']}")
    print(f"  el modelo devuelve branch_confidence : {r_si['confianza']}")
    print(f"  parametros en confidence_net         : {r_si['n_conf_params']}")
    print(f"  la perdida auxiliar se aplica        : {r_si['aux_aplicada']}")
    print(f"  la perdida de calibracion se aplica  : {r_si['conf_aplicada']}")

    print(f"\n  {'parametro':<40}{'CON aux':>14}{'SIN aux':>14}")
    print("  " + "-" * 68)
    for k in AUX:
        print(f"  {k:<40}{r_si['movidos'][k]:>14.8f}{r_no['movidos'][k]:>14.8f}")

    ok_aux = True
    if w_aux > 0:
        ok_aux = (r_si["ramas"] and r_si["aux_aplicada"]
                  and all(r_si["movidos"][k] > 1e-9 for k in AUX)
                  and all(r_no["movidos"][k] < 1e-12 for k in AUX))

    ok_conf = True
    if w_conf > 0:
        print(f"\n  |grad| max del termino de calibracion sobre confidence_net: "
              f"{r_si['grad_conf']:.8f}")
        ok_conf = (r_si["confianza"] and r_si["conf_aplicada"]
                   and r_si["n_conf_params"] > 0 and r_si["grad_conf"] > 1e-9)

    print()
    if ok_aux and ok_conf:
        if w_aux > 0:
            print("  Los tres clasificadores SE MUEVEN con la auxiliar y NO se mueven")
            print("  sin ella, por la ruta real de entrenamiento.")
        if w_conf > 0:
            print("  El termino de calibracion produce gradiente no nulo sobre la")
            print("  confidence_net por la ruta real de entrenamiento.")
        print(f"  El estudio {args.study} ya mide lo que dice medir. Lanzalo.\n")
        return 0

    if w_aux > 0 and not r_si["ramas"]:
        print("  FALLO: el modelo no devuelve branch_logits. No lanzar.")
    elif w_aux > 0 and not r_si["aux_aplicada"]:
        print("  FALLO: el termino auxiliar no se construye. No lanzar.")
    elif not ok_aux:
        print("  FALLO: los clasificadores por rama no se mueven como deben. No lanzar.")
    elif w_conf > 0 and not r_si["confianza"]:
        print("  FALLO: el modelo no devuelve branch_confidence. No lanzar.")
    elif w_conf > 0 and not r_si["conf_aplicada"]:
        print("  FALLO: el termino de calibracion no se construye. No lanzar.")
    else:
        print("  FALLO: el termino de calibracion no alcanza a la confidence_net.")
        print("  No lanzar.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
