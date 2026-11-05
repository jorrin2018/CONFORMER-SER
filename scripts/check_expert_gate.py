#!/usr/bin/env python
"""check_expert_gate.py -- que la compuerta del estudio 97 use DE VERDAD a los expertos.

Antes de gastar GPU en un fold del 97 (paso 2 de docs/EXPERTOS_CONGELADOS.md), monta el
modelo real con el config real y el reparto real de ese fold, y comprueba:

  1. el cache tiene predicciones fuera de muestra y cubre train, val y test del fold;
  2. la tasa de g es parecida en train, val y test (si train sale muy por debajo, el
     cache no es el de cross-fitting y la compuerta entrenaria casi apagada);
  3. en un batch real, la compuerta de la fusion es EXACTAMENTE la g del batch: pesos
     1/3 donde g = 0 y los de la confidence_net donde g = 1;
  4. un paso de optimizacion por la ruta real (multimodal_forward) da perdida finita.

    python scripts\\check_expert_gate.py --study 97 --fold 1
    python scripts\\check_expert_gate.py --study 97 --fold 1 --tiny --no-data   # solo codigo

Sale con 0 si todo cuadra.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import glob
import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_ablation import create_ablation_model, load_ablation_config  # noqa: E402
from src.ablation_studies.utils import create_ablation_optimizer  # noqa: E402
from src.models.forward_utils import multimodal_forward  # noqa: E402

fallos: list[str] = []
MAX_GAP_PP = 10.0


def ok(cond: bool, msg: str) -> None:
    print(f"  [{'OK' if cond else 'FALLO'}] {msg}")
    if not cond:
        fallos.append(msg)


def fold_config(study: int, fold: int, tiny: bool, seed: int) -> dict:
    hits = sorted(glob.glob(str(ROOT / f"config/ablation/ablation_{study}_*.yaml")))
    if not hits:
        raise SystemExit(f"No encuentro el config del estudio {study}")
    if tiny:
        from tests.tiny_encoders import offline_config
        cfg = offline_config(load_ablation_config(hits[0], {}))
    else:
        from src.ablation_studies import load_base_config
        cfg = load_ablation_config(hits[0], load_base_config())
    cfg = copy.deepcopy(cfg)
    d = cfg["dataset"]
    # Mismo reparto que run_ablation_study_kfold con --protocol standard.
    d["train_sessions"] = [s for s in (1, 2, 3, 4, 5) if s != fold]
    d["val_sessions"] = [None]
    d["test_sessions"] = [fold]
    d["val_fraction"] = 0.12
    d.pop("val_speaker", None)
    cfg["seed"] = seed
    cfg.setdefault("compute", {})["num_workers"] = 0
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", type=int, default=97)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tiny", action="store_true", help="encoders diminutos (sin HF)")
    ap.add_argument("--no-data", action="store_true",
                    help="batch sintetico en vez del dataloader real (solo prueba el codigo)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    log = logging.getLogger("check_expert_gate")

    cfg = fold_config(args.study, args.fold, args.tiny, args.seed)
    ca = cfg["model"].get("conflict_aware", {})
    print(f"\n  Estudio {args.study}, fold S{args.fold}, semilla {args.seed}")
    ok(ca.get("conflict_source") == "experts", f"conflict_source = {ca.get('conflict_source')!r}")
    ok(ca.get("enabled", True), "compuerta activada (conflict_aware.enabled)")

    if args.no_data:
        g = torch.tensor([0.0, 1.0, 1.0, 0.0])
        B = len(g)
        batch = {"audio": torch.randn(B, 3 * 16000), "labels": torch.tensor([0, 1, 2, 3]),
                 "text_input_ids": torch.randint(1000, 2000, (B, 24)),
                 "text_attention_mask": torch.ones(B, 24, dtype=torch.long),
                 "audio_lengths": torch.full((B,), 3 * 16000), "expert_g": g}
    else:
        from src.train import create_dataloaders
        from src.data.expert_gate import check_coverage, load_expert_gate
        gate = load_expert_gate(ca.get("expert_cache_dir", "cache/experts"), args.fold)
        train_loader, val_loader, test_loader, _ = create_dataloaders(cfg, log)
        tasas = {}
        for nombre, ld in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
            tasas[nombre] = 100 * check_coverage(
                gate, [d["utterance_id"] for d in ld.dataset.data], nombre)
        print("  tasa de g: " + "   ".join(f"{k} {v:.1f} %" for k, v in tasas.items()))
        ok(abs(tasas["train"] - tasas["test"]) <= MAX_GAP_PP,
           f"g en train y test dentro de {MAX_GAP_PP:.0f} pp "
           f"({tasas['train'] - tasas['test']:+.1f} pp)")
        batch = next(iter(test_loader))
        ok("expert_g" in batch, "el batch trae expert_g")

    sc = (cfg.get("text", {}) or {}).get("speechcue", {}) or {}
    if sc.get("enabled") and not cfg["text"].get("_tokenizer_len"):
        cfg["text"]["_tokenizer_len"] = 30522 + 9
    ctx = contextlib.nullcontext()
    if args.tiny:
        from tests.tiny_encoders import tiny_encoders
        ctx = tiny_encoders()
    torch.manual_seed(0)
    with ctx:
        model = create_ablation_model(args.study, cfg, args.device, log)
    ok(getattr(model, "uses_expert_gate", False), "el modelo pide expert_g")

    dev = args.device
    audio, ids, mask, y = (batch[k].to(dev) for k in
                           ("audio", "text_input_ids", "text_attention_mask", "labels"))
    model.eval()
    with torch.no_grad():
        multimodal_forward(model, audio, ids, mask, y, batch=batch)
    diag = model.fusion.last_diag
    g = batch["expert_g"].to(dev).float()
    ok(torch.equal(diag["gate"].float(), g), "la compuerta de la fusion es la g del batch")
    w = diag["weights"]
    uni = (g == 0)
    if uni.any():
        ok(torch.allclose(w[uni], torch.full_like(w[uni], 1 / 3)), "pesos 1/3 donde g = 0")
    if (~uni).any():
        c = diag["confidence"][~uni]
        ok(torch.allclose(w[~uni], c / c.sum(-1, keepdim=True), atol=1e-6),
           "pesos de la confidence_net donde g = 1")
    aux = diag["conflict"].float()
    print(f"  (diagnostico) los auxiliares habrian dicho g = 1 en el "
          f"{100 * aux.mean().item():.0f} % de este batch; los expertos, {100 * g.mean().item():.0f} %")

    model.train()
    opt = create_ablation_optimizer(model, cfg, args.study)
    out = multimodal_forward(model, audio, ids, mask, y, batch=batch)
    loss = torch.nn.functional.cross_entropy(out["logits"], y)
    loss.backward()
    opt.step()
    ok(bool(torch.isfinite(loss)), f"paso de optimizacion por la ruta real (loss {loss.item():.4f})")

    if fallos:
        print(f"\n  {len(fallos)} comprobacion(es) fallaron. NO lanzar el fold S{args.fold}.\n")
        return 1
    print(f"\n  Todo cuadra. El fold S{args.fold} del estudio {args.study} usa la compuerta de "
          f"los expertos.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
