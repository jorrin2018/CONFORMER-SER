#!/usr/bin/env python
"""check_sota_studies.py -- comprobacion previa de los estudios 98-103, antes de gastar GPU.

Monta el modelo real con el config real (encoders preentrenados de verdad) y un batch
real del fold, y comprueba lo que cada estudio dice que cambia:

  98   el ViT no se ejecuta y su vector es cero; mismos parametros que el 74.
  99   training.model_selection = last y num_epochs fijas.
  100  el cache de ASR existe, esta completo, cubre el fold y trae WER.
  101  las capas crossmodal existen, tienen su grupo de lr y se mueven tras un paso.
  102  WavLM Base+ carga con pesos preentrenados (solo publica .bin: torch >= 2.6).
  103  RoBERTa-base y su tokenizer cargan; el batch trae ids de RoBERTa.

Ademas, para todos: un paso de optimizacion con perdida finita y, en CUDA, el pico de
memoria de ese paso con el batch del config, para saber si cabe en tu GPU.

    python scripts\\check_sota_studies.py --study 101 --fold 1 --device cuda
    python scripts\\check_sota_studies.py --study 101 --fold 1 --tiny --no-data   # solo codigo

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
    ap.add_argument("--study", type=int, required=True, choices=range(98, 104))
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiny", action="store_true", help="encoders diminutos (sin HF)")
    ap.add_argument("--no-data", action="store_true", help="batch sintetico")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    log = logging.getLogger("check_sota_studies")
    dev = args.device
    cfg = fold_config(args.study, args.fold, args.tiny, args.seed)
    print(f"\n  Estudio {args.study}, fold S{args.fold}, {dev}")

    if args.study == 99:
        t = cfg["training"]
        ok(t.get("model_selection") == "last", "model_selection = last")
        ok(t.get("num_epochs") == 17, f"num_epochs fijas = {t.get('num_epochs')}")

    if args.no_data:
        B = int(cfg["training"]["batch_size"])
        S = 24
        batch = {"audio": torch.randn(B, int(cfg["audio"]["max_length"] * 16000)),
                 "labels": torch.arange(B) % 4,
                 "text_input_ids": torch.randint(1000, 2000, (B, S)),
                 "text_attention_mask": torch.ones(B, S, dtype=torch.long)}
    else:
        from src.train import create_dataloaders
        train_loader, _, _, _ = create_dataloaders(cfg, log)
        if args.study == 100:
            from src.train import load_transcript_override
            mapa = load_transcript_override(cfg, log)
            uids = [d["utterance_id"] for d in train_loader.dataset.data]
            ok(all(u in mapa for u in uids), "el cache de ASR cubre el train del fold")
            iguales = sum(d["transcription"] == mapa[d["utterance_id"]]
                          for d in train_loader.dataset.data)
            ok(iguales == len(uids), "el dataset usa el texto de ASR")
        batch = next(iter(train_loader))
        if args.study == 103:
            tok = train_loader.dataset.text_preprocessor.tokenizer
            ok("roberta" in type(tok).__name__.lower(), f"tokenizer = {type(tok).__name__}")

    ctx = contextlib.nullcontext()
    if args.tiny:
        from tests.tiny_encoders import tiny_encoders
        ctx = tiny_encoders()
    torch.manual_seed(0)
    with ctx:
        model = create_ablation_model(args.study, cfg, dev, log)
    if args.study == 102 and not args.tiny:
        ok("wavlm" in type(model.audio_encoder).__name__.lower(),
           f"encoder de audio = {type(model.audio_encoder).__name__}")
    if args.study == 103 and not args.tiny:
        ok("roberta" in type(model.text_encoder).__name__.lower(),
           f"encoder de texto = {type(model.text_encoder).__name__}")
    if args.study == 98:
        ok(model._vision_stream == "off", "vision_stream = off")
        llamadas = []
        model.vision_encoder.register_forward_hook(lambda *a: llamadas.append(1))

    audio, ids, mask, y = (batch[k].to(dev) for k in
                           ("audio", "text_input_ids", "text_attention_mask", "labels"))
    opt = create_ablation_optimizer(model, cfg, args.study)
    if args.study == 101:
        cm = [p for n, p in model.named_parameters() if n.startswith("crossmodal.")]
        ok(bool(cm), f"capas crossmodal: {sum(p.numel() for p in cm):,} parametros")
        lr = [g["lr"] for g in opt.param_groups
              if {id(p) for p in g["params"]} == {id(p) for p in cm}]
        ok(lr == [cfg["training"]["learning_rate_crossmodal"]], f"grupo de lr propio: {lr}")
        antes = [p.detach().clone() for p in cm]

    model.train()
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    out = multimodal_forward(model, audio, ids, mask, y, batch=batch)
    loss = torch.nn.functional.cross_entropy(out["logits"], y)
    loss.backward()
    opt.step()
    ok(bool(torch.isfinite(loss)), f"paso de optimizacion (loss {loss.item():.4f})")
    if dev == "cuda":
        pico = torch.cuda.max_memory_allocated() / 2**30
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        ok(pico < 0.92 * total, f"pico de memoria {pico:.2f} GB de {total:.1f} GB")
    if args.study == 98:
        ok(not llamadas, "el ViT no se ejecuto")
        ok(int(torch.count_nonzero(out["vision_features"])) == 0, "vector de vision = 0")
    if args.study == 101:
        ok(any(not torch.equal(a, p) for a, p in zip(antes, cm)), "los pesos crossmodal se movieron")

    n = sum(p.numel() for p in model.parameters())
    nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parametros: {n:,} totales, {nt:,} entrenables")
    if fallos:
        print(f"\n  {len(fallos)} comprobacion(es) fallaron. NO lanzar el estudio {args.study}.\n")
        return 1
    print(f"\n  Todo cuadra para el estudio {args.study}.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
