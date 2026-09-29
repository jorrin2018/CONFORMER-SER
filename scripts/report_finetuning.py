#!/usr/bin/env python
"""
report_finetuning.py -- Build a study's model and report exactly which parts train.

Answers, from the actual objects rather than from the YAML, the two questions the
paper's Implementation Details has to state correctly:
  - which blocks of each encoder are frozen
  - how many parameters are trainable

    python scripts/report_finetuning.py --study 45
    python scripts/report_finetuning.py --study 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fmt(n: int) -> str:
    return f"{n:,}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", type=int, required=True)
    ap.add_argument("--config", default=None, help="override the study's YAML")
    args = ap.parse_args()

    import importlib.util
    import torch  # noqa: F401

    ra_path = ROOT / "scripts" / "run_ablation.py"
    spec = importlib.util.spec_from_file_location("_run_ablation", ra_path)
    ra = importlib.util.module_from_spec(spec)
    sys.modules["_run_ablation"] = ra
    spec.loader.exec_module(ra)
    create_ablation_model = ra.create_ablation_model

    cfg_path = args.config
    if cfg_path is None:
        src = ra_path.read_text(encoding="utf-8")
        marker = f"({args.study}, \""
        if marker not in src:
            print(f"No encuentro el estudio {args.study} en run_ablation.py")
            return 1
        cfg_path = src.split(marker, 1)[1].split('"', 1)[0]

    print(f"\n  estudio {args.study}\n  config  {cfg_path}\n")
    config = yaml.safe_load(open(cfg_path, encoding="utf-8"))

    # SpeechCue adds nine tokens; declare them so the embedding resize is exercised.
    sc = (config.get("text", {}) or {}).get("speechcue", {}) or {}
    if sc.get("enabled"):
        from transformers import AutoTokenizer
        from src.data.speechcue import SPEECHCUE_TOKENS
        tok = AutoTokenizer.from_pretrained(config["text"]["model_name"])
        tok.add_tokens(SPEECHCUE_TOKENS)
        config.setdefault("text", {})["_tokenizer_len"] = len(tok)
        print(f"  SpeechCue activo: vocabulario {len(tok)}\n")

    class _Log:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass

    model = create_ablation_model(args.study, config, "cpu", _Log())

    # SpeechCue keeps the embedding matrix requires_grad=True but masks the gradient
    # of every pre-existing row, so the nominal count overstates what learns.
    masked = getattr(model, "speechcue_frozen_embeddings", 0)

    print(f"\n  {'modulo':<26}{'total':>16}{'entrenable':>16}{'%':>8}")
    print("  " + "-" * 64)
    tot = tr = 0
    for name, mod in model.named_children():
        t = sum(p.numel() for p in mod.parameters())
        r = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        if t == 0:
            continue
        note = ""
        if name == "text_encoder" and masked:
            r -= masked
            note = "  (embeddings enmascarados descontados)"
        tot += t
        tr += r
        print(f"  {name:<26}{fmt(t):>16}{fmt(r):>16}{100.0*r/t:>7.1f}%{note}")
    print("  " + "-" * 64)
    print(f"  {'TOTAL':<26}{fmt(tot):>16}{fmt(tr):>16}{100.0*tr/max(tot,1):>7.1f}%")
    if masked:
        new_emb = getattr(model, "speechcue_new_embeddings", 0)
        dim = new_emb // 9 if new_emb else 0
        print(f"\n  SpeechCue: {fmt(new_emb)} parametros aprendibles "
              f"(9 tokens x {dim} dim). Las otras {fmt(masked // dim if dim else 0)} "
              f"filas de la matriz de embeddings ({fmt(masked)} parametros) reciben "
              f"gradiente cero.")
    print()

    # Grupos del optimizador: src/train.py reparte por subcadena del nombre del
    # parametro. Un encoder que no se llame audio_encoder/text_encoder acaba en el
    # grupo de fusion y entrena con la tasa equivocada, sin aviso ninguno.
    groups = {"audio (lr_audio)": 0, "text (lr_text)": 0, "resto (lr_fusion)": 0}
    for name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        key = ("audio (lr_audio)" if "audio_encoder" in name
               else "text (lr_text)" if "text_encoder" in name
               else "resto (lr_fusion)")
        groups[key] += prm.numel()
    t = config.get("training", {})
    lrs = {"audio (lr_audio)": t.get("learning_rate_audio"),
           "text (lr_text)": t.get("learning_rate_text"),
           "resto (lr_fusion)": t.get("learning_rate_fusion")}
    print("  grupos del optimizador (parametros entrenables):")
    for k, v in groups.items():
        print(f"    {k:<20}{fmt(v):>16}   lr={lrs[k]}")
    print()

    # Per-block detail for the two fine-tuned encoders.
    from src.models.conformer import _transformer_layers
    for enc_name in ("audio_encoder", "text_encoder", "vision_encoder"):
        enc = getattr(model, enc_name, None)
        if enc is None:
            continue
        layers = _transformer_layers(enc)
        if not layers:
            continue
        flags = ["T" if any(p.requires_grad for p in l.parameters()) else "."
                 for l in layers]
        print(f"  {enc_name:<16} bloques (. congelado, T entrenable): {''.join(flags)}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
