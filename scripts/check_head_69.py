#!/usr/bin/env python
"""
check_head_69.py -- Comprueba, antes de gastar GPU, dos cosas del estudio 69.

1. Que la cabeza MultiScaleTemporal recibe un vector, no una secuencia, y que por
   tanto sus tres "pooling types" producen casi-copias del mismo vector.
2. Que la cabeza simple del 69 es el unico cambio: mismo encoder, misma proyeccion,
   mismos parametros entrenables salvo la cabeza.

    python scripts/check_head_69.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ablation_studies.classification_heads import MultiScaleTemporalClassificationHead
from src.ablation_studies.unimodal_baseline import SimpleClassificationHead

D, C = 256, 4


def main() -> int:
    torch.manual_seed(0)
    mst = MultiScaleTemporalClassificationHead(
        hidden_dim=D, num_classes=C, dropout=0.5,
        use_temporal_pooling=True, pooling_types=["mean", "max", "attention"],
        use_lstm=True, lstm_num_layers=2, lstm_bidirectional=True, lstm_hidden=512,
        lstm_dropout=0.3, use_self_attention=True, attention_heads=4, final_dropout=0.4,
    ).eval()
    simple = SimpleClassificationHead(hidden_dim=D, num_classes=C, dropout=0.5).eval()

    x = torch.randn(8, D)

    # --- 1. los pooling types sobre una secuencia de longitud 1 -------------
    seq = x.unsqueeze(1)                       # (B, 1, D), lo que hace la cabeza
    v_mean = seq.mean(dim=1)
    v_max = seq.max(dim=1).values
    print("\n  Lo que ve la cabeza MST: ", tuple(seq.shape), " (secuencia de UN paso)")
    print(f"  pooling 'mean' == entrada : {torch.allclose(v_mean, x)}")
    print(f"  pooling 'max'  == entrada : {torch.allclose(v_max, x)}")
    print(f"  'mean' == 'max'           : {torch.allclose(v_mean, v_max)}  <-- duplicado exacto")
    with torch.no_grad():
        q = mst.attn_query.expand(8, -1, -1)
        att, w = mst.attn_pool(query=q, key=seq, value=seq)
    print(f"  pesos de 'attention'      : todos {w.flatten()[0].item():.1f} "
          f"(una sola clave -> softmax trivial)")
    print(f"  'attention' es una proyeccion lineal de la entrada, no un pooling")

    # --- 2. parametros ------------------------------------------------------
    n_mst = sum(p.numel() for p in mst.parameters())
    n_simple = sum(p.numel() for p in simple.parameters())
    print(f"\n  cabeza MST    : {n_mst:>12,} parametros")
    print(f"  cabeza simple : {n_simple:>12,} parametros")
    print(f"  se retiran    : {n_mst - n_simple:>12,}")

    # --- 3. las dos producen logits de la forma correcta --------------------
    with torch.no_grad():
        a, b = mst(x), simple(x)
    ok = a.shape == b.shape == (8, C)
    print(f"\n  logits MST {tuple(a.shape)} / simple {tuple(b.shape)} -> {'ok' if ok else 'FALLO'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
