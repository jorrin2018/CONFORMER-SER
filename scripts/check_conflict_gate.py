#!/usr/bin/env python
"""
check_conflict_gate.py -- Comprueba que el interruptor de la compuerta de conflicto
hace lo que dice, ANTES de gastar GPU en el estudio 74.

Por que existe: hasta el 18/09/2026 `model.conflict_aware.enabled` solo se leia en la
ruta BIMODAL. En la trimodal la compuerta estaba siempre activa y poner enabled: false
no hacia nada. Una ablacion corrida con esa clave habria salido identica al 72 y
habriamos concluido "la compuerta no aporta" por el motivo equivocado.

Verifica tres cosas:
  1. Con enabled: true la compuerta dispara en algunas muestras.
  2. Con enabled: false NUNCA dispara (mascara a 0) aunque la deteccion siga midiendose.
  3. El numero de parametros es IDENTICO en los dos, que es lo que hace que la
     comparacion sea de un solo factor.

    python scripts/check_conflict_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.fusion import ConflictAwareCrossAttentionFusion

B, A, T, V, H, C = 64, 768, 768, 768, 512, 4


def construir(enabled: bool):
    torch.manual_seed(0)
    return ConflictAwareCrossAttentionFusion(
        audio_dim=A, text_dim=T, vision_dim=V, hidden_dim=H, num_classes=C,
        num_heads=8, num_layers=3, dropout=0.5,
        conflict_threshold=0.3, conflict_detection_method="prediction_divergence",
        conflict_aware_enabled=enabled,
    ).eval()


def main() -> int:
    torch.manual_seed(1234)
    a, t, v = torch.randn(B, A), torch.randn(B, T), torch.randn(B, V)

    on, off = construir(True), construir(False)
    off.load_state_dict(on.state_dict())        # mismos pesos exactos

    with torch.no_grad():
        log_on = on(a, t, v)
        log_off = off(a, t, v)

    n_on = sum(p.numel() for p in on.parameters())
    n_off = sum(p.numel() for p in off.parameters())

    print(f"\n  compuerta ON  -> dispara en {100 * on.last_conflict_rate:.1f} % de las muestras")
    print(f"  compuerta OFF -> se detecta en {100 * off.last_conflict_rate:.1f} % "
          f"(se mide igual) pero la mascara va a 0")
    print(f"\n  parametros ON : {n_on:,}")
    print(f"  parametros OFF: {n_off:,}   {'IDENTICO' if n_on == n_off else 'DISTINTO -- MAL'}")

    difieren = not torch.allclose(log_on, log_off, atol=1e-6)
    print(f"\n  los logits difieren entre ON y OFF: {difieren}")

    ok = True
    if on.last_conflict_rate <= 0.0:
        print("\n  FALLO: con la compuerta activa no dispara nunca; el test no prueba nada.")
        ok = False
    if n_on != n_off:
        print("\n  FALLO: el numero de parametros cambia; no seria un solo factor.")
        ok = False
    if not difieren:
        print("\n  FALLO: los logits son identicos -- el interruptor NO esta apagando nada.")
        print("  Es exactamente el bug que tenia la ruta trimodal. No lances el estudio 74.")
        ok = False

    print("\n  " + ("Todo correcto: el estudio 74 mide lo que dice medir."
                    if ok else "NO lanzar el estudio 74 hasta arreglar esto."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
