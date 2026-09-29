#!/usr/bin/env python
"""
check_aux_loss.py -- Comprueba que los clasificadores por rama SI reciben gradiente
con la perdida auxiliar, y que sin ella NO lo reciben. Antes de gastar GPU.

Por que existe: los tres clasificadores que alimentan la deteccion de conflicto se
consumen via argmax, que no es diferenciable. Durante toda la vida del proyecto se
quedaron en su init normal(0, 0.01) y la compuerta disparaba sobre ruido en el 94.5 %
de las muestras. Este test verifica que el arreglo funciona de verdad.

    python scripts/check_aux_loss.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.fusion import ConflictAwareCrossAttentionFusion

B, A, T, V, H, C = 32, 768, 768, 768, 512, 4
AUX = ("audio_classifier", "text_classifier", "vision_classifier")


def grad_de(usar_aux: bool):
    torch.manual_seed(7)
    fus = ConflictAwareCrossAttentionFusion(
        audio_dim=A, text_dim=T, vision_dim=V, hidden_dim=H, num_classes=C,
        num_heads=8, num_layers=3, dropout=0.1,
        conflict_threshold=0.3, conflict_detection_method="prediction_divergence",
    ).train()
    a, t, v = torch.randn(B, A), torch.randn(B, T), torch.randn(B, V)
    y = torch.randint(0, C, (B,))

    logits = fus(a, t, v)
    loss = nn.functional.cross_entropy(logits, y)
    if usar_aux:
        rama = fus.branch_logits
        loss = loss + 0.3 * sum(nn.functional.cross_entropy(bl, y) for bl in rama) / 3
    fus.zero_grad(); loss.backward()

    out = {}
    for nombre in AUX:
        w = dict(fus.named_parameters())[f"{nombre}.weight"]
        out[nombre] = 0.0 if w.grad is None else float(w.grad.abs().sum())
    # una referencia que siempre entrena
    w = dict(fus.named_parameters())["audio_proj.weight"]
    out["audio_proj (referencia)"] = 0.0 if w.grad is None else float(w.grad.abs().sum())
    out["_tasa_conflicto"] = fus.last_conflict_rate
    return out


def main() -> int:
    sin = grad_de(False)
    con = grad_de(True)

    print(f"\n  {'parametro':<28}{'SIN aux':>14}{'CON aux':>14}")
    print("  " + "-" * 56)
    for k in list(AUX) + ["audio_proj (referencia)"]:
        print(f"  {k:<28}{sin[k]:>14.6f}{con[k]:>14.6f}")
    print("  " + "-" * 56)
    print(f"  tasa de conflicto en init: {100*sin['_tasa_conflicto']:.1f} % "
          f"(esperado ~93.75 % con clasificadores aleatorios)")

    muertos_sin = all(sin[k] == 0.0 for k in AUX)
    vivos_con = all(con[k] > 0.0 for k in AUX)
    ref_ok = sin["audio_proj (referencia)"] > 0.0

    print()
    if not ref_ok:
        print("  FALLO: ni la referencia recibe gradiente; el test esta mal montado.")
        return 1
    if not muertos_sin:
        print("  INESPERADO: los clasificadores reciben gradiente SIN la auxiliar.")
        print("  Entonces el diagnostico del 20/09 hay que revisarlo. No lanzar el 75.")
        return 1
    if not vivos_con:
        print("  FALLO: la perdida auxiliar no llega a los clasificadores. No lanzar.")
        return 1

    print("  Confirmado: sin la auxiliar el gradiente es EXACTAMENTE 0 en los tres")
    print("  clasificadores (el argmax los desconecta), y con ella reciben gradiente.")
    print("  El estudio 75 mide lo que dice medir.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
