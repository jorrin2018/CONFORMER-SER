"""
Esqueleto para evaluar en test el promedio de logits de dos checkpoints compatibles.

No altera el entrenamiento. Flujo previsto:
  1. Dos directorios de run con el mismo ``config.json`` salvo semilla o cabeza (o mismo
     ``num_classes`` y dimensiones de logits).
  2. Cargar ``best_model.pt`` de cada uno, construir modelo con ``create_ablation_model``
     según ``--study``, cargar pesos.
  3. Crear ``test_loader`` con ``create_dataloaders`` y, por cada batch, obtener
     ``outputs['logits']``, promediar en float32 y aplicar argmax.

Implementación completa: copiar el bucle de ``scripts/run_test_only.validate`` y acumular
predicciones desde logits medios; reutilizar ``EmotionMetrics`` para UA/WA y métricas por
clase (incl. ``neutral_accuracy``, ``neutral_f1``).

Ejemplo (PowerShell, tras rellenar la lógica forward):

    python scripts/ensemble_logits_avg_eval.py --checkpoint_dir_a ... --checkpoint_dir_b ...

Este archivo cumple la tarea OpenSpec 5.1 como punto de partida sin duplicar todo el
código de evaluación hasta que se fijen rutas y estudios concretos.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Sketch: ensemble logits average on test set")
    parser.add_argument("--checkpoint_dir_a", type=str, required=True)
    parser.add_argument("--checkpoint_dir_b", type=str, required=True)
    parser.add_argument("--study", type=int, default=45, help="Ablation study number for both models")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    raise NotImplementedError(
        "Integrar create_ablation_model, cargas de estado y bucle validate con "
        f"promedio de logits; args were {args}"
    )


if __name__ == "__main__":
    main()
