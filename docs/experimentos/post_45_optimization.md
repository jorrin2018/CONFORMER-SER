<!--
  Documentación del experimento post–Ablation 45 (OpenSpec: mejoras-post-ablation-45).
  Incluye split de sesiones, semilla, métricas Neutral y advertencia de varianza por semilla.
-->

# Post-45 optimization (sweep v1 y notas de verificación)

## Split y baseline

| Ítem | Valor |
|------|--------|
| Sesiones train / val / test | 1–3 / 4 / 5 (IEMOCAP estándar del repo) |
| Baseline de referencia | `checkpoints/ablation_45/post_fix_conflict_chain_study_45` (`config.json`, `final_results.json`) |
| Config de sweep acotado | `config/ablation/ablation_45_sweep_v1.yaml` (`conflict_threshold: 0.25` vs `0.3` en `ablation_45_multiscale_head.yaml`) |
| Semilla por defecto | `42` (campo `seed` en YAML) |

## Contrato de pérdida (`training.loss`)

- Con `loss.type: cross_entropy`, **`gamma` no modifica la pérdida** (solo `label_smoothing` y pesos de clase). Detalle en `src/utils/losses.py` y en el log de arranque vía `log_loss_contract_traceability`.
- Para focal, usar `loss.type: focal` (o componentes `focal_loss` en `composite_loss`).

## Métricas y Neutral

`EmotionMetrics` (`src/utils/metrics.py`) incluye siempre, por etiqueta en `dataset.emotions`, claves del tipo `neutral_accuracy` y `neutral_f1`. Los JSON agregados por `scripts/run_test_only.py` / flujo ablation (`final_results.json`) deben listar esas claves cuando el evaluador las escribe en `test_metrics`.

**Advertencia (multi-seed):** un único entrenamiento no prueba ganancia definitiva en test; registrar la **semilla** y repetir o documentar incertidumbre según `openspec/.../specs/post-45-optimization/spec.md`.

## Cómo ejecutar un run de verificación

PowerShell (ajustar rutas si hace falta):

```powershell
python src/train.py --config config/ablation/ablation_45_sweep_v1.yaml --num_epochs 1 --experiment_name verify_post45
```

Comprobar en el directorio del experimento bajo `logging.checkpoint_dir`: `config.json` guardado y, tras evaluación test, presencia de `neutral_*` en métricas (vía TensorBoard, logs o `final_results.json` si se usa `run_test_only.py`).

## Fase 2 (WavLM / SupCon + cabeza 45)

La combinación **no** está cableada en `run_ablation.py` (45 → `Ablation34Model`; 49 → `Ablation48Model`). Ver `config/ablation/ablation_phase2_49_wavlm_multiscale_head_FUTURE.yaml` como lista de campos a fusionar cuando exista modelo unificado.

## Ensemble de logits (opcional)

Esqueleto: `scripts/ensemble_logits_avg_eval.py` (promedio de logits en test entre dos checkpoints compatibles).
