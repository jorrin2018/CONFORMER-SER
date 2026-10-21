---
title: Evaluation how-to
description: PowerShell commands for ablation checkpoints, ensembles, and common pitfalls.
---

# Evaluation

`evaluate.py` targets the **standard** multimodal checkpoint API. Many **ablation** checkpoints use different module signatures; use `evaluate_ensemble.py` (even for a single model) with matching YAML configs.

## Quick interactive menu (Windows)

```powershell
.\EVALUAR_RAPIDO.bat
```

## Single ablation checkpoints

**Important:** Prefer `src\evaluate_ensemble.py` with the original ablation YAML for architecture dispatch.

### Ablation 25 (enhanced cross-attention)

```powershell
python src\evaluate_ensemble.py `
    --checkpoints checkpoints\ablation_25\ablation_25_20251125_092101\best_model.pt `
    --configs config\ablation\ablation_25_enhanced_cross_attention.yaml `
    --study_numbers 25 `
    --weights 1.0 `
    --split test `
    --output_dir evaluation_results\ablation_25 `
    --device cuda
```

### Ablation 34 (conflict-aware)

```powershell
python src\evaluate_ensemble.py `
    --checkpoints checkpoints\ablation_34\ablation_34_20251205_155722\best_model.pt `
    --configs config\ablation\ablation_34_conflict_aware_attention.yaml `
    --study_numbers 34 `
    --weights 1.0 `
    --split test `
    --output_dir evaluation_results\ablation_34 `
    --device cuda
```

### Ablation 30 (Dirichlet — strong val, watch generalization)

```powershell
python src\evaluate_ensemble.py `
    --checkpoints checkpoints\ablation_30\ablation_30_20251126_175535\best_model.pt `
    --configs config\ablation\ablation_30_dirichlet.yaml `
    --study_numbers 30 `
    --weights 1.0 `
    --split test `
    --output_dir evaluation_results\ablation_30 `
    --device cuda
```

Update checkpoint folders to match your runs.

## Weighted ensemble (three checkpoints)

```powershell
python src\evaluate_ensemble.py `
    --checkpoints `
        checkpoints\ablation_25\ablation_25_20251125_092101\best_model.pt `
        checkpoints\ablation_34\ablation_34_20251205_155722\best_model.pt `
        checkpoints\ablation_30\ablation_30_20251126_175535\best_model.pt `
    --configs `
        config\ablation\ablation_25_enhanced_cross_attention.yaml `
        config\ablation\ablation_34_conflict_aware_attention.yaml `
        config\ablation\ablation_30_dirichlet.yaml `
    --study_numbers 25 34 30 `
    --weights 0.4 0.35 0.25 `
    --split test `
    --method weighted_average `
    --output_dir evaluation_results\ensemble `
    --device cuda
```

The script expects **YAML** configs that match each study’s constructor (not only `config.json` snapshots). If a path fails, align YAMLs with the checkpoint’s training recipe.

### Batch helper

```powershell
.\evaluate_ensemble.bat
```

Edit checkpoint paths inside the `.bat` file before running.

## Metrics output layout

- Single runs: `evaluation_results\<study>\metrics_test.txt`
- Ensemble: `evaluation_results\ensemble\ensemble_metrics_test.txt`

Ensembles often add **~2–4%** UA vs the best singleton on this project, but gains depend on diversity and weighting.

## Practical notes

1. **`evaluate.py` vs `evaluate_ensemble.py`:** The ensemble driver knows how to rebuild several ablation classes; plain `evaluate.py` may hit shape or class mismatches.
2. **VRAM:** Lower `batch_size` in YAML or pass `--device cpu` (slow).
3. **Mismatched architectures:** Always pair checkpoint + YAML + `--study_numbers` consistently.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Model architecture mismatch | Supply the YAML used during training; verify `--study_numbers`. |
| CUDA OOM | Reduce `batch_size` or use CPU. |
| Checkpoint not found | Use repo-relative paths from the project root. |

## See also

- Experiment index: [../experiments/overview.md](../experiments/overview.md)
- Setup: [../getting-started.md](../getting-started.md)
