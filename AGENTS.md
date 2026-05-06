# AGENTS.md

## Project
Multimodal Speech Emotion Recognition on IEMOCAP. PyTorch-based, Windows/PowerShell primary environment. Targets 4 emotion classes: angry, happy, sad, neutral.

## Setup Traps

- **`requirements.txt`** matches the packages exercised by `setup_check.py`; for CUDA, install matching `torch`/`torchaudio` wheels from PyTorch's official instructions if needed. **Study 50** additionally needs **`funasr`** (see commented line at the bottom of `requirements.txt`).
- **Python 3.8+**, GPU strongly recommended for training.

## Dataset

- IEMOCAP: configure `dataset.data_path` in `config/config.yaml`.
- Standard split: sessions 1–3 train, 4 val, 5 test.
- The dataset contains raw audio only. In trimodal studies, the ViT computes **log-mel on the fly from the waveform inside the model** (not from precomputed files on disk).

## Two Workflows: Baseline vs Ablation

The repo has two completely separate training/evaluation paths. Using the wrong script for a checkpoint is the #1 mistake.

| Goal | Script | Model source |
|------|--------|-------------|
| Train baseline | `python src\train.py --config config\config.yaml --experiment_name <name> --device cuda` | `ConformerMultimodalModel` / `AudioOnlyModel` |
| Train ablation | `python scripts\run_ablation.py --study N --device cuda` | `create_ablation_model()` in `scripts/run_ablation.py` |
| Evaluate baseline checkpoint | `python src\evaluate.py --checkpoint <best_model.pt> --config config\config.yaml --split test --device cuda` | Rebuilds baseline model |
| Evaluate ablation by YAML | `python src\evaluate_ensemble.py --checkpoints <ckpt> --configs <yaml> --study_numbers N --weights 1.0 --split test --device cuda` | Loads ablation model class |
| Test-only on saved run | `python scripts\run_test_only.py --checkpoint_dir <run_dir> --study N --device cuda` | Reads `config.json` from run dir |

**Critical:** `src/evaluate.py` does **not** know ablation model classes. Never use it for ablation checkpoints; use `evaluate_ensemble.py` or `run_test_only.py`.

## Ablation Studies

- Valid studies: **1–50, 52–57** (there is **no study 51**).
- Many studies reuse the same model class; `create_ablation_model()` in `scripts/run_ablation.py` is the single source of truth for the mapping.
- Configs live in `config/ablation/ablation_*.yaml`. Catalog and descriptions: `config/ablation/ABLATION_SUMMARY.md`.

## Checkpoint & Run Layout

Typical ablation run directory:
```
checkpoints/ablation_XX/ablation_XX_YYYYMMDD_HHMMSS/
  best_model.pt
  config.json          # saved merged config
  best_metrics.json
  final_results.json   # written by run_test_only.py
  confusion_matrix_test.png
  training_curves.png
```

`run_test_only.py` writes `final_results.json`, confusion matrix, and `test_eval.log` directly into the run directory.

## Key Architecture Notes

- **Encoders:** audio (Wav2Vec2 / WavLM / Emotion2Vec), text (BERT/RoBERTa/DistilBERT), vision (ViT on log-mel computed from waveform).
- **Fusions:** Enhanced Cross-Attention, Conflict-Aware, Gated Interactive, Tensor, GLU, Late, Hierarchical, etc. Defined in `src/models/fusion.py`.
- **Heads:** MLP, LSTM, Transformer, CNN1D, MultiScaleTemporal in `src/ablation_studies/classification_heads.py`.
- **Losses:** Cross Entropy, Focal, Dirichlet, SupCon, composite. Configurable via `src/utils/losses.py`.
- **LR optimizer groups:** `src/ablation_studies/utils.py` defines separate learning rate groups for encoders, fusion, and classifier.

## Quick Commands

```powershell
# Verify environment and dataset path
python setup_check.py

# Baseline training
python src\train.py --config config\config.yaml --experiment_name baseline --device cuda

# Ablation training
python scripts\run_ablation.py --study 34 --device cuda

# TensorBoard
.\start_tensorboard.bat
# or: tensorboard --logdir=logs

# Evaluate ablation with explicit YAML
python src\evaluate_ensemble.py `
  --checkpoints checkpoints\ablation_34\run\best_model.pt `
  --configs config\ablation\ablation_34_conflict_aware_attention.yaml `
  --study_numbers 34 --weights 1.0 --split test --device cuda

# Test-only on existing run (reads config.json)
python scripts\run_test_only.py `
  --checkpoint_dir checkpoints\ablation_34\run\ablation_34_20260203_094355 `
  --study 34 --device cuda
```

## Important File References

- `docs/ARCHITECTURE_AND_WORKFLOWS.md` — detailed mapping of scripts, models, and data flow
- `config/ablation/ABLATION_SUMMARY.md` — canonical catalog of all ablation studies
- `scripts/run_ablation.py` — single source of truth for study number → model class mapping
- `src/ablation_studies/` — all ablation model variants and classification heads
- `src/models/conformer.py` — baseline model
- `src/models/fusion.py` — all fusion mechanisms

## Git Notes

- `.gitignore` excludes `.cursor/`; `docs/` is tracked in git.
- **`openspec/` is gitignored**: change proposals under `openspec/changes/` stay local unless the team removes that rule from `.gitignore`.
- Large checkpoint files (`.pt`, `.pth`) are gitignored.
