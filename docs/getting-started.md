---
title: Getting started
description: Environment setup, dataset path, training, evaluation, and common issues (single source of truth with README).
---

# Getting started

Guide for training and evaluating the Conformer multimodal emotion recognition model on IEMOCAP.

## Prerequisites

- **Python 3.10, 3.11, 3.12, or 3.13** for GPU training (PyTorch CUDA wheels are not published for Python 3.14 as of common CUDA stacks).
- **CUDA-capable GPU** recommended (12 GB+ VRAM typical for trimodal runs).
- **IEMOCAP** downloaded and extracted.
- **PyTorch 2.0+** with CUDA; `torchaudio`, `transformers`, `timm` (for ViT).

## Installation

1. Clone the repository and enter the project directory.

2. Create and activate a virtual environment (example on Windows):

   ```powershell
   py -3.12 -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. **Windows / CUDA:** If `pip install` for CUDA fails (e.g. Python 3.14), use Python 3.12 and run `.\install_pytorch_cuda.ps1` from the repo root if provided.

5. Verify the environment:

   ```powershell
   python tools/setup_check.py
   ```

   This checks Python version, packages, PyTorch/CUDA, project layout, and dataset path.

## Configuration

### Dataset path

Edit `config/config.yaml` and set:

```yaml
dataset:
  data_path: "YOUR_PATH_TO_IEMOCAP/IEMOCAP_full_release"
```

### Optional: Hugging Face token

To avoid unauthenticated download limits, copy `.env.example` to `.env` and set `HF_TOKEN=hf_...` ([token settings](https://huggingface.co/settings/tokens)).

### Other settings

You can adjust batch size, learning rates, epochs, and session splits in `config/config.yaml`. Reduce `batch_size` if you hit GPU OOM.

## Training

### Baseline training

```powershell
python src/train.py
```

### Custom config

```powershell
python src/train.py --config config/config.yaml --experiment_name my_exp
```

### Ablation studies

```powershell
python scripts/run_ablation.py --study 25
```

See [experiments/overview.md](experiments/overview.md) for the full ablation list.

### Monitoring

```powershell
tensorboard --logdir=logs
```

Open `http://localhost:6006`.

## Evaluation

Standard checkpoint evaluation:

```powershell
python src/evaluate.py --checkpoint checkpoints/best_model.pt --split test
```

Ablation and ensemble workflows use different entry points. See [how-to/evaluation.md](how-to/evaluation.md).

## Inference (single file)

```powershell
python src/inference.py --checkpoint checkpoints/best_model.pt --audio path/to/audio.wav --text "Optional transcription"
```

## Expected runtime and performance

- **Runtime** depends on GPU; trimodal training is often several hours per run at tens of epochs.
- **Reported high-performance setups** use the 4-class IEMOCAP split and advanced fusion; see the root [README.md](../README.md) and [research/comprehensive-ablation-analysis.md](research/comprehensive-ablation-analysis.md) for result tables. Generic “65–72% WA” plateaus in this document applied to older baselines and are not a guarantee for every config.

## Troubleshooting

### Out of memory

Lower `training.batch_size` in `config/config.yaml` (e.g. 8 or 4).

### Dataset not found

Confirm `dataset.data_path` matches your local IEMOCAP layout.

### Verify GPU use

```powershell
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

## Next steps

- Experiment index: [experiments/overview.md](experiments/overview.md)
- YAML catalog: [experiments/ablation-configs.md](experiments/ablation-configs.md)
