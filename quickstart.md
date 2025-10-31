# Quick Start Guide

This guide will help you get started with training the Conformer multimodal emotion recognition model on IEMOCAP.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) with at least 12GB VRAM
- IEMOCAP dataset downloaded and extracted

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
python setup_check.py
```

This will check:
- Python version
- Required packages
- PyTorch and CUDA
- Project structure
- Dataset path

## Configuration

### 1. Update Dataset Path

Edit `config/config.yaml` and update the dataset path:

```yaml
dataset:
  data_path: "YOUR_PATH_TO_IEMOCAP/IEMOCAP_full_release"
```

### 2. Adjust Settings (Optional)

You can modify other settings in `config/config.yaml`:
- Batch size (reduce if out of memory)
- Learning rates
- Number of epochs
- Session splits

## Training

### Basic Training

```bash
python src/train.py
```

### Custom Configuration

```bash
python src/train.py --config config/config.yaml --experiment_name my_exp
```

### Monitor Training

In a separate terminal:

```bash
tensorboard --logdir=logs
```

Then open http://localhost:6006

## Evaluation

### Evaluate Best Model

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pt --split test
```

### Save Predictions

```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pt --split test --save_predictions --output_dir results
```

## Inference

### Single Audio File

```bash
python src/inference.py --checkpoint checkpoints/best_model.pt --audio path/to/audio.wav --text "Optional transcription"
```

## Expected Training Time

- **With GPU (RTX 3090)**: ~4-6 hours for 50 epochs
- **With GPU (RTX 2080 Ti)**: ~6-8 hours for 50 epochs
- **With CPU**: Not recommended (would take days)

## Expected Performance

After training, you should see:
- **Weighted Accuracy (WA)**: 65-72%
- **Unweighted Accuracy (UA)**: 62-68%
- **F1-Score**: 64-70%

## Troubleshooting

### Out of Memory

Reduce batch size in `config/config.yaml`:
```yaml
training:
  batch_size: 8  # or even 4
```

### Dataset Not Found

Double-check the path in `config/config.yaml` matches your dataset location.

### Slow Training

Make sure you're using GPU:
```bash
python src/train.py --device cuda
```

Check GPU usage:
```bash
nvidia-smi
```

### Low Accuracy

- Train for more epochs (up to 100)
- Try different fusion types: `concat`, `cross_attention`, or `gated`
- Adjust learning rates

## Next Steps

1. **Explore Data**: Check out `notebooks/data_exploration.ipynb`
2. **Experiment**: Try different fusion strategies and hyperparameters
3. **Analyze Results**: Look at confusion matrices and per-class performance
4. **Fine-tune**: Adjust model architecture and training settings

## Common Commands Summary

```bash
# Setup verification
python setup_check.py

# Training
python src/train.py

# Evaluation
python src/evaluate.py --checkpoint checkpoints/best_model.pt --split test

# Inference
python src/inference.py --checkpoint checkpoints/best_model.pt --audio audio.wav

# TensorBoard
tensorboard --logdir=logs
```

## Need Help?

Check the main README.md for more detailed information or open an issue on GitHub.

