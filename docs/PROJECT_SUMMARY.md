# Conformer-IEMOCAP-SER Project Summary

## Overview

This project implements a state-of-the-art **Trimodal (Audio + Text + Visual)** Speech Emotion Recognition (SER) system using Wav2Vec2, DistilBERT with SpeechCue, and Vision Transformer (ViT), trained on the IEMOCAP dataset. The system achieves **~78% Unweighted Accuracy** on the 4-class task (Angry, Happy, Sad, Neutral).

## Project Status: ✅ Complete and Actively Developed

All core components are implemented. **39 ablation studies** explore fusion mechanisms, uncertainty modeling, conflict-aware attention, and SOTA improvements.

## Architecture Overview

### Trimodal Design
- **Audio**: Wav2Vec2-Conformer (raw waveform, last 6 layers unfrozen)
- **Text**: DistilBERT + **SpeechCue** (prosody-aware prompts)
- **Visual**: ViT processing Mel-Spectrograms as 224×224 images
- **Fusion**: Enhanced Cross-Attention (Abl 25) or Conflict-Aware Cross-Attention (Abl 34)
- **Classification**: Bi-LSTM head or Dirichlet head (uncertainty-aware)

### Best Models (as of Dec 2024)
| Model | Test UA | Val UA | Gap | Notes |
|-------|---------|--------|-----|-------|
| **Ablation 34** (Conflict-Aware) | **77.98%** | 80.66% | 2.68% | Best generalization |
| Ablation 25 (Enhanced CA) | 77.44% | 82.30% | 4.86% | Stable baseline |
| Ablation 30 (Dirichlet) | 75.24% | 85.02% | 9.78% | Best validation, overfitting |

## Ablation Studies (1-39)

- **01-10**: Bimodal baseline, classification heads, unfreezing, GIA fusion
- **11-13**: Trimodal (ViT), SpeechCue, base model
- **20-28**: Fusion mechanisms (Hierarchical, Tensor, GMU, Enhanced CA, Late, GLU)
- **30**: Dirichlet Evidence Learning
- **31-35**: Temporal alignment, bucketing, Noise-Aware SpeechCue, Conflict-Aware, Active Learning
- **36-39**: Conflict-Aware + Dirichlet/Focal, OpenSMILE residual, ACN+MIC+BCAF

See `src/ablation_studies/README.md` and `config/ablation/ABLATION_SUMMARY.md` for details.

## Project Structure

```
CONFORMER-SER/
├── config/
│   ├── config.yaml                    # Main configuration
│   └── ablation/                      # 39 ablation configs
├── src/
│   ├── data/                          # IEMOCAP, preprocessing, ProsodyCue
│   ├── models/                        # conformer.py, fusion.py
│   ├── ablation_studies/              # 39 ablation model implementations
│   ├── utils/                         # metrics, logger, losses
│   ├── train.py, evaluate.py, inference.py
│   └── evaluate_ensemble.py           # Ensemble evaluation
├── scripts/
│   └── run_ablation.py                # Run any ablation 1-39
├── docs/                              # Project documentation
├── checkpoints/                       # Model checkpoints per ablation
└── logs/                              # TensorBoard logs
```

## Usage

### Training (Ablation Studies)
```powershell
python scripts/run_ablation.py --study 34   # Conflict-Aware (best)
python scripts/run_ablation.py --study 25   # Enhanced Cross-Attention
python scripts/run_ablation.py --study 30   # Dirichlet
```

### Evaluation
```powershell
python src/evaluate_ensemble.py --checkpoints checkpoints/ablation_34/.../best_model.pt --configs config/ablation/ablation_34_conflict_aware_attention.yaml --study_numbers 34 --weights 1.0 --split test
```

### Inference
```powershell
python src/inference.py --checkpoint checkpoints/ablation_34/.../best_model.pt --audio audio.wav
```

## Key Features

- **Trimodal**: Audio + Text + Visual (ViT on Mel-Spectrograms)
- **4-Class**: Angry, Happy, Sad, Neutral (reduced from 7 for reliability)
- **SpeechCue**: Prosody-to-text prompting for semantic enrichment
- **Conflict-Aware Fusion**: Handles irony/sarcasm via confidence weighting
- **Dirichlet**: Uncertainty-aware classification
- **OpenSMILE**: eGeMAPSv02 features (Ablation 38)
- **Speaker-Independent**: Sessions 1-3 train, 4 val, 5 test

## Documentation

- `README.md` - Main project documentation
- `src/ablation_studies/README.md` - Ablation studies guide
- `config/ablation/ABLATION_SUMMARY.md` - Config reference
- `docs/ANALISIS_RESULTADOS_EXPERIMENTOS.md` - Results analysis
- `docs/COMANDOS_EVALUACION.md` - Evaluation commands

## Technical Specifications

- **Python**: 3.8+
- **PyTorch**: 2.0+
- **CUDA**: 11.7+ (recommended)
- **Dataset**: IEMOCAP (~5K utterances, 4 classes)

## System Requirements

- **Recommended**: 32GB RAM, NVIDIA GPU 12GB+ VRAM
- **Training**: ~4-8 hours per ablation (depending on GPU)

---

**Date**: March 2025  
**Version**: 5.0  
**Status**: Production Ready  
**License**: MIT

