# Conformer Multimodal Speech Emotion Recognition on IEMOCAP

This project implements a state-of-the-art **Trimodal (Audio + Text + Visual)** emotion recognition system using Conformer, DistilBERT, and Vision Transformer (ViT) architectures for the IEMOCAP dataset.

## Features

- **Trimodal Architecture**: Combines three powerful streams:
  - **Audio**: Wav2Vec2-Conformer for raw waveform processing.
  - **Text**: DistilBERT with **SpeechCue** (prosody-aware prompts) for semantic processing.
  - **Visual**: Vision Transformer (ViT) processing Mel-Spectrograms as images to capture spectral patterns.
- **4-Class Emotion Recognition**: Focused on **Angry, Happy, Sad, Neutral**.
  - *Note: Reduced from 7 classes to improve practical performance and eliminate ambiguity (e.g., Happy vs Excited).*
- **Advanced Fusion**: 
  - **Enhanced Cross-Attention**: Deep multi-layer attention between Audio, Text, and Vision.
  - **Conflict-Aware Cross-Attention**: Detects modality contradictions (irony, sarcasm) and weights by confidence. Supports three conflict detection methods: `prediction_divergence`, `feature_distance`, and `learned`. See [docs/CONFLICT_DETECTION.md](docs/CONFLICT_DETECTION.md).
  - **Gated Fusion (GMU)**: Learnable gates to control information flow.
  - **Tensor Fusion**: Captures high-order interactions.
  - **Hierarchical Fusion**: Multi-stage modality integration.
- **Uncertainty & Robustness**:
  - **Dirichlet Evidence Learning**: Models epistemic uncertainty for better calibration.
  - **Noise-Aware SpeechCue**: Injects [UNCERTAIN] token when background noise is detected.
  - **OpenSMILE Residual**: eGeMAPSv02 features with attention gates and residual connections.
- **Robust Training**:
  - **Speaker-Independent Split**: Ensures generalization to unseen speakers.
  - **Unfreezing Strategy**: Fine-tuning of top encoder layers for domain adaptation.
  - **Class Imbalance Handling**: Weighted loss and Focal Loss.
- **High Performance**: Achieves **~80% Weighted Accuracy** on the 4-class task (State-of-the-art range for this split).

## Project Structure

```
CONFORMER-SER/
├── config/
│   ├── config.yaml                    # Main configuration file
│   └── ablation/                      # Ablation study configurations
│       ├── ablation_01_... to _39_... # Ablation studies 01-39
│       ├── ABLATION_SUMMARY.md        # Detailed summary of all experiments
│       ├── ablation_25_enhanced_cross_attention.yaml
│       ├── ablation_34_conflict_aware_attention.yaml
│       └── ...
├── src/
│   ├── data/
│   │   ├── dataset.py                 # IEMOCAP dataset with ProsodyCue extraction
│   │   ├── preprocessing.py         # Audio, text, and melspec preprocessing
│   ├── models/
│   │   ├── conformer.py              # Base Conformer model
│   │   ├── fusion.py                 # Advanced Fusion modules (GMU, Tensor, etc.)
│   ├── ablation_studies/             # Ablation studies module (see src/ablation_studies/README.md)
│   │   ├── ablation_25_enhanced_cross_attention.py
│   │   ├── ablation_34_model.py       # Conflict-Aware Cross-Attention
│   │   ├── ablation_38_model.py      # OpenSMILE + Residual
│   │   └── ...
│   ├── utils/
│   │   ├── metrics.py                 # Evaluation metrics
│   │   └── logger.py                  # Logging utilities
│   ├── train.py                       # Main training script
│   └── evaluate.py                    # Evaluation script
├── checkpoints/                       # Model checkpoints
├── logs/                              # Training logs
└── README.md
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU training)
- `torchaudio`, `transformers`, `timm` (for ViT)
- **Ablation 50 (Emotion2Vec)**: `pip install funasr` (optional; required only for study 50)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd CONFORMER-SER
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify IEMOCAP dataset path in `config/config.yaml`.

## Model Architecture

**Ablation 25** (Enhanced Cross-Attention) and **Ablation 34** (Conflict-Aware) are the main architectures:

1.  **Audio Branch**: `Wav2Vec2-Conformer` takes raw audio (16kHz). Last 6 transformer layers are unfrozen.
2.  **Visual Branch**: Audio is converted to **Mel-Spectrograms** (224x224 images) processed by `ViT-Base`.
3.  **Text Branch**: `DistilBERT` with SpeechCue processes transcripts.
4.  **Fusion**: **TriModal Enhanced Cross-Attention** (Abl 25) or **Conflict-Aware Cross-Attention** (Abl 34).
    *   Abl 25: Stack of 3 multi-head attention layers; each modality attends to the others.
    *   Abl 34: Adds conflict detection and confidence-weighted resolution for irony/sarcasm.
5.  **Classification**: **Bi-Directional LSTM** head (or Dirichlet head in Abl 30, 36, 37).

### Ablation Series Overview
*   **20-28**: Fusion variants (Hierarchical, Tensor, GMU, Late, GLU, etc.).
*   **30**: Dirichlet Evidence Learning for uncertainty modeling.
*   **31-35**: Temporal alignment, bucketing, Noise-Aware SpeechCue, Conflict-Aware, Active Learning.
*   **36-39**: Conflict-Aware + Dirichlet/Focal, OpenSMILE residual, ACN+MIC+BCAF.

## Usage

### Training
To train the Enhanced Cross-Attention model (Ablation 25) or Conflict-Aware (Ablation 34):

```bash
python scripts/run_ablation.py --study 25   # Enhanced Cross-Attention
python scripts/run_ablation.py --study 34   # Conflict-Aware Cross-Attention
```

To train other variants (Dirichlet, OpenSMILE, etc.):
```bash
python scripts/run_ablation.py --study 30   # Dirichlet Evidence
python scripts/run_ablation.py --study 38   # OpenSMILE + Residual
```

See `src/ablation_studies/README.md` for the full list of studies (1-39).

### Evaluation
Evaluate on the test set (Session 5):

```bash
python src/evaluate.py --checkpoint checkpoints/ablation_25/best_model.pt --split test
# Or for Conflict-Aware (Ablation 34):
python src/evaluate.py --checkpoint checkpoints/ablation_34/best_model.pt --split test
```

## Results

### Performance Comparison

| Model Version | Classes | Architecture | Weighted Accuracy (WA) | Unweighted Accuracy (UA) |
| :--- | :---: | :--- | :---: | :---: |
| **Baseline** | 7 | Audio + Text (Concat) | ~63% | ~55% |
| **Intermediate** | 7 | GIA Fusion + LSTM | ~65% | ~56% |
| **Final (Abl 13)** | **4** | **Trimodal (ViT + SpeechCue)** | **~80.3%** | **~82.0%** |

### Per-Class Accuracy (Final Model)

| Emotion | Accuracy | Status |
| :--- | :---: | :--- |
| **Sad** | **93.5%** | Excellent |
| **Happy** | **84.3%** | Great improvement (resolved Happy/Excited confusion) |
| **Angry** | 77.8% | Solid performance |
| **Neutral** | 72.6% | Hardest class, often ambiguous |

## Recent Updates

### Version 5.0 (2025-03) - Conflict-Aware & SOTA
- ✅ **Conflict-Aware Cross-Attention (Ablation 34)**: Detects modality contradictions and weights by confidence.
- ✅ **Dirichlet Evidence Learning (Ablation 30)**: Uncertainty modeling with Dirichlet classification head.
- ✅ **Series 36-39**: Conflict-Aware + Dirichlet/Focal, OpenSMILE residual, ACN+MIC+BCAF.
- ✅ **Noise-Aware SpeechCue (Ablation 33)**: [UNCERTAIN] token injection for noisy audio.
- ✅ **Temporal Alignment & Bucketing (Ablations 31-32)**: Adaptive DTW and multimodal bucketing.

### Version 4.0 (2025-11-26) - Fusion Odyssey
- ✅ **Advanced Fusion Series (Ablation 20-28)**: 8 fusion strategies (Hierarchical, Tensor, GMU, etc.).
- ✅ **Enhanced Cross-Attention**: Multi-layer stack for deep multimodal interaction.
- ✅ **Documentation**: ABLATION_SUMMARY.md and ablation_studies/README.md.

### Version 3.0 (2025-11-20)
- ✅ **Trimodal Architecture**: Added Vision Transformer branch for Mel-Spectrograms.
- ✅ **4-Class Refinement**: Focused on core emotions to maximize reliability.
- ✅ **SpeechCue Integration**: Internal prosody-to-text prompting implemented.
- ✅ **State-of-the-art Results**: Reached ~80% WA on IEMOCAP (4-class).

### Version 2.0 (2025-10-29)
- ✅ Ablation studies module.
- ✅ LSTM and Transformer classification heads.
- ✅ GIA Fusion.

## Citation

If you use this code, please cite:

```bibtex
@misc{conformer-iemocap-ser,
  title={Trimodal Conformer-ViT Speech Emotion Recognition on IEMOCAP},
  author={Your Name},
  year={2025}
}
```
