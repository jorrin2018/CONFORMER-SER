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
  - **Gated Fusion (GMU)**: Learnable gates to control information flow.
  - **Tensor Fusion**: Captures high-order interactions.
  - **Hierarchical Fusion**: Multi-stage modality integration.
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
│       ├── ablation_01_... to _28_... # Ablation studies from 01 to 28
│       ├── ABLATION_SUMMARY.md        # Detailed summary of all experiments
│       ├── ablation_21_hierarchical_fusion.yaml
│       ├── ablation_25_enhanced_cross_attention.yaml
│       └── ...
├── src/
│   ├── data/
│   │   ├── dataset.py                 # IEMOCAP dataset with ProsodyCue extraction
│   │   ├── preprocessing.py           # Audio, text, and melspec preprocessing
│   ├── models/
│   │   ├── conformer.py               # Base Conformer model
│   │   ├── fusion.py                  # Advanced Fusion modules (GMU, Tensor, etc.)
│   ├── ablation_studies/              # Ablation studies module
│   │   ├── ablation_25_enhanced_cross_attention.py # TriModal Enhanced Fusion
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

## Model Architecture (Ablation 25 - Current Best)

The most advanced architecture (**Ablation 25**) integrates three modalities with **Enhanced Cross-Attention**:

1.  **Audio Branch**: `Wav2Vec2-Conformer` takes raw audio (16kHz). Last 6 transformer layers are unfrozen.
2.  **Visual Branch**: Audio is converted to **Mel-Spectrograms** (224x224 images) processed by `ViT-Base`.
3.  **Text Branch**: `DistilBERT` processes transcripts.
4.  **Fusion**: **TriModal Enhanced Cross-Attention Fusion**.
    *   Stack of 3 multi-head attention layers.
    *   Allows deep interaction where each modality attends to the others (Audio ↔ Text ↔ Vision).
    *   Residual connections and LayerNorm for stability.
5.  **Classification**: A **Bi-Directional LSTM** head captures final dependencies.

### Other Fusion Variants Explored (Series 20+)
*   **Hierarchical Fusion**: (Audio + Text) then result + Vision.
*   **Tensor Fusion**: Full outer product of modalities.
*   **Gated Fusion (GMU)**: Learnable gates to control information flow.
*   **Adaptive Weighted**: Dynamic scalar weights per modality.

## Usage

### Training
To train the Enhanced Cross-Attention model (Ablation 25):

```bash
python scripts/run_ablation.py --study 25
```

To train other fusion variants (e.g., Gated Fusion):
```bash
python scripts/run_ablation.py --study 26
```

### Evaluation
Evaluate on the test set (Session 5):

```bash
python src/evaluate.py --checkpoint checkpoints/ablation_25/best_model.pt --split test
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

### Version 4.0 (2025-11-26) - Fusion Odyssey
- ✅ **Advanced Fusion Series (Ablation 20-28)**: Implemented 8 new fusion strategies.
  - Hierarchical, Tensor, Bilinear, Adaptive Weighted, Enhanced Cross-Attention, GMU, Late Fusion, GLU.
- ✅ **Enhanced Cross-Attention**: New multi-layer stack for deep multimodal interaction.
- ✅ **Documentation**: Comprehensive ABLATION_SUMMARY.md detailing all experiments.
- ✅ **Notion Integration**: Auto-generated technical report with results database.

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
