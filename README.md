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
  - **TriModal Cross-Attention**: Jointly attends to audio, text, and visual tokens.
  - **GIA Fusion**: Gated Interaction Attention for bi-modal setups.
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
│       ├── ablation_01_... to _10_... # Previous studies
│       ├── ablation_11_melspec_vit.yaml
│       ├── ablation_12_speechcue_loso.yaml
│       └── ablation_13_melspec_vit_speechcue_unfreeze_vit.yaml
├── src/
│   ├── data/
│   │   ├── dataset.py                 # IEMOCAP dataset with ProsodyCue extraction
│   │   ├── preprocessing.py           # Audio, text, and melspec preprocessing
│   ├── models/
│   │   ├── conformer.py               # Base Conformer model
│   │   ├── fusion.py                  # Fusion modules (including TriModal)
│   ├── ablation_studies/              # Ablation studies module
│   │   ├── ablation_11_model.py       # Trimodal ViT model
│   │   ├── ablation_13_model.py       # Final optimized model
│   │   └── ...
│   ├── utils/
│   │   ├── metrics.py                 # Evaluation metrics
│   │   └── logger.py                  # Logging utilities
│   ├── train.py                       # Main training script
│   └── evaluate.py                    # Evaluation script
├── checkpoints/                       # Model checkpoints (including ablation_13)
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

## Model Architecture (Ablation 13)

The final architecture (**Ablation 13**) integrates three modalities:

1.  **Audio Branch**: `Wav2Vec2-Conformer` takes raw audio (16kHz). The last 6 transformer layers are unfrozen to learn emotional prosody features.
2.  **Visual Branch**: Audio is converted to **Mel-Spectrograms** (treated as 224x224 images). A `Vision Transformer (ViT-Base)` extracts spectral features.
3.  **Text Branch**: `DistilBERT` processes transcripts. Crucially, **SpeechCue** prompts (e.g., *"[PROSODY] high energy, fast pace..."*) are injected at the start of the text to provide prosodic context to the language model.
4.  **Fusion**: A **Tri-Modal Cross-Attention** block projects all features to a common space and uses self-attention to weigh inter-modality importance.
5.  **Classification**: A **Bi-Directional LSTM** head captures final sequential dependencies before the Softmax layer.

## Usage

### Training (Best Model)

To train the best-performing model (Ablation 13 configuration):

```bash
python scripts/run_ablation.py --study 13
```

### Evaluation

Evaluate on the test set (Session 5):

```bash
python src/evaluate.py --checkpoint checkpoints/ablation_13/best_model.pt --split test
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
