# Conformer Multimodal Speech Emotion Recognition on IEMOCAP

This project implements a multimodal (audio + text) emotion recognition system using Conformer architecture for the IEMOCAP dataset.

## Features

- **Multimodal Architecture**: Combines audio (Wav2Vec2/Conformer) and text (DistilBERT) modalities
- **Multiple Fusion Strategies**: Concatenation, cross-attention, and gated fusion
- **7-Class Emotion Recognition**: angry, happy, sad, neutral, frustrated, excited, disgusted
  - *Note: fearful, surprised, and other classes removed due to low sample counts*
- **Speaker-Independent Split**: Session-based split ensuring generalization across different speakers
- **Class Imbalance Handling**: Weighted loss function and focal loss support
- **Advanced Training Features**: 
  - Mixed precision training (optional)
  - Gradient accumulation for effective larger batch sizes
  - Learning rate scheduling (cosine, linear, plateau)
  - Early stopping with configurable patience
- **Ablation Studies Module**: Systematic evaluation of architectural components
- **Comprehensive Evaluation**: WA, UA, F1-scores, confusion matrices, and per-class metrics

## Project Structure

```
CONFORMER-SER/
├── config/
│   ├── config.yaml                    # Main configuration file
│   └── ablation/                      # Ablation study configurations
│       ├── ablation_01_unfreeze_encoders.yaml
│       ├── ablation_02_lstm_head.yaml
│       ├── ablation_03_transformer_head.yaml
│       ├── ablation_04_cnn1d_head.yaml
│       ├── ablation_05_unfreeze_lstm.yaml
│       ├── ablation_06_unfreeze_transformer.yaml
│       ├── ablation_07_unfreeze_mid_high_lstm.yaml
│       └── ablation_08_deep_lstm.yaml
├── src/
│   ├── data/
│   │   ├── dataset.py                 # IEMOCAP dataset implementation
│   │   ├── preprocessing.py           # Audio and text preprocessing
│   │   └── augmentation.py            # Data augmentation utilities
│   ├── models/
│   │   ├── conformer.py               # Conformer multimodal model
│   │   └── fusion.py                  # Fusion modules
│   ├── ablation_studies/              # Ablation studies module
│   │   ├── __init__.py
│   │   ├── base_config.py             # Base config loader
│   │   ├── classification_heads.py    # Alternative classification heads
│   │   ├── ablation_01_model.py       # Unfreeze encoders model
│   │   ├── ablation_02_model.py       # LSTM head model
│   │   ├── ablation_03_model.py       # Transformer head model
│   │   ├── ablation_04_model.py       # CNN 1D head model
│   │   ├── ablation_05_model.py       # Unfreeze encoders + LSTM head
│   │   ├── ablation_06_model.py       # Unfreeze encoders + Transformer head
│   │   ├── ablation_07_model.py       # Unfreeze mid/high layers + LSTM head
│   │   ├── ablation_08_model.py       # Unfreeze encoders + Deep LSTM head
│   │   ├── utils.py                   # Ablation utilities
│   │   └── README.md                  # Ablation studies documentation
│   ├── utils/
│   │   ├── metrics.py                 # Evaluation metrics
│   │   ├── logger.py                  # Logging utilities (UTF-8 support)
│   │   └── losses.py                  # Loss functions (Focal, CrossEntropy)
│   ├── train.py                       # Main training script
│   ├── evaluate.py                    # Evaluation script
│   └── inference.py                   # Inference script
├── scripts/
│   └── run_ablation.py                # Ablation studies runner
├── checkpoints/                       # Model checkpoints
│   ├── 20251029_190659/               # Best performing base model
│   └── ablation_XX/                   # Ablation study results
├── logs/                              # Training logs
├── notebooks/
│   └── data_exploration.ipynb         # Data analysis (optional)
├── requirements.txt
├── README.md
└── TRAINING_IMPROVEMENTS.md           # Training optimization notes
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU training)

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

3. Verify IEMOCAP dataset path in `config/config.yaml`:
```yaml
dataset:
  data_path: "D:\\DOCTORADO\\DATASETS\\IEMOCAP\\archive (1)\\IEMOCAP_full_release"
```

## Dataset

The IEMOCAP (Interactive Emotional Dyadic Motion Capture) dataset contains:
- 10,039 utterances from 10 actors (5 male, 5 female)
- 5 sessions with scripted and improvised scenarios
- Audio recordings and transcriptions
- **7 emotion categories** (angry, happy, sad, neutral, frustrated, excited, disgusted)
  - *Note: fearful, surprised, and other classes excluded due to low sample counts (3, 9, and imbalanced respectively)*

**Session Split**:
- **Training**: Sessions 1, 2, 3
- **Validation**: Session 4
- **Test**: Session 5

**Dataset Structure:**
```
IEMOCAP_full_release/
├── Session1/
│   ├── sentences/
│   │   └── wav/
│   │       └── Ses01F_impro01/
│   │           └── *.wav
│   └── dialog/
│       ├── EmoEvaluation/
│       │   └── *.txt (emotion labels)
│       └── transcriptions/
│           └── *.txt (transcriptions)
├── Session2/
...
```

## Configuration

Edit `config/config.yaml` to customize:

- **Dataset**: Sessions for train/val/test splits, emotion categories
- **Model**: Encoder models, hidden dimensions, fusion type
- **Training**: Batch size, learning rates, epochs, early stopping
- **Evaluation**: Batch size
- **Logging**: TensorBoard, checkpoint directory

### Important Parameters

```yaml
# Session-based split (IEMOCAP has 5 sessions)
train_sessions: [1, 2, 3]
val_sessions: [4]
test_sessions: [5]

# Model
fusion_type: "cross_attention"  # Options: concat, cross_attention, gated, gia
freeze_audio_encoder: true       # Freeze base, unfreeze last N layers
audio_unfreeze_last_n_layers: 8
freeze_text_encoder: true
text_unfreeze_last_n_layers: 0  # Can unfreeze last N layers
dropout: 0.3                     # Increased to combat overfitting
num_classes: 7                   # Updated from 10

# Training (optimized for 8GB GPU)
batch_size: 2
gradient_accumulation_steps: 8   # Effective batch size = 16
learning_rate_audio: 2.5e-7      # Conservative for stability
learning_rate_text: 2.0e-5
learning_rate_fusion: 5.0e-4
weight_decay: 0.05               # Increased L2 regularization
mixed_precision: false           # Disabled to prevent NaN gradients
scheduler_type: "cosine"         # Cosine with warmup
warmup_ratio: 0.1
early_stopping_patience: 8
```

## Usage

### Training

Train the model with default configuration:
```bash
python src/train.py
```

With custom configuration:
```bash
python src/train.py --config config/config.yaml --experiment_name my_experiment
```

Resume from checkpoint:
```bash
python src/train.py --checkpoint checkpoints/checkpoint_epoch_50.pt
```

Use CPU instead of GPU:
```bash
python src/train.py --device cpu
```

### Ablation Studies

Run all ablation studies:
```bash
python scripts/run_ablation.py
```

Run a specific study:
```bash
# Study 01: Unfreeze encoders
python scripts/run_ablation.py --study 1

# Study 02: LSTM classification head
python scripts/run_ablation.py --study 2

# Study 03: Transformer classification head
python scripts/run_ablation.py --study 3

# Study 04: CNN 1D classification head
python scripts/run_ablation.py --study 4

# Study 05: Unfreeze encoders + LSTM head
python scripts/run_ablation.py --study 5

# Study 06: Unfreeze encoders + Transformer head
python scripts/run_ablation.py --study 6

# Study 07: Unfreeze middle/high encoder layers + LSTM head
python scripts/run_ablation.py --study 7

# Study 08: Unfreeze encoders + Deep LSTM head
python scripts/run_ablation.py --study 8
```

For detailed information about ablation studies, see [`src/ablation_studies/README.md`](src/ablation_studies/README.md).

### Evaluation

Evaluate on test set:
```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pt --split test
```

Evaluate on all splits:
```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pt --split all
```

Save predictions to CSV:
```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pt --split test --save_predictions
```

### Monitoring Training

View training progress with TensorBoard:
```bash
tensorboard --logdir=logs
```

Then open http://localhost:6006 in your browser.

## Model Architecture

### Audio Branch
- **Encoder**: Wav2Vec2-Conformer pre-trained model
- **Input**: Raw waveform (16kHz, 10s max)
- **Output**: Audio embeddings (pooled from sequence)

### Text Branch
- **Encoder**: DistilBERT pre-trained model
- **Input**: Transcription (max 128 tokens)
- **Output**: Text embeddings (CLS token)

### Fusion Module
Three fusion strategies available:

1. **Concatenation** (default): Simple concatenation + MLP
2. **Cross-Attention**: Bidirectional cross-attention between modalities
3. **Gated Fusion**: Learnable gating to weight each modality
4. **GIA Fusion (Gated Interaction Attention)**: Bidirectional cross-attention followed by modality-specific gates that control how much cross-modal information is injected into each stream, inspired by the GIA block in the architecture figure.

### Classification Head
- **Base Model**: Multi-layer perceptron with dropout
- **Ablation Variants**:
  - **LSTM Head**: Bidirectional LSTM for sequential feature processing
  - **Transformer Head**: Small transformer encoder with attention pooling
  - **CNN 1D Head**: 1D convolutional layers for pattern extraction
- Output: 7 emotion classes

## Results

### Base Model Performance

Expected performance on IEMOCAP with 7 classes (approximate):

| Metric | Value |
|--------|-------|
| Weighted Accuracy (WA) | 63-68% |
| Unweighted Accuracy (UA) | 55-62% |
| F1-Score (Weighted) | 63-68% |
| F1-Score (Macro) | 58-64% |

**Note**: Results may vary based on:
- Session split configuration
- Hyperparameters
- Random seed
- Hardware (GPU vs CPU)
- Training stability (overfitting mitigation)

### Per-Class Performance

Some emotions are easier to recognize than others:
- **High accuracy**: sad, angry, neutral
- **Moderate accuracy**: happy, excited, frustrated
- **Lower accuracy**: disgusted (class imbalance)

### Ablation Studies Results

Results from ablation studies are saved in `checkpoints/ablation_XX/` directories. Each study includes:
- Best model checkpoint
- Training curves
- Confusion matrices
- Final metrics (JSON format)

Compare results across studies to identify the most effective architectural modifications.

## Training Tips

1. **GPU Memory**: 
   - Default batch_size=2 with gradient_accumulation_steps=8 (effective batch=16)
   - Reduce batch_size if out of memory
   - Use gradient accumulation to maintain effective batch size

2. **Learning Rate**: 
   - Conservative rates for encoders (2.5e-7 for audio, 2.0e-5 for text)
   - Higher rate for fusion module (5.0e-4)
   - Use cosine scheduler with warmup for stability

3. **Overfitting Prevention**:
   - Dropout: 0.3 (increased from 0.2)
   - Weight decay: 0.05 (increased from 0.01)
   - Early stopping patience: 8 epochs
   - Aggressive data augmentation (audio and text)

4. **Class Weights**: 
   - Essential for handling imbalanced dataset
   - Focal loss with class weights recommended

5. **Mixed Precision**: 
   - Currently disabled (was causing NaN gradients)
   - Can be enabled if stability issues are resolved

6. **Data Augmentation**:
   - Audio: Time masking, noise injection, speed perturbation
   - Text: Random masking and insertion
   - Increased augmentation probabilities to combat overfitting

7. **Stability**:
   - Monitor for NaN/Inf gradients
   - Check for class collapse (model predicting only one class)
   - Use gradient clipping (max_grad_norm: 0.5)

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in `config.yaml` (default: 2)
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Reduce `max_length` for audio (default: 10 seconds)
- Disable mixed precision if enabled

### NaN/Inf Gradients
- **Disable mixed precision**: Set `mixed_precision: false`
- **Reduce learning rates**: Especially `learning_rate_audio` and `learning_rate_fusion`
- **Reduce gradient clipping**: Set `max_grad_norm: 0.5` (default)
- **Check input data**: Ensure no NaN values in audio/text inputs

### Class Collapse
If model predicts only one class:
- **Reduce dropout**: Try 0.1-0.2 (if currently higher)
- **Reduce learning rates**: Especially for audio encoder
- **Disable label smoothing**: Set `label_smoothing: 0.0`
- **Reduce weight decay**: Try 0.01 instead of 0.05
- **Check class distribution**: Ensure balanced dataset

### Overfitting
If train accuracy >> validation accuracy:
- **Increase dropout**: Try 0.3-0.5
- **Increase weight decay**: Try 0.05-0.1
- **Reduce learning rates**: More conservative training
- **Increase data augmentation**: More aggressive augmentation
- **Reduce early stopping patience**: Stop earlier to prevent memorization

### Slow Training
- Increase `num_workers` for data loading (default: 2)
- Use smaller pre-trained models (e.g., wav2vec2-base instead of large)
- Consider enabling mixed precision (if stability allows)

### Poor Performance
- Check data preprocessing and session splits
- Verify correct emotion classes (7 classes, not 10)
- Try different fusion strategies (cross_attention recommended)
- Tune learning rates based on training curves
- Check for class imbalance and adjust class weights
- Review ablation studies for architectural improvements

## Common Issues

### Dataset Path Not Found
Ensure the dataset path in `config.yaml` matches your local path:
```yaml
data_path: "D:\\DOCTORADO\\DATASETS\\IEMOCAP\\archive (1)\\IEMOCAP_full_release"
```

### Model Download Issues
Pre-trained models are downloaded from HuggingFace. If you have connection issues:
```bash
# Set environment variable before running
export HF_DATASETS_OFFLINE=0
```

### Windows Path Issues
Use raw strings or double backslashes:
```yaml
data_path: "D:\\DOCTORADO\\DATASETS\\IEMOCAP\\archive (1)\\IEMOCAP_full_release"
```

### Unicode Encoding Errors
The logger now handles UTF-8 encoding automatically. If you still see encoding errors:
- Ensure Python 3.8+ is used
- Check that log files are saved with UTF-8 encoding (handled automatically)

### Checkpoint Base Not Found (Ablation Studies)
If running ablation studies, ensure the base checkpoint exists:
```
checkpoints/20251029_190659/config.json
```
This checkpoint is used as the base configuration for all ablation studies.

## Citation

If you use this code, please cite:

```bibtex
@misc{conformer-iemocap-ser,
  title={Conformer Multimodal Speech Emotion Recognition on IEMOCAP},
  author={Your Name},
  year={2025}
}
```

### IEMOCAP Dataset
```bibtex
@article{busso2008iemocap,
  title={IEMOCAP: Interactive emotional dyadic motion capture database},
  author={Busso, Carlos and Bulut, Murtaza and Lee, Chi-Chun and Kazemzadeh, Abe and Mower, Emily and Kim, Samuel and Chang, Jeannette N and Lee, Sungbok and Narayanan, Shrikanth S},
  journal={Language resources and evaluation},
  volume={42},
  number={4},
  pages={335--359},
  year={2008}
}
```

## License

This project is for research and educational purposes only.

## Acknowledgments

- IEMOCAP dataset creators
- HuggingFace Transformers library
- PyTorch team

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

## Recent Updates

### Version 2.0 (2025-01-29)

- ✅ **Reduced emotion classes**: From 10 to 7 (removed fearful, surprised, other)
- ✅ **Ablation studies module**: Systematic evaluation of architectural components
- ✅ **Alternative classification heads**: LSTM, Transformer, and CNN 1D implementations
- ✅ **Improved training stability**: Better handling of NaN gradients and class collapse
- ✅ **Enhanced regularization**: Increased dropout and weight decay to combat overfitting
- ✅ **Better data augmentation**: More aggressive augmentation for audio and text
- ✅ **UTF-8 logging support**: Fixed Unicode encoding issues on Windows
- ✅ **Text encoder fine-tuning**: Support for unfreezing last N layers of text encoder
- ✅ **Optimized hyperparameters**: Learning rates and schedulers tuned for stability

See [`TRAINING_IMPROVEMENTS.md`](TRAINING_IMPROVEMENTS.md) for detailed optimization notes.

## TODO

- [ ] Add more data augmentation techniques (back-translation for text)
- [ ] Implement cross-validation across sessions
- [ ] Add model ensemble capabilities
- [ ] Support for other datasets (RAVDESS, MELD, etc.)
- [ ] Implement attention visualization for cross-attention fusion
- [ ] Add gradio demo for real-time inference
- [ ] Docker support for easy deployment
- [ ] Model quantization for deployment
- [ ] Hyperparameter search automation
- [ ] Additional ablation studies (different fusion strategies, encoder combinations)

