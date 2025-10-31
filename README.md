# Conformer Multimodal Speech Emotion Recognition on IEMOCAP

This project implements a multimodal (audio + text) emotion recognition system using Conformer architecture for the IEMOCAP dataset.

## Features

- **Multimodal Architecture**: Combines audio (Conformer/Wav2Vec2) and text (BERT) modalities
- **Multiple Fusion Strategies**: Concatenation, cross-attention, and gated fusion
- **10-Class Emotion Recognition**: angry, happy, sad, neutral, frustrated, excited, fearful, surprised, disgusted, other
- **Speaker-Independent Split**: Ensures generalization across different speakers
- **Class Imbalance Handling**: Weighted loss function based on class distribution
- **Mixed Precision Training**: Faster training with automatic mixed precision
- **Comprehensive Evaluation**: WA, UA, F1-scores, confusion matrices, and per-class metrics

## Project Structure

```
CONFORMER-SER/
├── config/
│   └── config.yaml              # Configuration file
├── src/
│   ├── data/
│   │   ├── dataset.py          # IEMOCAP dataset implementation
│   │   └── preprocessing.py    # Audio and text preprocessing
│   ├── models/
│   │   ├── conformer.py        # Conformer multimodal model
│   │   └── fusion.py           # Fusion modules
│   ├── utils/
│   │   ├── metrics.py          # Evaluation metrics
│   │   └── logger.py           # Logging utilities
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
├── notebooks/
│   └── data_exploration.ipynb  # Data analysis (optional)
├── requirements.txt
└── README.md
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
- Emotion labels for 10 categories

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
# Speaker-independent split
train_sessions: [1, 2, 3, 4, 5]
val_sessions: [6]
test_sessions: [7, 8]

# Model
fusion_type: "concat"  # Options: concat, cross_attention, gated

# Training
batch_size: 16
learning_rate_audio: 1.0e-5
learning_rate_text: 2.0e-5
learning_rate_fusion: 5.0e-5
num_epochs: 100
early_stopping_patience: 10
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

### Classification Head
- Multi-layer perceptron with dropout
- Output: 10 emotion classes

## Results

Expected performance on IEMOCAP (approximate):

| Metric | Value |
|--------|-------|
| Weighted Accuracy (WA) | 65-72% |
| Unweighted Accuracy (UA) | 62-68% |
| F1-Score (Weighted) | 64-70% |
| F1-Score (Macro) | 60-65% |

**Note**: Results may vary based on:
- Session split configuration
- Hyperparameters
- Random seed
- Hardware (GPU vs CPU)

### Per-Class Performance

Some emotions are easier to recognize than others:
- **High accuracy**: angry, happy, sad, neutral
- **Moderate accuracy**: frustrated, excited
- **Lower accuracy**: fearful, surprised, disgusted

## Training Tips

1. **GPU Memory**: Reduce batch size if out of memory
2. **Learning Rate**: Use different learning rates for different components
3. **Early Stopping**: Patience of 10 epochs works well
4. **Class Weights**: Essential for handling imbalanced dataset
5. **Mixed Precision**: Speeds up training by ~2x
6. **Data Augmentation**: SpecAugment can improve generalization

## Troubleshooting

### Out of Memory
- Reduce batch size in `config.yaml`
- Use gradient accumulation
- Reduce max audio length

### Slow Training
- Enable mixed precision training
- Increase number of workers for data loading
- Use smaller pre-trained models

### Poor Performance
- Check data preprocessing
- Verify correct session splits
- Increase training epochs
- Try different fusion strategies
- Tune learning rates

## Common Issues

### Dataset Path Not Found
Ensure the dataset path in `config.yaml` matches your local path.

### Model Download Issues
Pre-trained models are downloaded from HuggingFace. If you have connection issues:
```python
# Set environment variable before running
export HF_DATASETS_OFFLINE=0
```

### Windows Path Issues
Use raw strings or double backslashes:
```yaml
data_path: "D:\\DOCTORADO\\DATASETS\\IEMOCAP\\archive (1)\\IEMOCAP_full_release"
```

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

## TODO

- [ ] Add data augmentation (SpecAugment, back-translation)
- [ ] Implement cross-validation
- [ ] Add model ensemble
- [ ] Support for other datasets (RAVDESS, MELD, etc.)
- [ ] Implement attention visualization
- [ ] Add gradio demo for inference
- [ ] Docker support
- [ ] Model quantization for deployment

