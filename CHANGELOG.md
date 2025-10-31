# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-10-24

### Added
- Initial release of Conformer multimodal emotion recognition system
- Support for IEMOCAP dataset with 10 emotion categories
- Multimodal architecture combining audio (Conformer/Wav2Vec2) and text (BERT)
- Three fusion strategies: concatenation, cross-attention, and gated fusion
- Comprehensive training pipeline with mixed precision support
- Speaker-independent session splits
- Class imbalance handling with weighted loss
- Evaluation metrics: WA, UA, F1-scores, confusion matrices
- TensorBoard integration for training monitoring
- Complete documentation and examples
- Setup verification script
- Inference script for single audio files
- Data exploration notebook
- Windows batch scripts for easy execution

### Features
- **Dataset**: IEMOCAP parser with emotion labels and transcriptions
- **Preprocessing**: Audio normalization and text tokenization
- **Model**: Conformer/Wav2Vec2 for audio, BERT for text
- **Training**: Early stopping, gradient clipping, mixed precision
- **Evaluation**: Comprehensive metrics and visualization
- **Logging**: TensorBoard and file logging

### Documentation
- Complete README with installation and usage instructions
- Quick start guide for beginners
- API documentation in code
- Example configurations

## Future Releases

### [1.1.0] - Planned
- Data augmentation (SpecAugment, back-translation)
- Cross-validation support
- Model ensemble
- Attention visualization
- Gradio demo interface

### [1.2.0] - Planned
- Support for additional datasets (RAVDESS, MELD, etc.)
- Model quantization for deployment
- Docker support
- Hyperparameter optimization tools

---

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

