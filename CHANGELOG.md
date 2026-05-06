# Changelog

All notable changes to this project will be documented in this file.

## [5.0.0] - 2026-04-27

### Added
- Central documentation index in `docs/README.md` with active, operational and historical documents.
- Architecture and workflow guide in `docs/ARCHITECTURE_AND_WORKFLOWS.md`.
- Documentation for the current execution paths:
  - baseline training with `src/train.py`;
  - ablation training with `scripts/run_ablation.py`;
  - baseline evaluation with `src/evaluate.py`;
  - ablation and ensemble evaluation with `src/evaluate_ensemble.py`;
  - test-only checkpoint evaluation with `scripts/run_test_only.py`.
- Ablation 43 entry in `config/ablation/ABLATION_SUMMARY.md`.

### Changed
- Updated `README.md` and `quickstart.md` to reflect the current project layout, Windows/PowerShell commands and the difference between baseline, ablation, ensemble and test-only workflows.
- Updated `src/ablation_studies/README.md` to reflect the study set enforced by `run_ablation.py` (`1-50`, `52-57`; no study 51).
- Clarified that study 51 is not defined in the current `scripts/run_ablation.py` study list.

### Documentation
- Marked older experiment-analysis documents as historical where they mainly cover studies 1-40.
- Linked the canonical ablation catalog: `config/ablation/ABLATION_SUMMARY.md`.
- Documented that checkpoints from ablation models should be evaluated with `src/evaluate_ensemble.py` or `scripts/run_test_only.py`, not generally with `src/evaluate.py`.
- Clarified in `ABLATION_SUMMARY.md`, `README.md`, `docs/ARCHITECTURE_AND_WORKFLOWS.md` and `src/ablation_studies/README.md` that **ViT log-mel is computed on-the-fly from the batch waveform**; IEMOCAP in this pipeline does **not** ship pre-extracted mel features as a separate dataset field.

## [4.0.0] - 2025-11-26

### Added
- **Advanced Fusion Module**: Complete overhaul of `src/models/fusion.py` to support multiple fusion architectures.
  - `TriModalEnhancedCrossAttentionFusion`: Stacked multi-head attention layers.
  - `TriModalGatedFusion`: Gated Multimodal Unit (GMU) implementation.
  - `TriModalHierarchicalFusion`: (Audio+Text) -> +Vision hierarchy.
  - `TriModalTensorFusion`: Outer product fusion (TFN).
  - `TriModalBilinearFusion`: Low-rank bilinear pooling.
  - `TriModalAdaptiveWeightedFusion`: Learnable scalar weights.
- **New Ablation Studies (Series 20-28)**:
  - Configurations covering all new fusion methods (`config/ablation/ablation_2*.yaml`).
  - Specialized model classes in `src/ablation_studies/`.
- **Documentation**:
  - `config/ablation/ABLATION_SUMMARY.md`: Detailed catalog of all 28 experiments.
  - Automated Notion report generation.

### Improved
- **Fusion Architecture**: Modularized fusion logic to easily swap strategies.
- **Cross-Attention**: Increased depth (num_layers) and flexibility (heads, dim) in configuration.
- **Visualization**: Enhanced mermaid diagrams for architecture explanation.

## [3.0.0] - 2025-11-20

### Added
- **Trimodal Architecture**: Integration of Audio, Text, and Vision streams.
- **Vision Branch**: ViT (Vision Transformer) on log-mel views **computed in the model** from the waveform (not pre-stored mel in the dataset).
- **SpeechCue**: Injection of prosodic prompts into the text stream.
- **Ablation Series 13**: Robust baseline establishing the state-of-the-art (~80% Acc).
- **Regularization**: Advanced dropout and weight decay tuning.

## [2.0.0] - 2025-10-29

### Added
- Ablation studies framework (`scripts/run_ablation.py`).
- Support for K-Fold Cross Validation.
- LSTM and Transformer classification heads.
- Initial GIA Fusion implementation.

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
- Attention visualization
- Gradio demo interface

### [1.2.0] - Planned
- Support for additional datasets (RAVDESS, MELD, etc.)
- Model quantization for deployment
- Docker support
- Hyperparameter optimization tools

---

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

