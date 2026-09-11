# Changelog

All notable changes to this project will be documented in this file.

## [5.1.1] - 2026-09-10

Follow-up to 5.1.0: align the remaining files that the reorganization left behind.

### Changed
- **`requirements.txt`** rewritten. The core set is unchanged, but the five optional extras are now
  documented instead of a single commented `funasr` line: `funasr` (studies **50 and 57**, both use
  `emotion2vec/emotion2vec_base` — the old comment named only study 50), `opensmile` (study 38),
  and `bitsandbytes` / `xformers` / `triton` (runtime-detected training optimizations). All five are
  guarded imports, so the repository works without them. Added `huggingface-hub` (hard import in
  `src/experiments/train_lightweight.py`), lower bounds on torch/torchaudio/transformers, and a
  reference environment block with the verified installed versions.
- **`setup_check.py`**:
  - Resolves every path against the repository root instead of the current working directory, so it
    now passes when invoked from anywhere (previously `Project Structure` and `Dataset Path` failed
    unless the CWD was the repo root).
  - Structure check extended from 6 to 11 directories and from 15 to 23 files: it now covers
    `config/ablation/`, `src/ablation_studies/`, `scripts/`, `scripts/windows/`, `docs/`,
    `run_ablation.py`, `run_test_only.py`, `evaluate_ensemble.py`, `losses.py` and `AGENTS.md`.
  - Reports which optional extras are installed, naming the studies each one gates.
  - Footer now shows both the baseline and the ablation entry points and the valid study range.
- **`docs/PROJECT_SUMMARY.md`** rewritten. It claimed "39 ablation studies", "39 ablation configs"
  and "run any ablation 1-39"; the real figures are 64 runnable studies (`1-50`, `52-63`), 70 YAML
  configs and 34 model classes. Explains why those three numbers differ. Structure block now
  includes `paper/`, `scripts/windows/`, `tests/` and `src/experiments/`. Translated to Spanish per
  the `docs/` convention. Reported results are kept but labelled as not auto-regenerated, pointing
  at `RANKING_Y_RESULTADOS.md`.

### Fixed — alignment with the post-CIARP-2026 audit
- The audit (documented in `docs/REWRITE_NOTES.md`) invalidates every result measured before the
  `_find_audio_file` fix, which was silently discarding the 2 943 improvised utterances. Removed
  the reference-results table from `docs/PROJECT_SUMMARY.md`, which was reproducing exactly those
  discredited figures (77.98 / 77.44 / 75.24 % UA), and replaced it with the audit warning, the
  verified corpus size (5 531 utterances; train 3 259 / val 1 031 / test 1 241) and the
  requirements for publishable numbers (LOSO 5-fold with mean ± sigma, one factor per row).
- Added the same warning to `README.md` and `docs/README.md`, pointing at `REWRITE_NOTES.md`,
  `dataset_integrity_patch.md` and `scripts/verify_iemocap.py`.
- **Corrected the valid study range from `52-63` to `52-65`** across the documentation: studies 64
  (SpeechCue as described in the paper) and 65 (unfrozen ViT) are now wired into
  `create_ablation_model()`.

### Documentation
- Added scope banners to three historical analyses whose conclusions are drawn from the first 39
  studies only: `COMPREHENSIVE_ABLATION_ANALYSIS.md`, `ESTRATEGIAS_MEJORA_ACCURACY.md` and
  `RESUMEN_ESTRATEGIAS.md`. Marked them 🕰️ in the `docs/README.md` index.

## [5.1.0] - 2026-09-09

### Repository hygiene
- **Removed `node_modules/` (9,250 files) from version control.** It was tracked by mistake; it is
  a local build artifact of `@mermaid-js/mermaid-cli`. Restore with `npm install`.
- **Removed the HuggingFace model cache (`.cache/hf/`) from version control.** Model weights and
  blobs are re-downloaded on demand.
- **Removed `.obsidian/` from version control** — personal editor state, not project configuration.
- Rewrote `.gitignore`: grouped and commented by category, added `.venv/`, `.cache/`, `.obsidian/`,
  `*.safetensors`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`; dropped stale entries.
- Deleted the duplicate `poster_18x24_300dpi - copia.png`.

### Changed — project layout
- **New `paper/` directory** for publication artifacts, moved out of the repository root and
  `docs/`:
  - `paper/manuscripts/` — the five CIARP 2025/2026 `.docx` drafts.
  - `paper/figures/` — `fig1`–`fig6` SVG and `graphical_abstract.pdf` (renamed from
    `docs/grafical abstract.pdf`, typo fixed).
  - `paper/posters/` — poster PDF/PNG outputs, the `poster_conflict_aware_print.html` source and
    its `assets/`.
  - `paper/presentations/` — the study-45 deep-dive `.pptx`.
- **New `scripts/windows/`** holding every `.bat` / `.ps1` convenience wrapper previously loose in
  the root. Each `.bat` now starts with `pushd "%~dp0..\.."`, so it works from the repository root
  *and* by double-click.
- Moved `screening_summary.py` and `view_results.py` to `scripts/`; moved
  `test_bf16_stability.py` to a new `tests/` directory. `setup_check.py` stays at the root as the
  documented entry point.
- Moved `docs/ablation_45_architecture.{mmd,puml,png}` into `docs/diagrams/`.
- Updated `scripts/export_poster_18x24.ps1` and `scripts/export_poster_a0.ps1` to read the poster
  HTML from `paper/posters/` and write their output there instead of the repository root.

### Fixed — documentation drift
- **Corrected the valid ablation study range from `1-50, 52-57` to `1-50, 52-63`** in `README.md`,
  `quickstart.md`, `AGENTS.md`, `docs/README.md`, `docs/ARCHITECTURE_AND_WORKFLOWS.md`,
  `docs/COMANDOS_EVALUACION.md`, `src/ablation_studies/README.md` and
  `config/ablation/ABLATION_SUMMARY.md`. Studies 58-63 had code paths in `run_ablation.py` but were
  documented nowhere.
- Added studies **58-63** to the `Study → Config → Model` table and a descriptive section in
  `config/ablation/ABLATION_SUMMARY.md`.
- **Documented orphan configs**: `ablation_65_unfrozen_vit.yaml` and
  `ablation_phase2_49_wavlm_multiscale_head_FUTURE.yaml` exist as YAML but have no code path in
  `run_ablation.py`. A config file is not a runnable study.
- Updated all references to the `.bat` scripts that moved to `scripts/windows/`.

### Added
- **`CLAUDE.md`** — Claude Code entry point that defers to `AGENTS.md`.
- Rewrote **`AGENTS.md`** as a structured operating manual: repository layout, baseline-vs-ablation
  table, study ranges and orphan configs, architecture notes, run-directory layout, the `paper/`
  and `scripts/windows/` conventions, and house rules.
- Rewrote **`README.md`** with badges, a table of contents, an ablation-series overview table and a
  BibTeX citation block.
- Rewrote **`quickstart.md`** with a troubleshooting table covering the most common failures.
- Rewrote **`docs/README.md`** as a proper documentation index with a git-policy table, a diagram
  inventory and a pointer to `paper/`.
- Aligned `pyproject.toml` version with the changelog and filled in project metadata.

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

