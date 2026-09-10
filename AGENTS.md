# AGENTS.md

Operating manual for AI coding assistants working in this repository. Human-facing docs:
[README.md](README.md) · [quickstart.md](quickstart.md) · [docs/README.md](docs/README.md).

---

## 1. Project at a glance

| | |
|---|---|
| **Domain** | Multimodal Speech Emotion Recognition (SER) on IEMOCAP |
| **Classes** | 4 — `angry`, `happy`, `sad`, `neutral` |
| **Stack** | PyTorch + HuggingFace Transformers |
| **Primary environment** | Windows / PowerShell (paths in docs use `\`) |
| **Python** | 3.8+ ; CUDA GPU strongly recommended for training |
| **Split** | Speaker-independent: sessions 1–3 train, 4 val, 5 test |

The research contribution is **CAMF** — a Conflict-Aware Multimodal Fusion framework combining
audio, text and vision streams. Everything else in the repo exists to train, ablate and evaluate it.

---

## 2. Repository layout

```text
CONFORMER-SER/
├── config/
│   ├── config.yaml                 # baseline config (set dataset.data_path here)
│   ├── config_audio_only.yaml
│   ├── experiment_lightweight.yaml
│   └── ablation/                   # 68 ablation YAMLs + ABLATION_SUMMARY.md (canonical catalog)
├── src/
│   ├── train.py                    # baseline training entry point
│   ├── evaluate.py                 # baseline evaluation ONLY
│   ├── evaluate_ensemble.py        # ablation + ensemble evaluation
│   ├── inference.py                # single-file inference (baseline model)
│   ├── data/                       # dataset, preprocessing, augmentation, prosody cues
│   ├── models/                     # baseline model, fusion mechanisms, encoders
│   ├── ablation_studies/           # per-study model classes + classification heads
│   ├── experiments/                # lightweight-encoder experiment track
│   └── utils/                      # losses, metrics, logging
├── scripts/
│   ├── run_ablation.py             # ablation training entry point (source of truth for study→model)
│   ├── run_test_only.py            # test-set evaluation of an existing run
│   ├── aggregate_ablation_results.py
│   ├── export_poster_*.ps1
│   └── windows/                    # .bat / .ps1 convenience wrappers (see §7)
├── docs/                           # versioned documentation (index: docs/README.md)
│   ├── experimentos/               # per-series experiment logs
│   └── diagrams/                   # Draw.io / PlantUML / Mermaid sources
├── paper/                          # publication artifacts (see §8)
├── tests/
├── notebooks/
├── setup_check.py                  # environment + dataset verification
└── requirements.txt
```

**Generated, gitignored:** `checkpoints/`, `logs/`, `evaluation_results/`, `data/`,
`node_modules/`, `.cache/`, `venv/`, `.venv/`, `openspec/`.

---

## 3. The #1 mistake: baseline vs ablation are separate paths

There are two independent training/evaluation pipelines. Using the wrong script for a checkpoint
is the most common failure in this repo.

| Goal | Command | Model source |
|------|---------|--------------|
| Train baseline | `python src\train.py --config config\config.yaml --experiment_name <name> --device cuda` | `ConformerMultimodalModel` / `AudioOnlyModel` |
| Train ablation | `python scripts\run_ablation.py --study N --device cuda` | `create_ablation_model()` in `scripts/run_ablation.py` |
| Evaluate **baseline** checkpoint | `python src\evaluate.py --checkpoint <best_model.pt> --config config\config.yaml --split test --device cuda` | Rebuilds baseline model |
| Evaluate **ablation** checkpoint | `python src\evaluate_ensemble.py --checkpoints <ckpt> --configs <yaml> --study_numbers N --weights 1.0 --split test --device cuda` | Loads the ablation model class |
| Test-only on a saved run | `python scripts\run_test_only.py --checkpoint_dir <run_dir> --study N --device cuda` | Reads `config.json` from the run dir |

> **Critical:** `src/evaluate.py` does **not** know about the classes in `src/ablation_studies/`.
> Never point it at an ablation checkpoint — use `evaluate_ensemble.py` or `run_test_only.py`.

---

## 4. Ablation studies

- **Valid study numbers: `1–50` and `52–63`.** There is **no study 51**.
  The range is enforced by `create_ablation_model()` in `scripts/run_ablation.py`.
- Many studies **reuse the same model class** with a different YAML. `create_ablation_model()` is
  the single source of truth for the study number → model class mapping — read it before assuming
  a study has its own class.
- Configs live in `config/ablation/ablation_*.yaml`. Human-readable catalog:
  [`config/ablation/ABLATION_SUMMARY.md`](config/ablation/ABLATION_SUMMARY.md).
- **Orphan configs exist.** `ablation_65_unfrozen_vit.yaml` and
  `ablation_phase2_49_wavlm_multiscale_head_FUTURE.yaml` have **no code path** in
  `run_ablation.py`. A YAML file existing does *not* mean the study is runnable.
- Studies **20, 29, 41** run K-Fold / LOSO cross-validation instead of the standard split.

### Recent series (58–63)

| Study | Focus | Model class |
|-------|-------|-------------|
| 58 | MultiScaleTemporal head + SupCon (3 views) + Wav2Vec2 | `Ablation48Model` |
| 59 | MultiScaleTemporal + WavLM + SpecAugment on ViT (no waveform aug) | `Ablation34Model` |
| 60 | SOTA combo — WavLM + MultiScaleTemporal + SupCon + SpecAugment | `Ablation48Model` |
| 61 | SOTA combo + CompositeLoss with neutral-class focus | `Ablation48Model` |
| 62 | SOTA combo + attention-based audio pooling | `Ablation34Model` |
| 63 | Fine-tuned study 45 (lower dropout, lower label smoothing) | `Ablation34Model` |

---

## 5. Architecture notes

- **Audio encoders:** Wav2Vec2, WavLM (base / base-plus, optional weighted hidden states),
  HuBERT, Emotion2Vec.
- **Text encoders:** BERT / RoBERTa / DistilBERT, plus **SpeechCue** prosodic prompt injection
  in specific studies (`src/data/prosody_cues.py`).
- **Vision branch:** ViT over a **log-mel spectrogram computed inside the forward pass** from the
  *same* batch waveform (torchaudio `MelSpectrogram` → image). **IEMOCAP in this pipeline ships raw
  audio only — there is no precomputed mel feature stored on disk.** This trips people up
  repeatedly; do not add a "load mel features" path.
- **Fusion mechanisms** (`src/models/fusion.py`): Enhanced Cross-Attention, Conflict-Aware
  Cross-Attention, Gated Interactive Attention, Tensor Fusion, Bilinear, GLU, Late, Hierarchical,
  Adaptive Weighted.
- **Classification heads** (`src/ablation_studies/classification_heads.py`): MLP, LSTM,
  Transformer, CNN1D, MultiScaleTemporal.
- **Losses** (`src/utils/losses.py`): weighted Cross Entropy, Focal, Dirichlet Evidence Learning,
  SupCon, and composite combinations. `log_loss_contract_traceability()` prints the resolved loss
  contract at startup — note that `gamma` does **not** apply to plain `cross_entropy`.
- **Optimizer groups** (`src/ablation_studies/utils.py`): separate learning rates for encoders,
  fusion and classifier.

---

## 6. Run directory layout

```text
checkpoints/ablation_XX/ablation_XX_YYYYMMDD_HHMMSS/
├── best_model.pt              # gitignored
├── config.json                # merged config actually used — read this, not the YAML
├── best_metrics.json
├── final_results.json         # written by run_test_only.py
├── confusion_matrix_test.png
├── training_curves.png
└── test_eval.log
```

When reproducing or diagnosing a run, trust `config.json` inside the run directory over the YAML
in `config/ablation/` — the YAML may have changed since the run.

Aggregate results across runs:

```powershell
python scripts\aggregate_ablation_results.py --write-docs
```

This regenerates [`docs/RANKING_Y_RESULTADOS.md`](docs/RANKING_Y_RESULTADOS.md) from all
`final_results.json` files found under `checkpoints/`.

---

## 7. Windows wrappers

All `.bat` / `.ps1` convenience scripts live in `scripts/windows/`. Each `.bat` starts with
`pushd "%~dp0..\.."`, so it works both from the repo root and by double-click.

```powershell
.\scripts\windows\setup.bat              # pip install + setup_check.py
.\scripts\windows\train.bat              # baseline training (forwards all args)
.\scripts\windows\start_tensorboard.bat  # TensorBoard on logs/
.\scripts\windows\EVALUAR_RAPIDO.bat     # interactive evaluation menu
```

Their hardcoded checkpoint paths point at historical runs and are frequently stale — treat them as
templates, not as working commands.

---

## 8. `paper/` — publication artifacts

```text
paper/
├── manuscripts/    # CIARP 2025/2026 .docx drafts (v9, v10, v11)
├── figures/        # fig1–fig6 SVG + graphical_abstract.pdf
├── posters/        # poster PDFs/PNGs + poster_conflict_aware_print.html + assets/
└── presentations/  # .pptx deep-dive deck
```

Posters are **generated**: edit `paper/posters/poster_conflict_aware_print.html`, then run
`.\scripts\export_poster_18x24.ps1 -Format pdf` (or `-Format png -Dpi 300`). The scripts write
back into `paper/posters/`. Do not hand-edit the generated PDFs/PNGs.

---

## 9. Conventions for agents

- **Language:** code, identifiers and commit messages in English; user-facing docs (`README.md`,
  `quickstart.md`, `docs/`) are in **Spanish** — keep them that way. `AGENTS.md` and `CLAUDE.md`
  are in English.
- **Paths in docs:** Windows style (`config\config.yaml`), PowerShell line continuation is a
  backtick `` ` ``.
- **Before claiming a study range, model mapping or CLI flag**, verify it against
  `scripts/run_ablation.py` — the docs have drifted from the code before.
- **Never commit** checkpoints, datasets, HuggingFace caches or `node_modules/`. They are
  gitignored; if `git status` shows them, something is wrong with `.gitignore`.
- **Do not launch training runs** unless explicitly asked — they take hours on a single GPU.
- **`openspec/` is gitignored**: change proposals stay local.

---

## 10. Key files to read first

| File | Why |
|------|-----|
| `scripts/run_ablation.py` | Source of truth for study → model → config mapping |
| `config/ablation/ABLATION_SUMMARY.md` | Human-readable catalog of every study |
| `docs/ARCHITECTURE_AND_WORKFLOWS.md` | Detailed code map and data flow |
| `src/models/fusion.py` | All fusion mechanisms |
| `src/ablation_studies/classification_heads.py` | All classification heads |
| `src/utils/losses.py` | Loss configuration contract |
| `src/models/conformer.py` | Baseline model |
