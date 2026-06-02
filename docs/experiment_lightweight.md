# Lightweight Multimodal Experiment

## Goal
Reduce total parameter count by ~10× while preserving the Ablation 45
fusion and classification head. All three encoders are replaced with
small from-scratch networks.

## Architecture

| Component | Old (Ablation 45) | New (Lightweight) | Params (approx) |
|-----------|-------------------|-------------------|-----------------|
| Audio encoder | Wav2Vec2-base (~95 M) | SimpleConformerEncoder | ~7 M |
| Text encoder | DistilBERT (~66 M) | SimpleTransformerEncoder | ~6 M |
| Vision encoder | ViT-base (~86 M) | SimpleVisionCNNEncoder | ~2 M |
| Fusion + Head | Conflict-Aware + MultiScaleTemporal | **identical** | ~8 M |
| **Total** | **~250 M** | **~23 M** | **~10× reduction** |

### Audio — SimpleConformerEncoder
- **Input:** raw waveform `(B, T)`
- **Frontend:** 2× `Conv1d` (stride 2) → time downsampled 4×
- **Body:** 4× Conformer blocks (`d_model=256`, heads=4, conv_k=15, FFN=1024)
- **Output:** mean-pooled → `(B, 256)`

### Text — SimpleTransformerEncoder
- **Tokenizer:** DistilBERT tokenizer (vocab 30 522) — reused for SpeechCue compatibility
- **Embeddings:** learned token + positional embeddings (`d_model=256`)
- **Body:** 4× `TransformerEncoderLayer` (pre-norm, GELU, `d_ff=1024`)
- **Output:** CLS token → `(B, 256)`

### Vision — SimpleVisionCNNEncoder
- **Input:** raw waveform (log-mel computed on-the-fly inside `forward`)
- **Body:** small 2D CNN (32 → 64 → 128 → 256 channels, two `MaxPool2d`, `AdaptiveAvgPool2d`)
- **Output:** `(B, 256)`

### Fusion & Head
- `ConflictAwareCrossAttentionFusion(hidden_dim=512, heads=8, layers=3)`
- `MultiScaleTemporalClassificationHead(...)` — same config as Ablation 45

## SpeechCue
`use_prosody_cues: true` is kept in the config. The data pipeline
(`IEMOCAPDataset` + `TextPreprocessor`) injects prosody cues into the
transcription text **before** tokenization. Our `SimpleTransformerEncoder`
uses the same tokenizer and token IDs, so SpeechCue is transparent.

## Training
- **Batch size:** 8
- **Gradient accumulation:** 8 → effective batch 64
- **Learning rates:**
  - Audio / Text / Vision encoders: `1.0e-4`
  - Fusion + classification head: `5.0e-5`
- **Scheduler:** `ReduceLROnPlateau` (same as Ablation 45)
- **Loss:** CrossEntropy with label smoothing `0.1`

## Run

```powershell
python src\experiments\train_lightweight.py `
  --config config\experiment_lightweight.yaml `
  --device cuda `
  --experiment_name lightweight_speechcue_v1
```

## Files

| File | Description |
|------|-------------|
| `src/experiments/lightweight_encoders.py` | Model definition |
| `src/experiments/train_lightweight.py` | Training script |
| `config/experiment_lightweight.yaml` | Configuration |
| `docs/experiment_lightweight.md` | This document |
