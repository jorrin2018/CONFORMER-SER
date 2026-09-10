# CLAUDE.md

This project keeps a single, shared agent manual so that every assistant sees the same rules.

**→ Read [AGENTS.md](AGENTS.md) first.** It covers the repository layout, the baseline-vs-ablation
split, the valid ablation study range, architecture notes, run-directory conventions and the
house rules for making changes here.

Nothing in this file overrides `AGENTS.md`; the sections below only add Claude Code specifics.

---

## Quick orientation

Multimodal Speech Emotion Recognition on IEMOCAP (4 classes: angry, happy, sad, neutral).
PyTorch + HuggingFace. Windows / PowerShell is the primary environment.

```powershell
python setup_check.py                                                   # verify env + dataset path
python src\train.py --config config\config.yaml --device cuda           # baseline
python scripts\run_ablation.py --study 45 --device cuda                 # ablation
python scripts\run_test_only.py --checkpoint_dir <run_dir> --study 45 --device cuda
```

## Traps worth repeating

1. **`src/evaluate.py` is baseline-only.** For ablation checkpoints use `src/evaluate_ensemble.py`
   or `scripts/run_test_only.py`.
2. **Valid studies are `1–50` and `52–63`; study 51 does not exist.** Some YAMLs in
   `config/ablation/` (e.g. `ablation_65_*`) have no code path — a config file is not a runnable
   study.
3. **The ViT log-mel is computed on the fly inside the model** from the batch waveform. There is
   no precomputed mel on disk.
4. **`config.json` inside a run directory** is the config that was actually used. Prefer it over
   the YAML when reproducing results.

## Working agreements

- Shell is PowerShell: use `\` in paths and a backtick `` ` `` for line continuation.
- Docs under `docs/`, plus `README.md` and `quickstart.md`, are written in **Spanish**. Keep them
  in Spanish. `AGENTS.md` and this file are in English.
- Verify study numbers, model mappings and CLI flags against `scripts/run_ablation.py` before
  writing them into documentation — the docs have drifted from the code before.
- Do not start training runs on your own initiative; they take hours of GPU time.
- Do not commit checkpoints, datasets, HuggingFace caches or `node_modules/`.
