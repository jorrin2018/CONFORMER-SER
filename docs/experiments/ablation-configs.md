---
title: Ablation config catalog
description: Points to per-YAML ablation descriptions; pairs with phase deep-dives.
---

# Ablation configuration catalog

Every ablation has a YAML file under `config/ablation/`. The authoritative **filename → goal** list (kept next to the configs) is:

**[`config/ablation/ABLATION_SUMMARY.md`](../../config/ablation/ABLATION_SUMMARY.md)**

Use it when you need the exact `ablation_XX_*.yaml` name.

## Narrative architecture docs

For diagrams, tensor shapes, and source-file links, read the phase pages from [overview.md](overview.md):

- [01–04 Classification heads](phases/01-04_classification_heads.md)
- [05–07 Unfreezing](phases/05-07_unfreezing.md)
- [08–09 Depth and fusion](phases/08-09_depth_fusion.md)
- [10–13 SpeechCue and visual](phases/10-13_speechcue_visual.md)
- [21–28 Fusion mechanisms](phases/21-28_fusion_mechanisms.md)
- [30–39 Advanced multimodal](phases/30-39_advanced_multimodal.md)
- [40–44 Phase trimodal](phases/40-44_phase_trimodal.md)

## Phase / WavLM studies (40–44)

Configs such as `ablation_40_original_trimodal.yaml` live alongside the summary document; see the phase doc for encoder and loss details.
