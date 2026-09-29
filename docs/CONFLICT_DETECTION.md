# Conflict Detection Methods (Conflict-Aware Cross-Attention)

Conflict-Aware Cross-Attention fusion detects when modalities contradict each other (e.g. text says "happy" but audio sounds angry) and applies confidence-based weighting instead of simple averaging. You can choose how conflicts are detected via configuration.

## Overview

The fusion module supports three **conflict detection methods**. Set the method in your ablation or training config under `model.conflict_aware.conflict_detection_method`.

| Method | Description | Inputs | Typical use |
|--------|-------------|--------|-------------|
| **prediction_divergence** | Conflict when two or more modalities predict different classes (argmax of per-modality logits). | Per-modality logits | Default; interpretable "modalities disagree on label". |
| **feature_distance** | Conflict when modality embeddings are far apart: mean pairwise cosine distance > `conflict_threshold`. | Embeddings (after projection) | When you care about representation alignment, not only final class. |
| **learned** | A small MLP predicts conflict probability from concatenated embeddings; conflict when probability > `conflict_threshold`. | Embeddings | When you want the model to learn what counts as conflict. |

## How to change the method

1. Open the YAML config for your experiment (e.g. `config/ablation/ablation_40_wavlm_conflict_aware.yaml`).
2. Locate the `model.conflict_aware` section.
3. Set `conflict_detection_method` to one of: `"prediction_divergence"`, `"feature_distance"`, or `"learned"`.

### Example: use prediction_divergence (default)

```yaml
model:
  conflict_aware:
    enabled: true
    conflict_detection_method: "prediction_divergence"
    conflict_threshold: 0.3
    confidence_weighting: true
    conflict_resolution: "confidence_weighted"
```

### Example: use feature_distance

```yaml
model:
  conflict_aware:
    enabled: true
    conflict_detection_method: "feature_distance"
    conflict_threshold: 0.3   # mean cosine distance above this => conflict
    confidence_weighting: true
    conflict_resolution: "confidence_weighted"
```

### Example: use learned

```yaml
model:
  conflict_aware:
    enabled: true
    conflict_detection_method: "learned"
    conflict_threshold: 0.3   # MLP output probability above this => conflict
    confidence_weighting: true
    conflict_resolution: "confidence_weighted"
```

The rest of the pipeline (confidence weighting, conflict resolution) is unchanged; only the way conflicts are detected changes.

## Config files that support conflict detection

Any config that uses `fusion_type: "conflict_aware_cross_attention"` and has a `conflict_aware` block supports `conflict_detection_method`. For example:

- `config/ablation/ablation_34_conflict_aware_attention.yaml`
- `config/ablation/ablation_40_wavlm_conflict_aware.yaml`
- `config/ablation/ablation_36a_dirichlet_conflict.yaml`, `ablation_36b_focal_conflict.yaml`, `ablation_36c_hybrid_conflict.yaml`
- `config/ablation/ablation_37_dirichlet_conflict_aware.yaml`
- `config/ablation/ablation_38_opensmile_residual.yaml`
- `config/ablation/ablation_39_sota_improvements.yaml`

## Checkpoint compatibility

- **prediction_divergence** and **feature_distance**: No extra parameters. Checkpoints trained with these methods can be loaded regardless of which method you use at inference (only the detection logic changes).
- **learned**: Adds a `conflict_detector_net` submodule. Checkpoints saved with `learned` contain these weights; checkpoints saved with `prediction_divergence` or `feature_distance` do not. Loading an old checkpoint (e.g. from Ablation 34 or 40 with default config) while using `conflict_detection_method: "learned"` will fail unless you load with `strict=False` (and then the conflict detector will be randomly initialized). Best practice: use `prediction_divergence` when loading checkpoints that were trained with the default conflict method.

## Running experiments to compare methods

To compare the three methods on the same architecture (e.g. Ablation 40):

```powershell
# Default: prediction_divergence
python scripts/run_ablation.py --study 40

# To try feature_distance or learned, edit the config first:
# In config/ablation/ablation_40_wavlm_conflict_aware.yaml set:
#   conflict_detection_method: "feature_distance"
# or
#   conflict_detection_method: "learned"
# Then run again (use a different checkpoint_dir or run name if you want to keep both).
```

You can also duplicate the config file (e.g. `ablation_40_feature_distance.yaml`) and register it as a separate study in `scripts/run_ablation.py` to switch methods by study number.
