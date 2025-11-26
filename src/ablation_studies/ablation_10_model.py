"""
Ablation Study 10: Internal SpeechCue-style prosody prompts.

This ablation keeps the same architecture as Ablation 07
(unfreeze middle/high encoder layers + LSTM head) but relies
on an enriched textual input that includes prosody descriptions
generated from the raw audio:

    [PROSODY] <prosody description>. [SPEECH] <transcript>

The actual injection of prosody cues is handled in the dataset
pipeline via `use_prosody_cues: true` in the config. This model
wrapper exists mainly for clarity and bookkeeping in ablation runs.
"""

from src.ablation_studies.ablation_07_model import Ablation07Model


class Ablation10Model(Ablation07Model):
    """
    Same encoder + fusion + LSTM head as Ablation 07, but trained with
    prosody-aware text prompts (internal SpeechCue).
    """

    def __init__(self, config: dict):
        super().__init__(config)







