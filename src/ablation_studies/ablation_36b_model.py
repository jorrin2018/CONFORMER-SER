"""
Ablation Study 36B: Conflict-Aware + Focal Loss with Neutral Weighting
Base: Ablation 34 (Conflict-Aware Cross-Attention)
Modificación: Focal Loss con pesos aumentados para neutral
Objetivo: Mejorar clasificación de neutral usando Focal Loss con pesos específicos
"""

from typing import Dict

from .ablation_34_model import Ablation34Model


class Ablation36BModel(Ablation34Model):
    """
    Modelo trimodal con Conflict-Aware Cross-Attention + Focal Loss:
    - Audio (Wav2Vec2)
    - Texto (DistilBERT) con SpeechCue
    - Visual: Mel-espectrograma -> ViT (congelado)
    
    Fusión: Conflict-Aware Cross-Attention (detecta contradicciones y usa confianza)
    Clasificación: LSTM Classification Head estándar
    Loss: Focal Loss con alpha aumentado para neutral (2.0x) y gamma=2.5
    """

    def __init__(self, config: Dict):
        # Initialize as Ablation 34 (Conflict-Aware Fusion + LSTM Head)
        # No changes to architecture, only loss function will be different
        super().__init__(config)
