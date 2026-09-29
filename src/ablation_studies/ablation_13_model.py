"""
Ablation Study 13: Audio + Texto + Rama Visual (Mel-espectrogramas como imagen + ViT)
con fusión trimodal (TriModalCrossAttentionFusion) y cabeza LSTM, igual que Ablation 11,
pero con:

- Prompts internos de prosodia tipo SpeechCue (vía `text.use_prosody_cues: true`)
- Descongelado parcial de las capas medias y altas del Vision Transformer (ViT)

La inyección de prosodia se realiza en el pipeline de datos (igual que en Ablation 10),
por lo que este modelo reutiliza exactamente el forward de `Ablation11Model` y sólo
añade la lógica de descongelado parcial del ViT según la configuración.
"""

from typing import Dict

from .ablation_11_model import Ablation11Model


class Ablation13Model(Ablation11Model):
    """
    Modelo trimodal:
    - Audio (Wav2Vec2 / Conformer, a partir de waveform)
    - Texto (DistilBERT) con prompts internos de prosodia (SpeechCue-style)
    - Visual: Mel-espectrograma -> imagen -> Vision Transformer (ViT)

    Cambios respecto a Ablation 11:
    - Se asume `text.use_prosody_cues: true` en la config (SpeechCue interno).
    - Se descongelan las capas medias/altas del ViT (últimas N capas) en lugar de
      mantener toda la rama visual congelada.
    """

    def __init__(self, config: Dict):
        # Inicializar exactamente como Ablation 11 (backbone + ViT + fusión + cabeza LSTM)
        super().__init__(config)

        model_config = config.get("model", {})

        # ---------------------------------------------------------
        # Descongelado parcial del Vision Transformer (ViT)
        # ---------------------------------------------------------
        #
        # La politica de congelado del ViT vive en UN solo sitio:
        # Ablation11Model -> apply_finetuning_policy(). Aqui habia un segundo bloque
        # que descongelaba las ultimas N capas cuando freeze_vision_encoder era True,
        # deshaciendo la decision anterior: un config podia declarar "ViT congelado"
        # y acabar afinando sus bloques superiores. Eliminado.
        #
        #   freeze_vision_encoder: true                      -> ViT completamente congelado
        #   freeze_vision_encoder: false + unfreeze_last_n: N -> ultimas N capas entrenables




