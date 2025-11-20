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
        # Idea análoga a Ablation 07 para audio/texto:
        # - Mantener congeladas las capas bajas del ViT (features más genéricas).
        # - Descongelar las capas medias y altas (últimas N capas) para adaptar mejor
        #   la representación visual al dominio de SER.
        #
        # Se controla con:
        #   model.freeze_vision_encoder: bool (por defecto True)
        #   model.vision_unfreeze_last_n_layers: int (por defecto 0 => todo congelado)
        #
        # Nota: `Ablation11Model` ya congela toda la rama visual cuando
        # `freeze_vision_encoder` es True. Aquí revertimos parcialmente esa decisión.
        freeze_vision = model_config.get("freeze_vision_encoder", True)
        vision_unfreeze_last_n = int(model_config.get("vision_unfreeze_last_n_layers", 0) or 0)

        if freeze_vision and vision_unfreeze_last_n > 0:
            try:
                encoder = getattr(self.vision_encoder, "encoder", None)
                if encoder is not None and hasattr(encoder, "layer"):
                    encoder_layers = encoder.layer
                    # Descongelar las últimas N capas (capas medias/altas del ViT)
                    for layer in encoder_layers[-vision_unfreeze_last_n:]:
                        for p in layer.parameters():
                            p.requires_grad = True

                # También descongelamos las LayerNorm finales si existen, para
                # permitir una ligera recalibración de las activaciones.
                if hasattr(self.vision_encoder, "layernorm"):
                    for p in self.vision_encoder.layernorm.parameters():
                        p.requires_grad = True
                if encoder is not None and hasattr(encoder, "layernorm"):
                    for p in encoder.layernorm.parameters():
                        p.requires_grad = True
            except Exception as e:
                # En caso de que la estructura interna del ViT difiera,
                # mostramos un warning pero continuamos con entrenamiento.
                print(
                    f"Warning: could not partially unfreeze vision encoder layers "
                    f"(vision_unfreeze_last_n_layers={vision_unfreeze_last_n}): {e}"
                )




