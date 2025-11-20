"""
Ablation Study 11: Audio + Texto + Rama Visual (Mel-espectrogramas como imagen + ViT)
con fusión tipo Cross-Attention trimodal y cabeza LSTM.

Idea:
- Mantener el backbone multimodal (Wav2Vec2 + DistilBERT) como en otras ablations.
- Extraer un mel-espectrograma del waveform y tratarlo como imagen.
- Pasar esa “imagen” por un Vision Transformer (ViT).
- Fusionar las tres modalidades (audio, texto, visión) con un bloque
  de atención cruzada trimodal (`TriModalCrossAttentionFusion`),
  y usar una cabeza LSTM para clasificación (como en Ablation 07/09).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import ViTModel, ViTConfig

from src.models.conformer import ConformerMultimodalModel
from src.models.fusion import TriModalCrossAttentionFusion
from .classification_heads import LSTMClassificationHead


class Ablation11Model(ConformerMultimodalModel):
    """
    Modelo trimodal:
    - Audio (Wav2Vec2 / Conformer, a partir de waveform)
    - Texto (DistilBERT)
    - Visual: Mel-espectrograma -> imagen -> Vision Transformer (ViT)
    fusionados con `TriModalCrossAttentionFusion` y cabeza LSTM.
    """

    def __init__(self, config: Dict):
        # Extraer subconfiguración de modelo y audio
        model_config = config.get("model", {})
        audio_config = config.get("audio", {})
        dataset_config = config.get("dataset", {})

        # Inicializar backbone audio+texto como en otros ablations
        super().__init__(
            audio_encoder_name=model_config.get(
                "audio_encoder", "facebook/wav2vec2-base"
            ),
            text_encoder_name=model_config.get(
                "text_encoder", "distilbert-base-uncased"
            ),
            num_classes=model_config.get("num_classes", 4),
            hidden_dim=model_config.get("hidden_dim", 256),
            dropout=model_config.get("dropout", 0.3),
            fusion_type=model_config.get("fusion_type", "cross_attention"),
            fusion_num_heads=model_config.get("fusion_num_heads", 4),
            freeze_audio_encoder=model_config.get("freeze_audio_encoder", False),
            audio_unfreeze_last_n_layers=model_config.get(
                "audio_unfreeze_last_n_layers", 4
            ),
            freeze_text_encoder=model_config.get("freeze_text_encoder", False),
            text_unfreeze_last_n_layers=model_config.get(
                "text_unfreeze_last_n_layers", 2
            ),
            audio_pooling=model_config.get("audio_pooling", "mean"),
        )

        # -----------------------------
        # Rama Visual: ViT sobre Mel-espectrograma
        # -----------------------------
        vit_name = model_config.get("vision_encoder", "google/vit-base-patch16-224")
        try:
            self.vision_encoder = ViTModel.from_pretrained(vit_name)
        except Exception:
            # Fallback a configuración por defecto si no se puede cargar el checkpoint
            vit_config = ViTConfig()
            self.vision_encoder = ViTModel(vit_config)

        vision_hidden_size = self.vision_encoder.config.hidden_size

        # Opcional: congelar la rama visual (por defecto True para reducir coste)
        freeze_vision = model_config.get("freeze_vision_encoder", True)
        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        # Parámetros para construir Mel-espectrograma como imagen
        sample_rate = dataset_config.get("sample_rate", 16000)
        n_mels = int(model_config.get("mel_n_mels", audio_config.get("n_mels", 80)))
        n_fft = int(model_config.get("mel_n_fft", audio_config.get("n_fft", 400)))
        hop_length = int(
            model_config.get("mel_hop_length", audio_config.get("hop_length", 160))
        )
        win_length = int(
            model_config.get("mel_win_length", audio_config.get("win_length", 400))
        )

        self.vit_image_size = int(model_config.get("vision_image_size", 224))

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            normalized=True,
        )

        # -----------------------------
        # Fusión trimodal + cabeza LSTM
        # -----------------------------
        audio_hidden_size = self.audio_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size

        # Sustituimos el módulo de fusión estándar por el trimodal
        self.fusion = TriModalCrossAttentionFusion(
            audio_dim=audio_hidden_size,
            text_dim=text_hidden_size,
            vision_dim=vision_hidden_size,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_heads=model_config.get("fusion_num_heads", 4),
            dropout=model_config.get("dropout", 0.3),
        )

        # Reemplazamos la cabeza clasificadora por una LSTM (igual filosofía que Ablation 07/09)
        fusion_output_dim = self.hidden_dim // 2
        lstm_num_layers = model_config.get("lstm_num_layers", 2)
        lstm_bidirectional = model_config.get("lstm_bidirectional", True)

        self.fusion.classifier = LSTMClassificationHead(
            hidden_dim=fusion_output_dim,
            num_classes=self.num_classes,
            dropout=model_config.get("dropout", 0.3),
            num_layers=lstm_num_layers,
            bidirectional=lstm_bidirectional,
        )

    def _waveform_to_vit_image(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convierte un batch de waveforms (B, T) en imágenes (B, 3, H, W) para ViT
        a través de Mel-espectrogramas normalizados.
        """
        batch_size = audio.size(0)
        images = []

        for i in range(batch_size):
            # audio[i]: (T,)
            waveform = audio[i]
            mel = self.mel_transform(waveform)  # (n_mels, time)

            # Escala log y normalización por muestra
            mel = torch.log(mel + 1e-9)
            mel = (mel - mel.mean()) / (mel.std() + 1e-9)

            # A (1, 1, H', W') para hacer resize
            mel_img = mel.unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)
            mel_img = F.interpolate(
                mel_img,
                size=(self.vit_image_size, self.vit_image_size),
                mode="bilinear",
                align_corners=False,
            )  # (1, 1, H, W)

            # Repetimos a 3 canales para ViT estándar
            mel_img = mel_img.squeeze(0).repeat(3, 1, 1)  # (3, H, W)
            images.append(mel_img)

        images = torch.stack(images, dim=0)  # (B, 3, H, W)
        return images

    def forward(
        self,
        audio: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward trimodal:
        - audio: waveform (B, T)
        - texto: ids + mask
        - visión: Mel-espectrograma -> imagen -> ViT
        """
        # -----------------------------
        # Audio (igual que en ConformerMultimodalModel)
        # -----------------------------
        audio_outputs = self.audio_encoder(audio)
        audio_features = audio_outputs.last_hidden_state  # (B, S, H_a)
        audio_pooled = self.pool_audio_features(audio_features)  # (B, H_a)

        # -----------------------------
        # Texto
        # -----------------------------
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]  # CLS (B, H_t)

        # -----------------------------
        # Visión: Mel-espectrograma -> imagen -> ViT
        # -----------------------------
        vit_images = self._waveform_to_vit_image(audio)  # (B, 3, H, W)
        vit_images = vit_images.to(audio.device)

        vision_outputs = self.vision_encoder(pixel_values=vit_images)
        if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
            vision_features = vision_outputs.pooler_output  # (B, H_v)
        else:
            vision_features = vision_outputs.last_hidden_state[:, 0, :]  # CLS

        # -----------------------------
        # Fusión trimodal
        # -----------------------------
        logits = self.fusion(audio_pooled, text_features, vision_features)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        predictions = torch.argmax(logits, dim=-1)

        return {
            "logits": logits,
            "predictions": predictions,
            "loss": loss,
            "audio_features": audio_pooled,
            "text_features": text_features,
            "vision_features": vision_features,
        }



