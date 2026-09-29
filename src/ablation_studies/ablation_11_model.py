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

import warnings
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import ViTModel, ViTConfig

from src.models.conformer import (ConformerMultimodalModel, apply_finetuning_policy,
                                  build_layer_weights, weighted_hidden_states)
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
        # SpeechCue: nueve tokens nuevos
        # -----------------------------
        # create_dataloaders() los anade al tokenizer y deja aqui el tamano
        # resultante del vocabulario. El encoder de texto tiene que crecer las
        # mismas nueve filas, y esas filas deben quedar entrenables aunque el
        # resto del encoder este congelado: son "los nueve embeddings" del
        # articulo.
        tokenizer_len = config.get("text", {}).get("_tokenizer_len")
        if tokenizer_len:
            emb = self.text_encoder.get_input_embeddings()
            current = emb.weight.shape[0]
            if tokenizer_len != current:
                self.text_encoder.resize_token_embeddings(tokenizer_len)
                emb = self.text_encoder.get_input_embeddings()
                n_new = tokenizer_len - current
                emb.weight.requires_grad = True
                # "adds only nine learnable embeddings": the gradient of every
                # pre-existing row is zeroed, so only the new rows actually learn.
                # Without this the whole embedding matrix would train, which is a
                # different (and much larger) change than the paper claims.
                mask = torch.zeros_like(emb.weight)
                mask[current:] = 1.0
                emb.weight.register_hook(lambda grad, m=mask: grad * m)
                # The whole matrix has requires_grad=True so autograd reaches the new
                # rows, but the hook zeroes the gradient of every pre-existing row.
                # Record the split so parameter counts can report what actually learns
                # instead of the nominal size of the matrix.
                dim = emb.weight.shape[1]
                self.speechcue_new_embeddings = n_new * dim
                self.speechcue_frozen_embeddings = current * dim
                print(f"[SpeechCue] text embeddings {current} -> {tokenizer_len}; "
                      f"only the {n_new} new rows learn "
                      f"({self.speechcue_new_embeddings:,} params, not "
                      f"{self.speechcue_frozen_embeddings + self.speechcue_new_embeddings:,})")

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
        # ViT: "frozen" means frozen. vision_unfreeze_last_n_layers only applies when
        # freeze_vision_encoder is False, so a config cannot claim a frozen ViT while
        # quietly fine-tuning its top blocks.
        freeze_vision = model_config.get("freeze_vision_encoder", True)
        apply_finetuning_policy(
            self.vision_encoder,
            freeze_all=freeze_vision,
            unfreeze_last_n=(0 if freeze_vision
                             else model_config.get("vision_unfreeze_last_n_layers", 0)),
            name=f"vision/{vit_name}",
            freeze_feature_extractor=False,
        )

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

        # Optional SpecAugment on log-mel for ViT only (training); does not affect audio encoder input
        self._vision_mel_spec_augment = bool(
            model_config.get("vision_mel_spec_augment", False)
        )
        _sa = model_config.get("vision_spec_augment", {}) or {}
        _fparam = int(_sa.get("freq_mask_param", 10))
        _tparam = int(_sa.get("time_mask_param", 40))
        if self._vision_mel_spec_augment:
            self._vit_freq_mask = torchaudio.transforms.FrequencyMasking(_fparam)
            self._vit_time_mask = torchaudio.transforms.TimeMasking(_tparam)
        else:
            self._vit_freq_mask = None
            self._vit_time_mask = None

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

        # Suma ponderada de capas ocultas (Pepino et al.). Estaba restringida a WavLM
        # sin motivo: la informacion paralinguistica vive en las capas intermedias de
        # cualquier encoder de voz, wav2vec2 incluido.
        # El parametro se llama audio_encoder_layer_logits a proposito: create_optimizer()
        # agrupa por subcadena del nombre, asi que ese nombre lo deja en el grupo de
        # audio (3e-5). Como wavlm_layer_logits caia en el grupo de fusion y entrenaba
        # a 5e-6, seis veces mas lento, los pesos de capa apenas se movian.
        self._weighted_hidden_states = bool(
            model_config.get("audio_weighted_hidden_states",
                             model_config.get("wavlm_weighted_hidden_states", False))
        )
        self.audio_encoder_layer_logits: Optional[nn.Parameter] = None
        if self._weighted_hidden_states:
            self.audio_encoder_layer_logits = build_layer_weights(self.audio_encoder)

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

            # Log-mel; optional SpecAugment (ViT branch only, training mode)
            mel = torch.log(mel + 1e-9)
            if (
                self.training
                and self._vision_mel_spec_augment
                and self._vit_freq_mask is not None
                and self._vit_time_mask is not None
            ):
                # torchaudio expects (..., freq, time); add batch dim
                m = mel.unsqueeze(0)
                if m.size(-1) > 1:
                    m = self._vit_freq_mask(m)
                    m = self._vit_time_mask(m)
                mel = m.squeeze(0)

            # Normalización por muestra
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
        # Audio (WavLM optional: weighted hidden states)
        # -----------------------------
        if self._weighted_hidden_states and self.audio_encoder_layer_logits is not None:
            audio_features = weighted_hidden_states(
                self.audio_encoder, audio, self.audio_encoder_layer_logits
            )
            audio_pooled = self.pool_audio_features(audio_features)
        else:
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

        salida = {
            "logits": logits,
            "predictions": predictions,
            "loss": loss,
            "audio_features": audio_pooled,
            "text_features": text_features,
            "vision_features": vision_features,
        }
        # Logits de los clasificadores por rama que alimentan la deteccion de
        # conflicto. Se sacan para que train_epoch pueda ponerles una perdida
        # auxiliar: sin ella nunca reciben gradiente (los consume un argmax).
        rama = getattr(self.fusion, "branch_logits", None)
        if rama is not None:
            salida["branch_logits"] = rama
        conf = getattr(self.fusion, "branch_confidence", None)
        if conf is not None:
            salida["branch_confidence"] = conf
        return salida



