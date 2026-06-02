"""
Lightweight multimodal encoders and model for SER.

Replaces pretrained Wav2Vec2, DistilBERT, and ViT with small from-scratch
encoders (Conformer, Transformer, CNN) while keeping the Ablation 45
fusion + classification head unchanged.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fusion import ConflictAwareCrossAttentionFusion
from src.ablation_studies.classification_heads import MultiScaleTemporalClassificationHead


class ConvolutionModule(nn.Module):
    """Conformer-style convolution module."""

    def __init__(self, d_model: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = self.norm(x)
        x = x.transpose(1, 2)  # (B, d_model, T)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)  # (B, d_model, T)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, T, d_model)
        return x


class ConformerBlock(nn.Module):
    """Single Conformer block (pre-norm style)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        conv_kernel_size: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm_ff1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion_factor, d_model),
            nn.Dropout(dropout),
        )
        self.norm_mhsa = nn.LayerNorm(d_model)
        self.mhsa = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.conv_module = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.norm_ff2 = nn.LayerNorm(d_model)
        self.ff2 = nn.Sequential(
            nn.Linear(d_model, d_model * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion_factor, d_model),
            nn.Dropout(dropout),
        )
        self.norm_final = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + 0.5 * self.ff1(self.norm_ff1(x))
        attn_out, _ = self.mhsa(
            self.norm_mhsa(x), self.norm_mhsa(x), self.norm_mhsa(x)
        )
        x = x + attn_out
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ff2(self.norm_ff2(x))
        x = self.norm_final(x)
        return x


class SimpleConformerEncoder(nn.Module):
    """Lightweight Conformer encoder for raw waveform.

    Frontend: 2x Conv1d with stride 2 (downsample by 4x).
    Body: stack of Conformer blocks + mean pooling.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_blocks: int = 4,
        conv_kernel_size: int = 15,
        expansion_factor: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Aggressive downsampling: 4x4x2x2 = 64x total
        # For 8s @ 16kHz (128k samples) -> ~2k frames, fits in 8GB GPU memory
        self.frontend = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.proj = nn.Linear(256, d_model)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model, n_heads, conv_kernel_size, expansion_factor, dropout
                )
                for _ in range(n_blocks)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_waveform)
        x = x.unsqueeze(1)  # (B, 1, T)
        x = self.frontend(x)  # (B, 256, T/4)
        x = x.transpose(1, 2)  # (B, T/4, 256)
        x = self.proj(x)  # (B, T/4, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # (B, d_model)
        return x


class SimpleTransformerEncoder(nn.Module):
    """Lightweight Transformer encoder for text (from scratch).

    Uses learned token + positional embeddings, then a stack of
    PyTorch TransformerEncoderLayer blocks. Returns the CLS token.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # input_ids: (B, seq_len)
        # attention_mask: 1 for real tokens, 0 for padding
        B, seq_len = input_ids.shape
        positions = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        x = self.dropout(x)
        key_padding_mask = attention_mask == 0  # True = padding
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x[:, 0, :]  # CLS token, (B, d_model)


class SimpleVisionCNNEncoder(nn.Module):
    """Small CNN encoder operating on raw log-mel spectrograms.

    Computes log-mel on the fly from waveform, then runs a tiny 2D CNN
    with adaptive global average pooling.
    """

    def __init__(
        self,
        d_model: int = 256,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
    ):
        super().__init__()
        import torchaudio

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            normalized=True,
        )
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, d_model)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (B, T)
        mel = self.mel_transform(waveform)  # (B, n_mels, time_frames)
        mel = torch.log(mel + 1e-9)
        # Per-sample z-normalization
        mel_mean = mel.mean(dim=(1, 2), keepdim=True)
        mel_std = mel.std(dim=(1, 2), keepdim=True)
        mel = (mel - mel_mean) / (mel_std + 1e-9)
        mel = mel.unsqueeze(1)  # (B, 1, n_mels, time_frames)
        x = self.cnn(mel)  # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 256)
        x = self.proj(x)  # (B, d_model)
        return x


class LightweightMultimodalModel(nn.Module):
    """Trimodal lightweight multimodal model.

    Encoders:
      - Audio: SimpleConformerEncoder (from scratch, raw waveform)
      - Text:  SimpleTransformerEncoder (from scratch, DistilBERT tokenizer)
      - Vision: SimpleVisionCNNEncoder (from scratch, raw log-mel)

    Fusion + Head:
      - ConflictAwareCrossAttentionFusion (same as Ablation 45)
      - MultiScaleTemporalClassificationHead (same as Ablation 45)
    """

    def __init__(self, config: Dict):
        super().__init__()
        model_config = config.get("model", {})
        dataset_config = config.get("dataset", {})
        audio_config = config.get("audio", {})

        self.num_classes = model_config.get("num_classes", 4)
        self.hidden_dim = model_config.get("hidden_dim", 512)
        dropout = model_config.get("dropout", 0.3)
        encoder_dim = 256  # fixed lightweight dimension

        # Encoders
        self.audio_encoder = SimpleConformerEncoder(
            d_model=encoder_dim,
            n_heads=4,
            n_blocks=4,
            conv_kernel_size=15,
            expansion_factor=4,
            dropout=dropout,
        )
        self.text_encoder = SimpleTransformerEncoder(
            vocab_size=30522,
            d_model=encoder_dim,
            n_heads=4,
            n_layers=4,
            max_seq_len=128,
            dropout=dropout,
        )
        self.vision_encoder = SimpleVisionCNNEncoder(
            d_model=encoder_dim,
            sample_rate=dataset_config.get("sample_rate", 16000),
            n_mels=audio_config.get("n_mels", 80),
            n_fft=audio_config.get("n_fft", 400),
            hop_length=audio_config.get("hop_length", 160),
            win_length=audio_config.get("win_length", 400),
        )

        # Fusion (identical to Ablation 45)
        conflict_config = model_config.get("conflict_aware", {})
        self.fusion = ConflictAwareCrossAttentionFusion(
            audio_dim=encoder_dim,
            text_dim=encoder_dim,
            vision_dim=encoder_dim,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_heads=model_config.get("fusion_num_heads", 8),
            num_layers=model_config.get("fusion_num_layers", 3),
            dropout=dropout,
            conflict_threshold=conflict_config.get("conflict_threshold", 0.3),
            conflict_detection_method=conflict_config.get(
                "conflict_detection_method", "prediction_divergence"
            ),
        )

        # Classification head (identical to Ablation 45)
        fusion_output_dim = self.hidden_dim // 2
        head_config = model_config.get("classification_head", {})
        self.fusion.classifier = MultiScaleTemporalClassificationHead(
            hidden_dim=fusion_output_dim,
            num_classes=self.num_classes,
            dropout=dropout,
            use_temporal_pooling=head_config.get("use_temporal_pooling", True),
            pooling_types=head_config.get(
                "pooling_types", ["mean", "max", "attention"]
            ),
            use_lstm=head_config.get("use_lstm", True),
            lstm_num_layers=head_config.get("lstm_num_layers", 2),
            lstm_bidirectional=head_config.get("lstm_bidirectional", True),
            lstm_hidden=head_config.get("lstm_hidden", 512),
            lstm_dropout=head_config.get("lstm_dropout", 0.3),
            use_self_attention=head_config.get("use_self_attention", True),
            attention_heads=head_config.get("attention_heads", 4),
            final_dropout=head_config.get("final_dropout", 0.4),
        )

        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[LightweightMultimodalModel] Total params: {total:,} | Trainable: {trainable:,}")

    def forward(
        self,
        audio: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        audio_features = self.audio_encoder(audio)  # (B, 256)
        text_features = self.text_encoder(
            text_input_ids, text_attention_mask
        )  # (B, 256)
        vision_features = self.vision_encoder(audio)  # (B, 256)

        logits = self.fusion(audio_features, text_features, vision_features)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        predictions = logits.argmax(dim=-1)
        return {
            "logits": logits,
            "predictions": predictions,
            "loss": loss,
            "audio_features": audio_features,
            "text_features": text_features,
            "vision_features": vision_features,
        }
