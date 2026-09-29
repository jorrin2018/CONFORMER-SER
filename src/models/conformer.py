"""Conformer-based multimodal model for speech emotion recognition.

This module provides the base ConformerMultimodalModel (audio + text fusion with
configurable fusion type and encoder freezing) and AudioOnlyModel baseline.
Supports Wav2Vec2 and WavLM audio encoders via AutoModel, and BERT/DistilBERT
for text. Trimodal fusion types (e.g. conflict_aware_cross_attention) are
handled by ablation study models that build on this base.
"""

import torch
import torch.nn as nn
import math
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Config,
    AutoModel,
    AutoConfig
)
from typing import Dict, Optional

from .emotion2vec_encoder import Emotion2VecEncoder, _is_emotion2vec_model
from .fusion import (
    FusionModule, CrossAttentionFusion, GatedFusion, GIAFusion,
    TriModalHierarchicalFusion, TriModalTensorFusion, TriModalBilinearFusion,
    TriModalAdaptiveWeightedFusion, TriModalEnhancedCrossAttentionFusion,
    TriModalGatedFusion, TriModalLateFusion, TriModalGLUFusion
)



def build_layer_weights(encoder) -> nn.Parameter:
    """Logits for a learned softmax over all hidden layers of an HF encoder.

    Pepino et al. (2021) report that for SER the useful paralinguistic information
    sits in the MIDDLE layers of wav2vec2, not the last one, and that a learned
    weighted sum over layers beats using `last_hidden_state`. This applies to any
    transformer speech encoder, not just WavLM.
    """
    n = 1 + int(getattr(encoder.config, "num_hidden_layers", 12))
    return nn.Parameter(torch.zeros(n))


def weighted_hidden_states(encoder, inputs, logits: torch.Tensor) -> torch.Tensor:
    """Softmax-weighted sum over the encoder's hidden layers -> (B, T, H)."""
    out = encoder(inputs, output_hidden_states=True)
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        raise RuntimeError(
            "weighted_hidden_states: the encoder returned no hidden_states."
        )
    stacked = torch.stack(hs, dim=0)                     # (L, B, T, H)
    w = torch.softmax(logits, dim=0).view(-1, 1, 1, 1)
    return (w * stacked).sum(dim=0)


def _transformer_layers(encoder):
    """Return the list of transformer blocks of a HuggingFace encoder, or None.

    Covers Wav2Vec2/WavLM (encoder.layers), BERT/RoBERTa/ViT (encoder.layer)
    and DistilBERT (transformer.layer).
    """
    for attr, sub in (("encoder", "layers"), ("encoder", "layer"), ("transformer", "layer")):
        parent = getattr(encoder, attr, None)
        if parent is not None:
            layers = getattr(parent, sub, None)
            if layers is not None and len(layers) > 0:
                return layers
    return None


def apply_finetuning_policy(encoder, freeze_all: bool, unfreeze_last_n: int,
                            name: str = "encoder",
                            freeze_feature_extractor: bool = True):
    """Apply the partial fine-tuning regime described in the paper.

    Semantics, driven by `unfreeze_last_n`:

      unfreeze_last_n > 0  -> PARTIAL: every parameter is frozen, then the last N
                              transformer blocks (plus the final layer norm and any
                              pooler/projection head) are unfrozen. This is what the
                              paper describes: "lower transformer blocks are frozen
                              ... only the upper transformer blocks are fine-tuned".
      unfreeze_last_n == 0 -> `freeze_all` decides: fully frozen or fully trainable.

    Before this function existed the partial regime was only reachable with
    `freeze_*_encoder: true`; every config that set `false` plus a non-zero
    `unfreeze_last_n` silently fine-tuned the WHOLE encoder, which is not what the
    paper or Figure 1 describe.

    For Wav2Vec2/WavLM the convolutional feature extractor stays frozen whenever any
    freezing applies, as the paper states.

    Returns:
        (total_params, trainable_params)
    """
    partial = bool(unfreeze_last_n and unfreeze_last_n > 0)

    if not partial and not freeze_all:
        mode = "full fine-tuning"
    else:
        for p in encoder.parameters():
            p.requires_grad = False
        mode = "frozen"

        if partial:
            layers = _transformer_layers(encoder)
            if layers is None:
                print(f"Warning [{name}]: no transformer blocks found; encoder stays frozen")
            else:
                n = min(unfreeze_last_n, len(layers))
                for layer in layers[-n:]:
                    for p in layer.parameters():
                        p.requires_grad = True
                # Final layer norm / pooler / projection heads ride with the top blocks
                for attr in ("layer_norm", "layernorm", "final_layer_norm"):
                    parent = getattr(encoder, "encoder", encoder)
                    mod = getattr(parent, attr, None) or getattr(encoder, attr, None)
                    if mod is not None and hasattr(mod, "parameters"):
                        for p in mod.parameters():
                            p.requires_grad = True
                for attr in ("pooler", "project_hid", "project_q"):
                    mod = getattr(encoder, attr, None)
                    if mod is not None and hasattr(mod, "parameters"):
                        for p in mod.parameters():
                            p.requires_grad = True
                mode = f"partial: last {n}/{len(layers)} blocks trainable"

        # Convolutional feature extractor (Wav2Vec2 / WavLM) stays frozen.
        if freeze_feature_extractor:
            fe = getattr(encoder, "feature_extractor", None)
            if fe is not None and hasattr(fe, "parameters"):
                for p in fe.parameters():
                    p.requires_grad = False
                mode += " (conv extractor frozen)"

    total = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"[finetune] {name}: {mode} -- {trainable:,}/{total:,} params trainable "
          f"({100.0 * trainable / max(total, 1):.1f}%)")
    return total, trainable


class ConformerMultimodalModel(nn.Module):
    """Multimodal emotion recognition model (audio + text, or placeholder for trimodal).

    Uses a pretrained audio encoder (Wav2Vec2/WavLM) and text encoder (BERT/DistilBERT),
    with configurable fusion (concat, cross_attention, gated, etc.). For trimodal
    types (e.g. conflict_aware_cross_attention), ablation models replace the fusion.
    """

    def __init__(
        self,
        audio_encoder_name: str = "facebook/wav2vec2-conformer-rel-pos-large",
        text_encoder_name: str = "distilbert-base-uncased",
        num_classes: int = 10,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        fusion_type: str = "cross_attention",
        fusion_num_heads: int = 8,
        freeze_audio_encoder: bool = False,
        audio_unfreeze_last_n_layers: int = 0,
        freeze_text_encoder: bool = False,
        text_unfreeze_last_n_layers: int = 0,
        audio_pooling: str = "mean"
    ):
        """
        Args:
            audio_encoder_name: HuggingFace model name for audio encoder
            text_encoder_name: HuggingFace model name for text encoder
            num_classes: Number of emotion classes
            hidden_dim: Hidden dimension for fusion module
            dropout: Dropout rate
            fusion_type: Type of fusion ('concat', 'cross_attention', 'gated')
            freeze_audio_encoder: Whether to freeze audio encoder
            freeze_text_encoder: Whether to freeze text encoder
            audio_pooling: Pooling strategy for audio features ('mean', 'max', 'attention')
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type
        self.audio_pooling = audio_pooling
        self.fusion_num_heads = fusion_num_heads
        
        # Audio encoder (Wav2Vec2 / WavLM via AutoModel, or Emotion2Vec via FunASR wrapper)
        if _is_emotion2vec_model(audio_encoder_name):
            try:
                self.audio_encoder = Emotion2VecEncoder(model_name=audio_encoder_name)
                audio_hidden_size = self.audio_encoder.config.hidden_size
            except ImportError as e:
                raise RuntimeError(
                    "Emotion2Vec encoder requested but FunASR is not installed. "
                    "Install with: pip install funasr"
                ) from e
        else:
            try:
                self.audio_encoder = AutoModel.from_pretrained(audio_encoder_name)
                audio_hidden_size = self.audio_encoder.config.hidden_size
            except Exception as e:
                print(f"Warning: Could not load audio encoder {audio_encoder_name}: {e}")
                print("Using default Wav2Vec2 configuration")
                config = Wav2Vec2Config()
                self.audio_encoder = Wav2Vec2Model(config)
                audio_hidden_size = config.hidden_size
        
        # Freezing strategy for audio encoder (see apply_finetuning_policy)
        apply_finetuning_policy(
            self.audio_encoder,
            freeze_all=freeze_audio_encoder,
            unfreeze_last_n=audio_unfreeze_last_n_layers,
            name=f"audio/{audio_encoder_name}",
            freeze_feature_extractor=True,
        )

        # Text encoder (BERT/DistilBERT)
        try:
            self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
            text_hidden_size = self.text_encoder.config.hidden_size
        except Exception as e:
            print(f"Warning: Could not load text encoder {text_encoder_name}: {e}")
            print("Using default BERT configuration")
            config = AutoConfig.from_pretrained("bert-base-uncased")
            self.text_encoder = AutoModel.from_config(config)
            text_hidden_size = config.hidden_size
        
        apply_finetuning_policy(
            self.text_encoder,
            freeze_all=freeze_text_encoder,
            unfreeze_last_n=text_unfreeze_last_n_layers,
            name=f"text/{text_encoder_name}",
            freeze_feature_extractor=False,
        )

        # Attention pooling for audio (if needed)
        if audio_pooling == "attention":
            self.audio_attention = nn.Sequential(
                nn.Linear(audio_hidden_size, 1),
                nn.Softmax(dim=1)
            )
        
        # Fusion module
        if fusion_type == "concat":
            self.fusion = FusionModule(
                audio_dim=audio_hidden_size,
                text_dim=text_hidden_size,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout
            )
        elif fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(
                audio_dim=audio_hidden_size,
                text_dim=text_hidden_size,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_heads=fusion_num_heads,
                dropout=dropout
            )
        elif fusion_type == "gated":
            self.fusion = GatedFusion(
                audio_dim=audio_hidden_size,
                text_dim=text_hidden_size,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout
            )
        elif fusion_type == "gia":
            self.fusion = GIAFusion(
                audio_dim=audio_hidden_size,
                text_dim=text_hidden_size,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_heads=fusion_num_heads,
                dropout=dropout
            )
        elif fusion_type in ("hierarchical", "tensor", "bilinear", "adaptive_weighted", 
                             "enhanced_cross_attention", "gated_trimodal", "late", "glu",
                             "temporal_bucketing", "conflict_aware_cross_attention"):
            # Trimodal fusion types - these will be replaced by child models
            # We create a placeholder here to avoid errors, but child models will override it
            # Using cross_attention as placeholder since it has similar interface
            self.fusion = CrossAttentionFusion(
                audio_dim=audio_hidden_size,
                text_dim=text_hidden_size,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_heads=fusion_num_heads,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def pool_audio_features(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool audio features across time dimension.
        
        Args:
            features: Audio features (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Pooled features (batch_size, hidden_size)
        """
        if self.audio_pooling == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(features.size())
                sum_features = torch.sum(features * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_features / sum_mask
            else:
                pooled = torch.mean(features, dim=1)
        
        elif self.audio_pooling == "max":
            pooled = torch.max(features, dim=1)[0]
        
        elif self.audio_pooling == "attention":
            # Attention-based pooling
            attn_weights = self.audio_attention(features)  # (batch_size, seq_len, 1)
            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    attention_mask.unsqueeze(-1) == 0,
                    float('-inf')
                )
            attn_weights = torch.softmax(attn_weights, dim=1)
            pooled = torch.sum(features * attn_weights, dim=1)
        
        else:
            raise ValueError(f"Unknown pooling method: {self.audio_pooling}")
        
        return pooled
    
    def forward(
        self,
        audio: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            audio: Audio waveform (batch_size, n_samples)
            text_input_ids: Text input IDs (batch_size, seq_len)
            text_attention_mask: Text attention mask (batch_size, seq_len)
            labels: Ground truth labels (batch_size,)
            
        Returns:
            Dictionary containing logits, loss, and predictions
        """
        # Extract audio features
        audio_outputs = self.audio_encoder(audio)
        audio_features = audio_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Pool audio features
        audio_pooled = self.pool_audio_features(audio_features)
        
        # Extract text features
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Fusion
        if self.fusion_type == "gated":
            logits, gate_weights = self.fusion(audio_pooled, text_features)
        else:
            logits = self.fusion(audio_pooled, text_features)
            gate_weights = None
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        
        output = {
            'logits': logits,
            'predictions': predictions,
            'loss': loss,
            'audio_features': audio_pooled,
            'text_features': text_features
        }
        
        if gate_weights is not None:
            output['gate_weights'] = gate_weights
        
        return output
    
    def freeze_audio_encoder(self):
        """Freeze audio encoder parameters."""
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_audio_encoder(self):
        """Unfreeze audio encoder parameters."""
        for param in self.audio_encoder.parameters():
            param.requires_grad = True
    
    def freeze_text_encoder(self):
        """Freeze text encoder parameters."""
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_text_encoder(self):
        """Unfreeze text encoder parameters."""
        for param in self.text_encoder.parameters():
            param.requires_grad = True
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """Get number of parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())


class AudioOnlyModel(nn.Module):
    """Audio-only baseline using a pretrained Wav2Vec2/Conformer encoder and a classifier head."""

    def __init__(
        self,
        audio_encoder_name: str = "facebook/wav2vec2-conformer-rel-pos-large",
        num_classes: int = 10,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
        audio_unfreeze_last_n_layers: int = 0
    ):
        """Initialize the audio-only model.

        Args:
            audio_encoder_name: HuggingFace model name for the audio encoder.
            num_classes: Number of output emotion classes.
            hidden_dim: Hidden dimension of the classifier MLP.
            dropout: Dropout rate for the classifier.
            freeze_encoder: If True, freeze encoder; optionally unfreeze last N layers.
            audio_unfreeze_last_n_layers: Number of top encoder layers to unfreeze (0 = none).
        """
        super().__init__()

        # Audio encoder
        self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_encoder_name)
        audio_hidden_size = self.audio_encoder.config.hidden_size
        
        # Freezing strategy (similar to ConformerMultimodalModel)
        if freeze_encoder:
            # Freeze all by default
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            # Optionally unfreeze last N transformer layers for fine-tuning
            if audio_unfreeze_last_n_layers and audio_unfreeze_last_n_layers > 0:
                try:
                    # Unfreeze last N encoder layers if present
                    if hasattr(self.audio_encoder, 'encoder') and hasattr(self.audio_encoder.encoder, 'layers'):
                        encoder_layers = self.audio_encoder.encoder.layers
                        for layer in encoder_layers[-audio_unfreeze_last_n_layers:]:
                            for p in layer.parameters():
                                p.requires_grad = True
                        # Unfreeze encoder layer_norm if exists
                        if hasattr(self.audio_encoder.encoder, 'layer_norm'):
                            for p in self.audio_encoder.encoder.layer_norm.parameters():
                                p.requires_grad = True
                    # Also unfreeze projection heads if present
                    if hasattr(self.audio_encoder, 'project_hid'):
                        for p in self.audio_encoder.project_hid.parameters():
                            p.requires_grad = True
                    if hasattr(self.audio_encoder, 'project_q'):
                        for p in self.audio_encoder.project_q.parameters():
                            p.requires_grad = True
                except Exception as e:
                    print(f"Warning: could not unfreeze last {audio_unfreeze_last_n_layers} layers of audio encoder: {e}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(audio_hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize classifier head with proper initialization
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize classifier head with proper weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                # Use Kaiming uniform initialization for better convergence (better for ReLU)
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    # Initialize bias to small random values to avoid collapse to single class
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(module.bias, -bound, bound)
    
    def forward(
        self,
        audio: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            audio: Raw audio waveform. Shape (batch_size, n_samples).
            labels: Optional ground-truth class indices. Shape (batch_size,).

        Returns:
            Dict with keys: logits (batch_size, num_classes), predictions (batch_size,),
            loss (scalar or None if labels not provided).
        """
        # Extract features
        outputs = self.audio_encoder(audio)
        features = outputs.last_hidden_state.mean(dim=1)
        
        # Classification
        logits = self.classifier(features)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        predictions = torch.argmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'predictions': predictions,
            'loss': loss
        }

