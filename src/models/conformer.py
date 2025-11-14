"""
Conformer-based multimodal model for emotion recognition.
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
from .fusion import FusionModule, CrossAttentionFusion, GatedFusion


class ConformerMultimodalModel(nn.Module):
    """
    Multimodal emotion recognition model using Conformer for audio and BERT for text.
    """
    
    def __init__(
        self,
        audio_encoder_name: str = "facebook/wav2vec2-conformer-rel-pos-large",
        text_encoder_name: str = "distilbert-base-uncased",
        num_classes: int = 10,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        fusion_type: str = "cross_attention",
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
        
        # Audio encoder (Conformer/Wav2Vec2)
        try:
            self.audio_encoder = Wav2Vec2Model.from_pretrained(audio_encoder_name)
            audio_hidden_size = self.audio_encoder.config.hidden_size
        except Exception as e:
            print(f"Warning: Could not load audio encoder {audio_encoder_name}: {e}")
            print("Using default Wav2Vec2 configuration")
            config = Wav2Vec2Config()
            self.audio_encoder = Wav2Vec2Model(config)
            audio_hidden_size = config.hidden_size
        
        # Freezing strategy for audio encoder
        if freeze_audio_encoder:
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
        
        if freeze_text_encoder:
            # Freeze all by default
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            # Optionally unfreeze last N transformer layers for fine-tuning
            if text_unfreeze_last_n_layers and text_unfreeze_last_n_layers > 0:
                try:
                    # For DistilBERT/BERT, layers are in transformer.layer or encoder.layer
                    if hasattr(self.text_encoder, 'transformer') and hasattr(self.text_encoder.transformer, 'layer'):
                        # DistilBERT structure
                        encoder_layers = self.text_encoder.transformer.layer
                        for layer in encoder_layers[-text_unfreeze_last_n_layers:]:
                            for p in layer.parameters():
                                p.requires_grad = True
                    elif hasattr(self.text_encoder, 'encoder') and hasattr(self.text_encoder.encoder, 'layer'):
                        # BERT structure
                        encoder_layers = self.text_encoder.encoder.layer
                        for layer in encoder_layers[-text_unfreeze_last_n_layers:]:
                            for p in layer.parameters():
                                p.requires_grad = True
                except Exception as e:
                    print(f"Warning: could not unfreeze last {text_unfreeze_last_n_layers} layers of text encoder: {e}")
        
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
    """
    Audio-only baseline model using Conformer/Wav2Vec2.
    """
    
    def __init__(
        self,
        audio_encoder_name: str = "facebook/wav2vec2-conformer-rel-pos-large",
        num_classes: int = 10,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
        audio_unfreeze_last_n_layers: int = 0
    ):
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
        """Forward pass."""
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

