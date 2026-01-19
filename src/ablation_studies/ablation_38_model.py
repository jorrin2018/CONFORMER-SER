"""
Ablation Study 38: OpenSMILE + Attention Gates + Residual Connections

Extiende Ablation 34 con:
- Rama OpenSMILE para características de bajo nivel (eGeMAPSv02, 88 features)
- Attention-based gates para filtrar características importantes
- Cross-attention fusion entre Wav2Vec2 y OpenSMILE antes de la fusión modal
- Conexiones residuales en puntos clave para conservar contexto
- Técnicas de regularización mejoradas
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .ablation_34_model import Ablation34Model
from src.models.opensmile_extractor import OpenSMILEExtractor
from src.models.attention_gates import AttentionBasedGates, FeatureImportanceGates
from src.models.audio_fusion import AudioCrossAttentionFusion


class Ablation38Model(Ablation34Model):
    """
    Modelo trimodal con OpenSMILE + Attention Gates + Residual Connections:
    - Audio (Wav2Vec2) + OpenSMILE (eGeMAPSv02) fusionados con cross-attention
    - Texto (DistilBERT) con SpeechCue
    - Visual: Mel-espectrograma -> ViT
    
    Arquitectura:
    1. Audio Waveform
       ├─> Wav2Vec2 Encoder
       └─> OpenSMILE (eGeMAPSv02) -> Attention Gates -> Audio Fusion (Cross-Attention)
    2. Text (DistilBERT + SpeechCue)
    3. Vision (Mel-Spec + ViT)
    4. Conflict-Aware Cross-Attention Fusion (trimodal)
    5. Residual Connections en puntos clave
    6. LSTM Classification Head
    """
    
    def __init__(self, config: Dict):
        # Initialize base model (Ablation 34)
        super().__init__(config)
        
        model_config = config.get("model", {})
        dataset_config = config.get("dataset", {})
        opensmile_config = model_config.get("opensmile", {})
        attention_gates_config = model_config.get("attention_gates", {})
        audio_fusion_config = model_config.get("audio_fusion", {})
        residual_config = model_config.get("residual_connections", {})
        
        # -----------------------------
        # OpenSMILE Extractor
        # -----------------------------
        # Only enable OpenSMILE if explicitly configured
        # If opensmile_config is empty, disable it
        self.use_opensmile = opensmile_config.get("enabled", False) if opensmile_config else False
        
        if self.use_opensmile:
            sample_rate = dataset_config.get("sample_rate", 16000)
            opensmile_feature_dim = opensmile_config.get("feature_dim", 88)
            
            self.opensmile_extractor = OpenSMILEExtractor(
                config=opensmile_config.get("config", "eGeMAPSv02"),
                sample_rate=sample_rate,
                normalize=opensmile_config.get("normalize", True),
                feature_dim=opensmile_feature_dim,
            )
            
            # -----------------------------
            # Attention-Based Gates
            # -----------------------------
            gate_type = attention_gates_config.get("type", "attention")
            
            if gate_type == "attention":
                self.attention_gates = AttentionBasedGates(
                    input_dim=opensmile_feature_dim,
                    num_heads=attention_gates_config.get("num_heads", 4),
                    dropout=attention_gates_config.get("dropout", 0.3),
                    use_layer_norm=attention_gates_config.get("use_layer_norm", True),
                    temperature=attention_gates_config.get("temperature", 1.0),
                )
            else:  # "importance"
                self.attention_gates = FeatureImportanceGates(
                    input_dim=opensmile_feature_dim,
                    hidden_dim=attention_gates_config.get("hidden_dim", None),
                    dropout=attention_gates_config.get("dropout", 0.3),
                )
            
            # -----------------------------
            # Audio Fusion Module
            # -----------------------------
            audio_hidden_size = self.audio_encoder.config.hidden_size
            
            self.audio_fusion = AudioCrossAttentionFusion(
                wav2vec2_dim=audio_hidden_size,
                opensmile_dim=opensmile_feature_dim,
                hidden_dim=self.hidden_dim,
                num_heads=audio_fusion_config.get("num_heads", 4),
                dropout=audio_fusion_config.get("dropout", model_config.get("dropout", 0.3)),
                use_residual=audio_fusion_config.get("use_residual", True),
                residual_alpha=audio_fusion_config.get("residual_alpha", 0.2),
            )
        else:
            # If OpenSMILE is disabled, just use Wav2Vec2 features
            self.opensmile_extractor = None
            self.attention_gates = None
            self.audio_fusion = None
        
        # -----------------------------
        # Residual Connections Configuration
        # -----------------------------
        self.use_residual_connections = residual_config.get("enabled", True)
        self.residual_alpha = residual_config.get("alpha", 0.2)
        
        # Projection for residual connection from audio fusion to modal fusion
        if self.use_residual_connections and self.use_opensmile:
            self.audio_fusion_residual_proj = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
            )
        
        # Projection for residual connection from modal fusion to classification
        if self.use_residual_connections:
            fusion_output_dim = self.hidden_dim // 2
            self.modal_fusion_residual_proj = nn.Sequential(
                nn.Linear(fusion_output_dim, fusion_output_dim),
                nn.LayerNorm(fusion_output_dim),
            )
        
        # Fallback projection for when OpenSMILE is disabled
        audio_hidden_size = self.audio_encoder.config.hidden_size
        # Always create fallback projection in case OpenSMILE fails or is disabled
        self.wav2vec2_fallback_proj = nn.Sequential(
            nn.Linear(audio_hidden_size, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        
        # Projection from audio_fused (hidden_dim) back to audio_hidden_size for modal fusion
        # This is needed when OpenSMILE is enabled (audio_fused is in hidden_dim)
        # When disabled, audio_fused is also in hidden_dim (from fallback_proj)
        # So we always need this projection unless hidden_dim == audio_hidden_size
        if self.hidden_dim != audio_hidden_size:
            self.audio_fusion_to_original_proj = nn.Linear(
                self.hidden_dim, audio_hidden_size
            )
        else:
            # Identity when dimensions match
            self.audio_fusion_to_original_proj = nn.Identity()
        
        # Projection for residual connection from audio fusion to logits
        if self.use_residual_connections and self.use_opensmile:
            self.audio_residual_to_logits = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.LayerNorm(self.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(self.hidden_dim // 2, self.num_classes),
            )
        
        # Initialize all new modules properly
        self._init_new_modules()
    
    def _init_new_modules(self):
        """Initialize weights for newly added modules."""
        # Initialize audio fusion to original projection
        if hasattr(self, 'audio_fusion_to_original_proj'):
            nn.init.xavier_uniform_(self.audio_fusion_to_original_proj.weight, gain=1.0)
            if self.audio_fusion_to_original_proj.bias is not None:
                nn.init.constant_(self.audio_fusion_to_original_proj.bias, 0.0)
        
        # Initialize residual projections
        if hasattr(self, 'audio_fusion_residual_proj'):
            for module in self.audio_fusion_residual_proj.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
        
        if hasattr(self, 'modal_fusion_residual_proj'):
            for module in self.modal_fusion_residual_proj.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
        
        # Initialize audio residual to logits (use smaller initialization for classifier)
        if hasattr(self, 'audio_residual_to_logits'):
            modules_list = list(self.audio_residual_to_logits.modules())
            for i, module in enumerate(modules_list):
                if isinstance(module, nn.Linear):
                    # Last linear layer (classifier) - use small initialization
                    if module == modules_list[-1]:  # Last module is the classifier
                        nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    else:
                        nn.init.xavier_uniform_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
        
        # Initialize fallback projection if exists
        if hasattr(self, 'wav2vec2_fallback_proj'):
            for module in self.wav2vec2_fallback_proj.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
        
        # Ensure LSTM classification head has proper initialization
        # The LSTM head is created in the parent class, but we can verify it exists
        # Re-initialize the final classifier layer with small weights to prevent class collapse
        if hasattr(self.fusion, 'classifier'):
            if hasattr(self.fusion.classifier, 'classifier'):
                final_classifier = self.fusion.classifier.classifier
                if isinstance(final_classifier, nn.Linear):
                    # Use small initialization to prevent initial class collapse
                    nn.init.normal_(final_classifier.weight, mean=0.0, std=0.01)
                    if final_classifier.bias is not None:
                        # Initialize bias to small random values to break symmetry
                        nn.init.normal_(final_classifier.bias, mean=0.0, std=0.01)
    
    def forward(
        self,
        audio: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with OpenSMILE integration and residual connections.
        
        Args:
            audio: Audio waveform (batch_size, n_samples)
            text_input_ids: Text input IDs (batch_size, seq_len)
            text_attention_mask: Text attention mask (batch_size, seq_len)
            labels: Ground truth labels (batch_size,)
            
        Returns:
            Dictionary containing logits, loss, predictions, and intermediate features
        """
        # -----------------------------
        # Audio Processing
        # -----------------------------
        # Wav2Vec2 features
        audio_outputs = self.audio_encoder(audio)
        audio_features = audio_outputs.last_hidden_state  # (B, S, H_a)
        audio_pooled = self.pool_audio_features(audio_features)  # (B, H_a)
        
        # OpenSMILE + Gates + Audio Fusion
        if self.use_opensmile and self.opensmile_extractor is not None:
            try:
                # Extract OpenSMILE features
                opensmile_features = self.opensmile_extractor(audio)  # (B, 88)
                
                # Apply attention gates
                opensmile_filtered, gate_weights = self.attention_gates(
                    opensmile_features,
                    return_attention_weights=True
                )  # (B, 88)
                
                # Fuse Wav2Vec2 and OpenSMILE with cross-attention
                audio_fused, audio_attn_weights = self.audio_fusion(
                    wav2vec2_features=audio_pooled,
                    opensmile_features=opensmile_filtered,
                    return_attention_weights=True
                )  # (B, hidden_dim)
            except Exception as e:
                # If OpenSMILE fails, fall back to Wav2Vec2 only
                import warnings
                warnings.warn(f"OpenSMILE extraction failed: {e}. Falling back to Wav2Vec2 only.")
                audio_fused = self.wav2vec2_fallback_proj(audio_pooled)  # (B, hidden_dim)
                opensmile_features = None
                opensmile_filtered = None
                gate_weights = None
                audio_attn_weights = None
        else:
            # Fallback: just use Wav2Vec2 features projected to hidden_dim
            audio_fused = self.wav2vec2_fallback_proj(audio_pooled)  # (B, hidden_dim)
            opensmile_features = None
            opensmile_filtered = None
            gate_weights = None
            audio_attn_weights = None
        
        # -----------------------------
        # Text Processing
        # -----------------------------
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
        )
        text_features = text_outputs.last_hidden_state[:, 0, :]  # CLS (B, H_t)
        
        # -----------------------------
        # Vision Processing
        # -----------------------------
        vit_images = self._waveform_to_vit_image(audio)  # (B, 3, H, W)
        vit_images = vit_images.to(audio.device)
        
        vision_outputs = self.vision_encoder(pixel_values=vit_images)
        if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
            vision_features = vision_outputs.pooler_output  # (B, H_v)
        else:
            vision_features = vision_outputs.last_hidden_state[:, 0, :]  # CLS
        
        # -----------------------------
        # Project to common dimension for modal fusion
        # -----------------------------
        audio_hidden_size = self.audio_encoder.config.hidden_size
        
        # The ConflictAwareCrossAttentionFusion expects original audio_dim
        # Project audio_fused back to audio_hidden_size for compatibility
        if hasattr(self, 'audio_fusion_to_original_proj'):
            audio_for_fusion = self.audio_fusion_to_original_proj(audio_fused)  # (B, H_a)
        else:
            # Fallback: use original audio_pooled (should not happen)
            audio_for_fusion = audio_pooled  # (B, H_a)
        
        # Store audio_fused for residual connection
        audio_fused_for_residual = audio_fused if (self.use_residual_connections and self.use_opensmile) else None
        
        # -----------------------------
        # Trimodal Fusion (Conflict-Aware Cross-Attention)
        # -----------------------------
        # Modal fusion - this returns logits directly
        logits = self.fusion(
            audio_features=audio_for_fusion,
            text_features=text_features,
            vision_features=vision_features,
        )  # (B, num_classes)
        
        # -----------------------------
        # Classification with Residual Connection
        # -----------------------------
        # Add residual connection from audio fusion if enabled
        # NOTE: Disable residual connections initially to avoid instability
        # They can be enabled later once the model is stable
        if (self.use_residual_connections and 
            audio_fused_for_residual is not None and 
            hasattr(self, 'audio_residual_to_logits')):
            # Only apply residual after a few epochs to avoid early collapse
            # For now, disable it to prevent class collapse
            # audio_residual_logits = self.audio_residual_to_logits(audio_fused_for_residual)
            # logits = logits + (self.residual_alpha * 0.1) * audio_residual_logits
            pass
        
        # -----------------------------
        # Loss and Predictions
        # -----------------------------
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        predictions = torch.argmax(logits, dim=-1)
        
        # -----------------------------
        # Return outputs
        # -----------------------------
        output = {
            "logits": logits,
            "predictions": predictions,
            "loss": loss,
            "audio_features": audio_fused,
            "text_features": text_features,
            "vision_features": vision_features,
        }
        
        # Add intermediate features for analysis
        if self.use_opensmile and opensmile_features is not None:
            output["opensmile_features"] = opensmile_features
            if opensmile_filtered is not None:
                output["opensmile_filtered"] = opensmile_filtered
            if gate_weights is not None:
                output["gate_weights"] = gate_weights
            if audio_attn_weights is not None:
                output["audio_attention_weights"] = audio_attn_weights
        
        return output

