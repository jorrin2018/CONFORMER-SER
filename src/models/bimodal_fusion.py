"""Fusion audio + texto de dos tokens y detector de conflicto con cross-fitting (123-126).

POR QUE EXISTE. Conclusion de la revision de noviembre de 2026:

1. Quitar la rama mel de la arquitectura. El estudio 98 la "apagaba" sustituyendo su
   vector por ceros, que deja un tercer token constante aprendido. Aqui la fusion tiene
   DOS tokens (audio y texto) y el ViT ni se carga.

2. Quitar el detector de conflicto simple (clasificadores auxiliares lineales + argmax +
   confidence_net) y decidir el peso de cada rama con lo unico que si funciono en la
   prueba sin reentrenar: los expertos de audio y texto entrenados por separado, con sus
   predicciones FUERA DE MUESTRA (cross-fitting). Combinados por una logistica, esas
   predicciones dieron 76.13 % frente a 74.63 % del promedio (4 de 5 folds); la
   compuerta simple, en sus cuatro formas, quedo entre -0.8 y -0.1 pp.

EL DETECTOR (detector: "crossfit"). Por enunciado recibe 14 rasgos de los expertos:
log-probabilidades de audio (4) y texto (4), entropia y probabilidad maxima de cada uno,
si coinciden, y log(1 + palabras). Una red pequena los convierte en pesos (w_a, w_t) que
multiplican los dos tokens antes del MLP de fusion. La ultima capa arranca en cero, asi
que al empezar w = (1/2, 1/2): el modelo parte exactamente de los pesos fijos y el
detector solo se aparta si eso baja la perdida.

Por que fuera de muestra: en train, un experto que ya vio el enunciado acierta casi
siempre y el detector aprenderia "confia en cualquiera"; en test los expertos fallan
como en la vida real. Con cross-fitting las dos situaciones se parecen. Los rasgos los
calcula src/data/expert_gate.load_expert_features desde cache/experts/fold_S<k>.json.

detector: "ninguno" = pesos fijos (1/2, 1/2): es el estudio 123, la base de los demas.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

N_RASGOS = 14
DETECTORES = ("ninguno", "crossfit")


class FusionBimodal(nn.Module):
    def __init__(self, audio_dim: int, text_dim: int, hidden_dim: int, num_classes: int,
                 num_heads: int = 8, num_layers: int = 3, dropout: float = 0.3,
                 detector: str = "ninguno", detector_oculto: int = 32):
        super().__init__()
        if detector not in DETECTORES:
            raise ValueError(f"model.detector_conflicto.tipo={detector!r}; usa uno de {DETECTORES}")
        self.detector_tipo = detector
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        # Misma atencion entre ramas que ConflictAwareCrossAttentionFusion, con 2 tokens.
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.detector = None
        if detector == "crossfit":
            self.detector = nn.Sequential(
                nn.Linear(N_RASGOS, detector_oculto), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(detector_oculto, 2))
            nn.init.zeros_(self.detector[-1].weight)
            nn.init.zeros_(self.detector[-1].bias)
        # Mismo MLP de fusion que la version de tres flujos, con entrada 2 * hidden.
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout * 0.6),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.GELU(), nn.Dropout(dropout * 0.4),
            nn.Linear(hidden_dim // 4, num_classes))
        # Para el diagnostico (no participan en el grafo).
        self.branch_logits = None
        self.branch_confidence = None
        self.last_weights = None

    def pesos(self, B: int, ref: torch.Tensor,
              expert_feats: Optional[torch.Tensor]) -> torch.Tensor:
        if self.detector is None:
            return ref.new_full((B, 2), 0.5)
        if expert_feats is None:
            raise RuntimeError("El detector con cross-fitting necesita 'expert_feats' en el batch")
        return torch.softmax(self.detector(expert_feats.to(ref.dtype)), dim=-1)

    def forward(self, audio_features: torch.Tensor, text_features: torch.Tensor,
                expert_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens = torch.stack([self.audio_proj(audio_features), self.text_proj(text_features)], 1)
        for attn, norm in zip(self.cross_attn_layers, self.layer_norms):
            out, _ = attn(tokens, tokens, tokens, need_weights=False)
            tokens = norm(tokens + out)
        w = self.pesos(tokens.size(0), tokens, expert_feats)          # (B, 2)
        self.last_weights = w.detach()
        # x2: con w = (1/2, 1/2) los tokens pasan sin escalar.
        tokens = tokens * (2.0 * w).unsqueeze(-1)
        return self.classifier(self.fusion(tokens.reshape(tokens.size(0), -1)))
