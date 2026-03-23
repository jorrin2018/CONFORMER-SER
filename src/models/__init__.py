"""Model components for multimodal SER: Conformer base and fusion modules.

Exposes ConformerMultimodalModel, AudioOnlyModel, and fusion variants
(FusionModule, CrossAttentionFusion, etc.) used by ablation studies.
"""

from .conformer import ConformerMultimodalModel, AudioOnlyModel
from .fusion import FusionModule, CrossAttentionFusion

__all__ = ['ConformerMultimodalModel', 'AudioOnlyModel', 'FusionModule', 'CrossAttentionFusion']

