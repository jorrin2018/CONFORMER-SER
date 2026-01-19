"""
Ablation studies for Conformer multimodal emotion recognition.
"""

from .base_config import load_base_config
from .classification_heads import (
    LSTMClassificationHead,
    TransformerClassificationHead,
    CNN1DClassificationHead
)
from .ablation_01_model import Ablation01Model
from .ablation_02_model import Ablation02Model
from .ablation_03_model import Ablation03Model
from .ablation_04_model import Ablation04Model
from .ablation_05_model import Ablation05Model
from .ablation_06_model import Ablation06Model
from .ablation_07_model import Ablation07Model
from .ablation_08_model import Ablation08Model
from .ablation_09_model import Ablation09Model
from .ablation_10_model import Ablation10Model
from .ablation_11_model import Ablation11Model
from .ablation_13_model import Ablation13Model
from .ablation_21_hierarchical_fusion import Ablation21Model
from .ablation_22_tensor_fusion import Ablation22Model
from .ablation_23_bilinear_fusion import Ablation23Model
from .ablation_24_adaptive_weighted_fusion import Ablation24Model
from .ablation_25_enhanced_cross_attention import Ablation25Model
from .ablation_26_gated_fusion import Ablation26Model
from .ablation_27_late_fusion import Ablation27Model
from .ablation_28_glu_fusion import Ablation28Model
from .ablation_30_dirichlet import Ablation30Model
from .ablation_31_model import Ablation31Model
from .ablation_32_model import Ablation32Model
from .ablation_33_model import Ablation33Model, NoiseAwareProsodyCueExtractor
from .ablation_34_model import Ablation34Model
from .ablation_35_model import Ablation35Model
from .ablation_36a_model import Ablation36AModel
from .ablation_36b_model import Ablation36BModel
from .ablation_36c_model import Ablation36CModel, DirichletNeutralRegularizedLoss
from .ablation_38_model import Ablation38Model
from .utils import create_ablation_optimizer

__all__ = [
    'load_base_config',
    'LSTMClassificationHead',
    'TransformerClassificationHead',
    'CNN1DClassificationHead',
    'Ablation01Model',
    'Ablation02Model',
    'Ablation03Model',
    'Ablation04Model',
    'Ablation05Model',
    'Ablation06Model',
    'Ablation07Model',
    'Ablation08Model',
    'Ablation09Model',
    'Ablation10Model',
    'Ablation11Model',
    'Ablation13Model',
    'Ablation21Model',
    'Ablation22Model',
    'Ablation23Model',
    'Ablation24Model',
    'Ablation25Model',
    'Ablation26Model',
    'Ablation27Model',
    'Ablation28Model',
    'Ablation30Model',
    'Ablation31Model',
    'Ablation32Model',
    'Ablation33Model',
    'Ablation34Model',
    'Ablation35Model',
    'Ablation36AModel',
    'Ablation36BModel',
    'Ablation36CModel',
    'Ablation38Model',
    'DirichletNeutralRegularizedLoss',
    'NoiseAwareProsodyCueExtractor',
    'create_ablation_optimizer',
]

