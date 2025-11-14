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
    'create_ablation_optimizer',
]

