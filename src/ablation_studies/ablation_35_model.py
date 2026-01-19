"""
Ablation Study 35: Modal-Stratified Active Learning

Estrategia de entrenamiento donde el modelo identifica qué modalidad le confunde
más y puede solicitar etiquetas adicionales solo para esa modalidad.
Eficiencia de datos y reducción de costos de anotación.

NOTA: Esta es una estrategia de entrenamiento, no un cambio arquitectónico.
El modelo en sí es idéntico a Ablation 13, pero el training loop debe
implementar la lógica de Active Learning.
"""

from typing import Dict

from .ablation_25_enhanced_cross_attention import Ablation25Model


class Ablation35Model(Ablation25Model):
    """
    Modelo trimodal con Modal-Stratified Active Learning:
    - Audio (Wav2Vec2)
    - Texto (DistilBERT) con SpeechCue
    - Visual: Mel-espectrograma -> ViT (congelado)
    
    El modelo arquitectónico es idéntico a Ablation 25 (Enhanced Cross-Attention).
    La diferencia está en la estrategia de entrenamiento que debe:
    1. Identificar muestras con alta incertidumbre en una modalidad específica
    2. Solicitar etiquetas adicionales solo para esa modalidad
    3. Entrenar con esas muestras priorizadas
    
    Basado en Ablation 25 (Enhanced Cross-Attention).
    
    NOTA: La implementación completa requiere modificar el training loop
    en src/train.py para incluir la lógica de Active Learning.
    """

    def __init__(self, config: Dict):
        # Initialize exactly as Ablation 25 (Enhanced Cross-Attention base)
        # The Active Learning strategy is implemented in the training loop, not here
        super().__init__(config)
        
        # No architectural changes needed
        # The Active Learning logic should be in the training script

