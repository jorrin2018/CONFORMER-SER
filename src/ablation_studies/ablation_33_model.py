"""
Ablation Study 33: Noise-Aware SpeechCue

Modifica el generador de prompts SpeechCue para detectar ruido de fondo.
Si el audio es ruidoso, inyecta un token [UNCERTAIN] al texto,
avisando a BERT que "desconfíe" de la prosodia.
"""

from typing import Dict

import torch

from .ablation_25_enhanced_cross_attention import Ablation25Model
from src.data.prosody_cues import ProsodyCueExtractor, ProsodyCueConfig


class NoiseAwareProsodyCueExtractor(ProsodyCueExtractor):
    """
    Extractor de prosody cues que detecta ruido y inyecta token [UNCERTAIN]
    cuando el audio es ruidoso.
    """

    def __init__(self, config: ProsodyCueConfig, noise_config: Dict):
        super().__init__(config)
        self.noise_config = noise_config
        self.noise_detection_method = noise_config.get("noise_detection_method", "snr_threshold")
        self.noise_snr_threshold = noise_config.get("noise_snr_threshold", 20.0)
        self.uncertainty_token = noise_config.get("uncertainty_token", "[UNCERTAIN]")
        self.inject_position = noise_config.get("inject_position", "before_prosody")

    def _estimate_snr(self, waveform: torch.Tensor) -> float:
        """
        Estima el SNR (Signal-to-Noise Ratio) del audio.
        Método simple: compara la energía de la señal con la varianza residual.
        """
        
        # Normalizar
        waveform = waveform.float()
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        
        # Estimar señal (componente de baja frecuencia)
        # y ruido (componente de alta frecuencia)
        # Método simplificado: usar filtro de media móvil
        window_size = min(100, len(waveform) // 10)
        if window_size < 3:
            window_size = 3
        
        # Señal estimada (suavizada)
        kernel = torch.ones(window_size) / window_size
        if waveform.is_cuda:
            kernel = kernel.cuda()
        signal_est = torch.conv1d(
            waveform.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=window_size // 2
        ).squeeze()
        
        # Ruido estimado (residual)
        noise_est = waveform - signal_est
        
        # Calcular SNR en dB
        signal_power = (signal_est ** 2).mean().item()
        noise_power = (noise_est ** 2).mean().item()
        
        if noise_power < 1e-10:
            return 100.0  # Muy limpio
        
        snr_db = 10 * torch.log10(torch.tensor(signal_power / noise_power)).item()
        return snr_db

    def _detect_noise(self, waveform: torch.Tensor) -> bool:
        """
        Detecta si el audio es ruidoso.
        """
        if self.noise_detection_method == "snr_threshold":
            snr = self._estimate_snr(waveform)
            return snr < self.noise_snr_threshold
        else:
            # Por defecto, no detectar ruido
            return False

    def __call__(self, waveform: torch.Tensor) -> str:
        """
        Extrae prosody cues y añade token [UNCERTAIN] si hay ruido.
        """
        
        # Detectar ruido
        is_noisy = self._detect_noise(waveform)
        
        # Obtener descripción de prosody normal
        prosody_desc = super().__call__(waveform)
        
        # Inyectar token de incertidumbre si hay ruido
        if is_noisy:
            if self.inject_position == "before_prosody":
                return f"{self.uncertainty_token} {prosody_desc}"
            elif self.inject_position == "after_prosody":
                return f"{prosody_desc} {self.uncertainty_token}"
            elif self.inject_position == "replace_prosody":
                return f"{self.uncertainty_token} audio quality is poor, prosody cues unreliable"
            else:
                return f"{self.uncertainty_token} {prosody_desc}"
        
        return prosody_desc


class Ablation33Model(Ablation25Model):
    """
    Modelo trimodal con Noise-Aware SpeechCue:
    - Audio (Wav2Vec2)
    - Texto (DistilBERT) con SpeechCue que detecta ruido e inyecta [UNCERTAIN]
    - Visual: Mel-espectrograma -> ViT (congelado)
    
    Fusión: Enhanced Cross-Attention (heredado de Ablation 25)
    Basado en Ablation 25 (Enhanced Cross-Attention).
    """

    def __init__(self, config: Dict):
        # Inicializar como Ablation 25 (Enhanced Cross-Attention base)
        super().__init__(config)

        # La modificación principal está en el dataset, no en el modelo.
        # El dataset usará NoiseAwareProsodyCueExtractor en lugar de ProsodyCueExtractor.
        # Este modelo es idéntico a Ablation 25, pero el dataset debe configurarse
        # para usar el extractor con detección de ruido.
        pass

