"""
Audio and text preprocessing for IEMOCAP dataset.
"""

import torch
import torchaudio
import torchaudio.transforms as T
from transformers import AutoTokenizer
from typing import Dict, Optional
import numpy as np


class AudioPreprocessor:
    """
    Audio preprocessing using Mel-spectrograms or raw waveforms.
    For Conformer/Wav2Vec2 models, we typically use raw waveforms.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        use_mel_spectrogram: bool = False,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        apply_spec_augment: bool = False,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        apply_waveform_augment: bool = False,
        augmentation_config: Optional[Dict] = None
    ):
        """
        Args:
            sample_rate: Audio sample rate
            use_mel_spectrogram: If True, convert to mel-spectrogram; otherwise use raw waveform
            n_mels: Number of mel filterbanks
            n_fft: FFT size
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            apply_spec_augment: Whether to apply SpecAugment (for training only)
            freq_mask_param: Frequency mask parameter for SpecAugment
            time_mask_param: Time mask parameter for SpecAugment
        """
        self.sample_rate = sample_rate
        self.use_mel_spectrogram = use_mel_spectrogram
        self.apply_spec_augment = apply_spec_augment
        self.training = False  # Initialize training mode
        
        # Initialize waveform augmentation (for raw waveforms)
        if apply_waveform_augment and not use_mel_spectrogram:
            from src.data.augmentation import AudioAugmentation
            import inspect
            aug_config = augmentation_config or {}
            allowed = set(inspect.signature(AudioAugmentation.__init__).parameters)
            # #region agent log
            try:
                extra = set(aug_config) - allowed
                with open("d:\\DOCTORADO\\PROYECTOS\\CONFORMER-SER\\.cursor\\debug.log", "a") as _f:
                    _f.write('{"location":"preprocessing.py:AudioPreprocessor","message":"aug_config keys vs allowed","data":{"aug_keys":list(aug_config.keys()),"extra_keys":list(extra)},"hypothesisId":"H1,H2,H3"}\n')
            except Exception:
                pass
            # #endregion
            filtered = {k: v for k, v in aug_config.items() if k in allowed}
            self.audio_augment = AudioAugmentation(
                sample_rate=sample_rate,
                **filtered
            )
        else:
            self.audio_augment = None
        
        if use_mel_spectrogram:
            self.mel_transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=n_mels,
                normalized=True
            )
            
            if apply_spec_augment:
                self.freq_masking = T.FrequencyMasking(freq_mask_param)
                self.time_masking = T.TimeMasking(time_mask_param)
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Process audio waveform.
        
        Args:
            waveform: Audio tensor of shape (n_samples,)
            
        Returns:
            Processed audio features
        """
        if self.use_mel_spectrogram:
            # Convert to mel-spectrogram
            mel_spec = self.mel_transform(waveform)
            
            # Convert to log scale
            mel_spec = torch.log(mel_spec + 1e-9)
            
            # Apply SpecAugment if enabled
            if self.apply_spec_augment and self.training:
                mel_spec = self.freq_masking(mel_spec)
                mel_spec = self.time_masking(mel_spec)
            
            # Normalize
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
            
            return mel_spec
        else:
            # Return raw waveform (for Wav2Vec2/Conformer models)
            # Apply waveform augmentation if enabled
            if self.audio_augment is not None and self.training:
                waveform = self.audio_augment(waveform)
            
            # Normalize waveform
            waveform = waveform - waveform.mean()
            waveform = waveform / (waveform.std() + 1e-9)
            return waveform
    
    def train(self):
        """Set to training mode (enables augmentation)."""
        self.training = True
        if self.audio_augment is not None:
            self.audio_augment.train()
    
    def eval(self):
        """Set to evaluation mode (disables augmentation)."""
        self.training = False
        if self.audio_augment is not None:
            self.audio_augment.eval()


class TextPreprocessor:
    """
    Text preprocessing using transformers tokenizers.
    """
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        padding: str = "max_length",
        truncation: bool = True,
        apply_text_augment: bool = False,
        text_augmentation_config: Optional[Dict] = None,
        additional_tokens: Optional[list] = None
    ):
        """
        Args:
            model_name: HuggingFace model name for tokenizer
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate sequences
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # SpeechCue's nine symbols are added as real tokens, so the existing
        # word-piece vocabulary is untouched and the encoder gains exactly nine
        # embeddings. The caller must then resize the text encoder to
        # `self.vocab_size` (see create_dataloaders / Ablation34Model).
        self.n_added_tokens = 0
        if additional_tokens:
            self.n_added_tokens = self.tokenizer.add_tokens(list(additional_tokens))
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.training = False  # Initialize training mode
        
        # Initialize text augmentation
        if apply_text_augment:
            from src.data.augmentation import TextAugmentation
            aug_config = text_augmentation_config or {}
            self.text_augment = TextAugmentation(
                tokenizer=self.tokenizer,
                **aug_config
            )
        else:
            self.text_augment = None
    
    @property
    def vocab_size(self) -> int:
        """Vocabulary size after adding SpeechCue tokens."""
        return len(self.tokenizer)

    def __call__(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Handle empty text
        if not text or text.strip() == "":
            text = "[EMPTY]"
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # Apply text augmentation if enabled
        if self.text_augment is not None and self.training:
            augmented = self.text_augment(input_ids, attention_mask)
            input_ids = augmented['input_ids']
            attention_mask = augmented['attention_mask']
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def train(self):
        """Set to training mode (enables augmentation)."""
        self.training = True
        if self.text_augment is not None:
            self.text_augment.train()
    
    def eval(self):
        """Set to evaluation mode (disables augmentation)."""
        self.training = False
        if self.text_augment is not None:
            self.text_augment.eval()


class SpecAugment:
    """
    SpecAugment implementation for data augmentation.
    Reference: https://arxiv.org/abs/1904.08779
    """
    
    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        n_freq_masks: int = 2,
        n_time_masks: int = 2
    ):
        """
        Args:
            freq_mask_param: Maximum frequency mask width
            time_mask_param: Maximum time mask width
            n_freq_masks: Number of frequency masks to apply
            n_time_masks: Number of time masks to apply
        """
        self.freq_masking = T.FrequencyMasking(freq_mask_param)
        self.time_masking = T.TimeMasking(time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment.
        
        Args:
            spec: Spectrogram tensor of shape (channels, freq, time)
            
        Returns:
            Augmented spectrogram
        """
        # Apply frequency masking
        for _ in range(self.n_freq_masks):
            spec = self.freq_masking(spec)
        
        # Apply time masking
        for _ in range(self.n_time_masks):
            spec = self.time_masking(spec)
        
        return spec


def normalize_audio(waveform: torch.Tensor, method: str = "peak") -> torch.Tensor:
    """
    Normalize audio waveform.
    
    Args:
        waveform: Audio tensor
        method: Normalization method ('peak', 'rms', or 'zscore')
        
    Returns:
        Normalized waveform
    """
    if method == "peak":
        # Peak normalization
        max_val = torch.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val
    elif method == "rms":
        # RMS normalization
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > 0:
            waveform = waveform / (rms * 10)  # Scale to reasonable range
    elif method == "zscore":
        # Z-score normalization
        mean = waveform.mean()
        std = waveform.std()
        if std > 0:
            waveform = (waveform - mean) / std
    
    return waveform


def extract_features(
    waveform: torch.Tensor,
    sample_rate: int,
    feature_type: str = "mfcc",
    **kwargs
) -> torch.Tensor:
    """
    Extract acoustic features from waveform.
    
    Args:
        waveform: Audio tensor
        sample_rate: Sample rate
        feature_type: Type of features to extract ('mfcc', 'mel', 'spectrogram')
        **kwargs: Additional arguments for feature extraction
        
    Returns:
        Extracted features
    """
    if feature_type == "mfcc":
        mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=kwargs.get('n_mfcc', 40),
            melkwargs={
                'n_fft': kwargs.get('n_fft', 400),
                'hop_length': kwargs.get('hop_length', 160),
                'n_mels': kwargs.get('n_mels', 80)
            }
        )
        return mfcc_transform(waveform)
    
    elif feature_type == "mel":
        mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=kwargs.get('n_fft', 400),
            hop_length=kwargs.get('hop_length', 160),
            n_mels=kwargs.get('n_mels', 80)
        )
        return mel_transform(waveform)
    
    elif feature_type == "spectrogram":
        spec_transform = T.Spectrogram(
            n_fft=kwargs.get('n_fft', 400),
            hop_length=kwargs.get('hop_length', 160)
        )
        return spec_transform(waveform)
    
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

