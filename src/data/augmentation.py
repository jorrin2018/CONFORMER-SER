"""
Data augmentation for audio and text in SER.
"""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import Dict, Optional, Tuple


class AudioAugmentation(nn.Module):
    """
    Audio augmentation for raw waveforms (SER).
    All augmentations are applied with probability.
    """
    
    def __init__(
        self,
        apply_time_masking: bool = True,
        apply_freq_masking: bool = True, # Add freq masking
        apply_noise: bool = True,
        apply_speed_perturb: bool = True,
        apply_pitch_shift: bool = False,  # Disabled by default, requires torchaudio >= 0.9.0
        
        # Time masking parameters
        time_mask_prob: float = 0.5,
        time_mask_max: int = 4000,  # samples (~0.25s at 16kHz)
        num_time_masks: int = 1,
        
        # Freq masking parameters
        freq_mask_prob: float = 0.3,
        freq_mask_max: int = 10,
        num_freq_masks: int = 1,
        
        # Noise parameters
        noise_prob: float = 0.3,
        noise_snr_min: float = 10.0,
        noise_snr_max: float = 30.0,
        
        # Speed perturbation
        speed_prob: float = 0.3,
        speed_range: Tuple[float, float] = (0.9, 1.1),  # ±10%
        
        # Pitch shift
        pitch_prob: float = 0.2,
        pitch_range: Tuple[int, int] = (-2, 2),  # semitones
        
        sample_rate: int = 16000
    ):
        super().__init__()
        self.time_mask_prob = time_mask_prob
        self.noise_prob = noise_prob
        self.speed_prob = speed_prob
        self.pitch_prob = pitch_prob
        self.freq_mask_prob = freq_mask_prob
        
        self.apply_time_masking = apply_time_masking
        self.apply_freq_masking = apply_freq_masking
        self.apply_noise = apply_noise
        self.apply_speed_perturb = apply_speed_perturb
        self.apply_pitch_shift = apply_pitch_shift
        
        self.time_mask_max = time_mask_max
        self.num_time_masks = num_time_masks
        self.freq_mask_max = freq_mask_max
        self.num_freq_masks = num_freq_masks
        self.noise_snr_min = noise_snr_min
        self.noise_snr_max = noise_snr_max
        self.speed_range = speed_range
        self.pitch_range = pitch_range
        self.sample_rate = sample_rate
        
        # Initialize pitch shift effect (optional, requires torchaudio >= 0.9.0)
        self.pitch_shift = None
        if apply_pitch_shift:
            try:
                import torchaudio
                # Use time-stretch + resampling for pitch shift if PitchShift not available
                self.pitch_shift = True  # Just a flag
            except ImportError:
                self.pitch_shift = None
                self.apply_pitch_shift = False
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply waveform augmentations (pure waveform, Wav2Vec2 compatible).
        
        Order of application:
        1. Time masking (first - most effective for SER)
        2. Speed perturbation (preserves emotion better than pitch shift)
        3. Gaussian noise (last - adds realism)
        
        Note: Frequency masking is disabled as it requires spectrogram conversion
        which is incompatible with Wav2Vec2's waveform input requirement.
        
        Args:
            waveform: Audio tensor (n_samples,)
            
        Returns:
            Augmented waveform
        """
        if not self.training:
            return waveform
        
        aug_waveform = waveform.clone()
        
        # Order matters for augmentation effectiveness:
        
        # 1. Time Masking (most effective for SER, preserves frequency content)
        if self.apply_time_masking and random.random() < self.time_mask_prob:
            aug_waveform = self._apply_time_masking(aug_waveform)
        
        # 2. Speed Perturbation (changes pitch + duration, preserves emotion)
        #    Apply before noise to maintain signal characteristics
        if self.apply_speed_perturb and random.random() < self.speed_prob:
            aug_waveform = self._apply_speed_perturbation(aug_waveform)
        
        # 3. Gaussian Noise (adds realism, applied last)
        if self.apply_noise and random.random() < self.noise_prob:
            aug_waveform = self._add_gaussian_noise(aug_waveform)
        
        # Note: Frequency masking disabled - requires spectrogram conversion
        # which can cause numerical instability and is incompatible with Wav2Vec2
        
        # Note: Pitch shift disabled by default - requires torchaudio >= 0.9.0
        # and can distort emotion more than speed perturbation
        
        return aug_waveform
    
    def _apply_time_masking(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply time masking to waveform."""
        waveform_len = len(waveform)
        if waveform_len <= self.time_mask_max:
            return waveform
        
        for _ in range(self.num_time_masks):
            mask_len = random.randint(0, min(self.time_mask_max, waveform_len // 10))
            if mask_len == 0:
                continue
            
            mask_start = random.randint(0, waveform_len - mask_len)
            waveform[mask_start:mask_start + mask_len] = 0.0
        
        return waveform
    
    def _apply_freq_masking(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency masking (requires conversion to spectrogram).
        
        WARNING: This method is disabled by default because:
        1. Requires spectrogram conversion which can cause numerical instability
        2. Inverse spectrogram conversion is lossy and can degrade audio quality
        3. Incompatible with Wav2Vec2's requirement for pure waveform input
        4. Can introduce artifacts that negatively impact model performance
        
        Returns:
            Original waveform (no augmentation applied)
        """
        # Disabled for Wav2Vec2 compatibility
        # If needed, implement frequency-domain filtering directly on waveform
        # or use mel-spectrogram pipeline separately
        return waveform
    
    def _add_gaussian_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise with random SNR."""
        signal_power = torch.mean(waveform ** 2)
        if signal_power == 0:
            return waveform
        
        snr_db = random.uniform(self.noise_snr_min, self.noise_snr_max)
        snr_linear = 10 ** (snr_db / 10.0)
        
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        
        return waveform + noise
    
    def _apply_speed_perturbation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply speed perturbation (changes pitch and duration)."""
        speed_factor = random.uniform(*self.speed_range)
        
        if speed_factor == 1.0:
            return waveform
        
        # Resample to change speed using interpolation
        original_length = len(waveform)
        new_length = int(original_length / speed_factor)
        
        # Create indices for interpolation
        indices = torch.linspace(0, original_length - 1, new_length, device=waveform.device)
        
        # Interpolate
        indices_floor = indices.floor().long()
        indices_ceil = torch.clamp(indices_floor + 1, max=original_length - 1)
        indices_frac = indices - indices_floor.float()
        
        # Linear interpolation
        aug_waveform = (
            waveform[indices_floor] * (1 - indices_frac) +
            waveform[indices_ceil] * indices_frac
        )
        
        # Truncate or pad to original length
        if len(aug_waveform) > original_length:
            aug_waveform = aug_waveform[:original_length]
        elif len(aug_waveform) < original_length:
            padding = original_length - len(aug_waveform)
            aug_waveform = torch.nn.functional.pad(aug_waveform, (0, padding))
        
        return aug_waveform
    
    def _apply_pitch_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply pitch shift using time-stretch and resampling."""
        try:
            import torchaudio
            n_steps = random.randint(*self.pitch_range)
            if n_steps == 0:
                return waveform
            
            # Simple pitch shift: resample and then stretch back
            # This is approximate but works without external libraries
            pitch_factor = 2 ** (n_steps / 12.0)
            
            # Resample
            new_length = int(len(waveform) * pitch_factor)
            indices = torch.linspace(0, len(waveform) - 1, new_length, device=waveform.device)
            indices_floor = indices.floor().long()
            indices_ceil = torch.clamp(indices_floor + 1, max=len(waveform) - 1)
            indices_frac = indices - indices_floor.float()
            
            shifted = (
                waveform[indices_floor] * (1 - indices_frac) +
                waveform[indices_ceil] * indices_frac
            )
            
            # Stretch back to original length
            if len(shifted) != len(waveform):
                indices_back = torch.linspace(0, len(shifted) - 1, len(waveform), device=waveform.device)
                indices_back_floor = indices_back.floor().long()
                indices_back_ceil = torch.clamp(indices_back_floor + 1, max=len(shifted) - 1)
                indices_back_frac = indices_back - indices_back_floor.float()
                
                shifted = (
                    shifted[indices_back_floor] * (1 - indices_back_frac) +
                    shifted[indices_back_ceil] * indices_back_frac
                )
            
            return shifted
        except Exception:
            # If pitch shift fails, return original
            return waveform


class TextAugmentation(nn.Module):
    """
    Text augmentation for SER.
    Uses simple techniques that preserve emotion labels.
    """
    
    def __init__(
        self,
        apply_random_mask: bool = True,
        apply_random_insert: bool = False,  # More aggressive, disabled by default
        
        mask_prob: float = 0.15,
        insert_prob: float = 0.1,
        
        tokenizer=None  # Tokenizer for masking tokens
    ):
        super().__init__()
        self.apply_random_mask = apply_random_mask
        self.apply_random_insert = apply_random_insert
        self.mask_prob = mask_prob
        self.insert_prob = insert_prob
        self.tokenizer = tokenizer
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Apply augmentations to text tokens.
        
        Args:
            input_ids: Token IDs (seq_len,)
            attention_mask: Attention mask (seq_len,)
            
        Returns:
            Dictionary with augmented input_ids and attention_mask
        """
        if not self.training:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        
        aug_input_ids = input_ids.clone()
        aug_attention_mask = attention_mask.clone()
        
        # Random masking (only for non-special tokens)
        if self.apply_random_mask and self.tokenizer is not None:
            aug_input_ids, aug_attention_mask = self._apply_random_masking(
                aug_input_ids, aug_attention_mask
            )
        
        # Random insertion (optional, more aggressive)
        if self.apply_random_insert and self.tokenizer is not None and random.random() < self.insert_prob:
            aug_input_ids, aug_attention_mask = self._apply_random_insertion(
                aug_input_ids, aug_attention_mask
            )
        
        return {
            'input_ids': aug_input_ids,
            'attention_mask': aug_attention_mask
        }
    
    def _apply_random_masking(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random token masking (BERT-style)."""
        if self.tokenizer is None:
            return input_ids, attention_mask
        
        # Check if tokenizer has mask token
        mask_token_id = None
        if hasattr(self.tokenizer, 'mask_token_id') and self.tokenizer.mask_token_id is not None:
            mask_token_id = self.tokenizer.mask_token_id
        elif hasattr(self.tokenizer, 'mask_token'):
            try:
                mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
            except:
                pass
        
        if mask_token_id is None:
            return input_ids, attention_mask
        
        # Get special token IDs
        cls_token_id = None
        sep_token_id = None
        pad_token_id = None
        
        if hasattr(self.tokenizer, 'cls_token_id'):
            cls_token_id = self.tokenizer.cls_token_id
        if hasattr(self.tokenizer, 'sep_token_id'):
            sep_token_id = self.tokenizer.sep_token_id
        if hasattr(self.tokenizer, 'pad_token_id'):
            pad_token_id = self.tokenizer.pad_token_id
        
        special_tokens = {cls_token_id, sep_token_id, pad_token_id}
        special_tokens.discard(None)
        
        # Find valid positions to mask
        valid_positions = []
        for i in range(len(input_ids)):
            if attention_mask[i].item() == 1:
                token_id = input_ids[i].item()
                if token_id not in special_tokens:
                    valid_positions.append(i)
        
        if len(valid_positions) == 0:
            return input_ids, attention_mask
        
        # Randomly mask some tokens (BERT-style: 80% MASK, 10% random, 10% unchanged)
        num_masks = max(1, int(len(valid_positions) * self.mask_prob))
        mask_positions = random.sample(valid_positions, min(num_masks, len(valid_positions)))
        
        for pos in mask_positions:
            rand_val = random.random()
            if rand_val < 0.8:  # 80% replace with MASK token
                input_ids[pos] = mask_token_id
            elif rand_val < 0.9:  # 10% replace with random token
                # Random token from vocabulary
                if hasattr(self.tokenizer, 'vocab_size'):
                    vocab_size = self.tokenizer.vocab_size
                    random_token_id = random.randint(100, vocab_size - 1)  # Skip first 100 (usually special)
                    input_ids[pos] = random_token_id
            # 10% unchanged
        
        return input_ids, attention_mask
    
    def _apply_random_insertion(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random token insertion (more aggressive augmentation)."""
        # For now, this is a placeholder - can be implemented if needed
        # Random insertion is more complex and may not preserve sequence length
        return input_ids, attention_mask

