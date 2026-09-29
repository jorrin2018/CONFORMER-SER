"""
Prosody cue extractor for internal SpeechCue-style prompts.

This module converts simple acoustic statistics (energy, variability,
speaking rate proxy) into short natural language descriptions that can
be prepended to the transcript and fed to the text encoder.
"""

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class ProsodyCueConfig:
    """Configuration for prosody cue extraction."""

    sample_rate: int = 16000
    # Thresholds for low/medium/high bucketing (heuristic)
    energy_low: float = 0.02
    energy_high: float = 0.08
    var_low: float = 0.02
    var_high: float = 0.08
    zcr_low: float = 0.02
    zcr_high: float = 0.08
    # Duration (in seconds) of analysis frames for variability / ZCR
    frame_duration: float = 0.5


class ProsodyCueExtractor:
    """
    Simple SpeechCue-style prosody descriptor.

    Given a waveform, returns a short description like:
    "high energy, high variability, fast speaking rate".
    """

    def __init__(self, config: ProsodyCueConfig):
        self.config = config

    def _bucket(self, value: float, low: float, high: float) -> str:
        if value < low:
            return "low"
        if value > high:
            return "high"
        return "medium"

    def _frame_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Split waveform into non-overlapping frames for statistics.
        Returns tensor of shape (num_frames, frame_len).
        """
        frame_len = int(self.config.frame_duration * self.config.sample_rate)
        if frame_len <= 0:
            return waveform.unsqueeze(0)

        total_len = waveform.shape[-1]
        if total_len <= frame_len:
            return waveform.unsqueeze(0)

        num_frames = total_len // frame_len
        trimmed = waveform[..., : num_frames * frame_len]
        return trimmed.view(num_frames, frame_len)

    def _zero_crossing_rate(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Approximate zero-crossing rate per frame.
        frames: (num_frames, frame_len)
        Returns: (num_frames,)
        """
        if frames.numel() == 0:
            return torch.zeros(1)

        signs = torch.sign(frames)
        # Zero crossings where sign changes between consecutive samples
        crossings = (signs[:, 1:] * signs[:, :-1] < 0).float()
        zcr = crossings.sum(dim=1) / frames.shape[1]
        return zcr

    def __call__(self, waveform: torch.Tensor) -> str:
        """
        Extract prosody cues from waveform.

        Args:
            waveform: Tensor of shape (1, n_samples) or (n_samples,)

        Returns:
            Natural language description of prosody.
        """
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        waveform = waveform.float()

        # Global energy and variability
        abs_wave = waveform.abs()
        energy = abs_wave.mean().item()
        variability = waveform.std().item()

        # Frame-level stats for speaking rate proxy (ZCR)
        frames = self._frame_waveform(waveform)
        zcr_per_frame = self._zero_crossing_rate(frames)
        zcr_mean = zcr_per_frame.mean().item()

        # Bucket into linguistic categories
        energy_level = self._bucket(energy, self.config.energy_low, self.config.energy_high)
        var_level = self._bucket(variability, self.config.var_low, self.config.var_high)
        zcr_level = self._bucket(zcr_mean, self.config.zcr_low, self.config.zcr_high)

        # Map ZCR bucket to an intuitive speaking rate description
        if zcr_level == "low":
            speaking_rate = "slow"
        elif zcr_level == "high":
            speaking_rate = "fast"
        else:
            speaking_rate = "moderate"

        # Build concise description
        description = (
            f"overall energy is {energy_level}, "
            f"variability is {var_level}, "
            f"speaking rate is {speaking_rate}"
        )
        return description



