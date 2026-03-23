"""
Emotion2Vec audio encoder wrapper for use with ConformerMultimodalModel.

Emotion2Vec is loaded via FunASR (not Hugging Face Transformers). This module
exposes the same interface as Wav2Vec2/WavLM: forward(audio) returns an object
with .last_hidden_state of shape (batch_size, 1, hidden_size) for utterance-level
embeddings, compatible with the existing pooling in the pipeline.

Requires: pip install funasr (and optionally modelscope for some model IDs).
"""

import glob
import os
import tempfile
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchaudio


# Emotion2Vec base embedding dimension (documented for emotion2vec_base)
EMOTION2VEC_HIDDEN_SIZE = 768


def _is_emotion2vec_model(audio_encoder_name: str) -> bool:
    """Return True if the given model id refers to an Emotion2Vec model."""
    if not audio_encoder_name:
        return False
    return "emotion2vec" in audio_encoder_name.lower()


def _funasr_model_id(audio_encoder_name: str) -> str:
    """Map HuggingFace-style id to FunASR/ModelScope id if needed."""
    if "emotion2vec/emotion2vec_base" in audio_encoder_name or audio_encoder_name == "emotion2vec/emotion2vec_base":
        return "iic/emotion2vec_base"
    if "emotion2vec_plus_base" in audio_encoder_name:
        return "iic/emotion2vec_plus_base"
    if "emotion2vec_plus_seed" in audio_encoder_name:
        return "iic/emotion2vec_plus_seed"
    if "emotion2vec_plus_large" in audio_encoder_name:
        return "iic/emotion2vec_plus_large"
    return audio_encoder_name


class _SimpleConfig:
    """Minimal config object with hidden_size for compatibility with fusion code."""
    def __init__(self, hidden_size: int = EMOTION2VEC_HIDDEN_SIZE):
        self.hidden_size = hidden_size


def _embedding_from_result(res: Any, device: torch.device) -> Optional[torch.Tensor]:
    """Extract embedding tensor from FunASR result (dict, list, or array)."""
    if res is None:
        return None
    if isinstance(res, torch.Tensor):
        return res.to(device=device)
    if isinstance(res, dict):
        for key in ("emotion_embedding", "utt_x", "embedding", "feats"):
            if key in res:
                t = res[key]
                if isinstance(t, torch.Tensor):
                    return t.to(device=device)
                if isinstance(t, (list, tuple)):
                    t = np.array(t)
                return torch.tensor(t, dtype=torch.float32, device=device)
    if isinstance(res, (list, tuple)):
        arr = np.array(res)
        return torch.tensor(arr, dtype=torch.float32, device=device)
    if hasattr(res, "__array__"):
        return torch.tensor(np.asarray(res), dtype=torch.float32, device=device)
    return None


class Emotion2VecEncoder(nn.Module):
    """
    Wrapper around FunASR's Emotion2Vec model so it can be used as the audio
    encoder in ConformerMultimodalModel. Outputs utterance-level embeddings
    with the same interface as Wav2Vec2 (last_hidden_state).
    """

    def __init__(self, model_name: str, device: Optional[torch.device] = None):
        super().__init__()
        self._model_name = model_name
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from funasr import AutoModel as FunASRAutoModel
        except ImportError as e:
            raise ImportError(
                "Emotion2Vec requires FunASR. Install with: pip install funasr"
            ) from e

        funasr_id = _funasr_model_id(model_name)
        self._model = FunASRAutoModel(model=funasr_id, device=str(self._device))
        self.config = _SimpleConfig(hidden_size=EMOTION2VEC_HIDDEN_SIZE)

    def forward(self, audio: torch.Tensor) -> Any:
        """
        Args:
            audio: Waveform (batch_size, num_samples), 16 kHz expected.

        Returns:
            Object with .last_hidden_state of shape (batch_size, 1, hidden_size)
            for utterance-level embeddings (compatible with mean pooling).
        """
        device = audio.device
        B = audio.size(0)
        sample_rate = 16000
        embeddings: List[torch.Tensor] = []

        with tempfile.TemporaryDirectory(prefix="emotion2vec_") as tmpdir:
            for i in range(B):
                wav = audio[i].detach().to(device="cpu", dtype=torch.float32)
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                path = os.path.join(tmpdir, f"sample_{i}.wav")
                torchaudio.save(path, wav, sample_rate)

            for i in range(B):
                path = os.path.join(tmpdir, f"sample_{i}.wav")
                try:
                    res = self._model(
                        input=path,
                        output_dir=tmpdir,
                        granularity="utterance",
                        extract_embedding=True,
                    )
                    emb = _embedding_from_result(res, device)
                    if emb is not None:
                        emb = emb.flatten()[:EMOTION2VEC_HIDDEN_SIZE]
                        embeddings.append(emb)
                    else:
                        # Try loading from output_dir (FunASR may write .npy)
                        npy_files = glob.glob(os.path.join(tmpdir, "*.npy"))
                        if npy_files:
                            arr = np.load(npy_files[0])
                            emb = torch.tensor(arr, dtype=torch.float32, device=device).flatten()[:EMOTION2VEC_HIDDEN_SIZE]
                            embeddings.append(emb)
                except Exception:
                    pass

        if len(embeddings) != B:
            # Pad or use zeros for missing (e.g. API changed or error)
            while len(embeddings) < B:
                embeddings.append(torch.zeros(EMOTION2VEC_HIDDEN_SIZE, device=device, dtype=torch.float32))

        stacked = torch.stack(embeddings[:B], dim=0)
        if stacked.dim() == 2:
            stacked = stacked.unsqueeze(1)  # (B, 1, H)
        return type("Output", (), {"last_hidden_state": stacked})()
