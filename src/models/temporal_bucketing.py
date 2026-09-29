"""
Temporal Bucketing utilities for dividing sequences into temporal windows.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def divide_into_buckets(
    sequence: torch.Tensor,
    bucket_duration_ms: float,
    sample_rate: int = 16000,
    overlap_ms: float = 50.0,
    feature_rate: Optional[float] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Divide a temporal sequence into buckets (windows).
    
    Args:
        sequence: Temporal sequence tensor
                  - For waveforms: (batch_size, n_samples) or (n_samples,)
                  - For features: (batch_size, seq_len, feature_dim) or (seq_len, feature_dim)
        bucket_duration_ms: Duration of each bucket in milliseconds
        sample_rate: Sample rate for audio (used if sequence is waveform)
        overlap_ms: Overlap between buckets in milliseconds
        feature_rate: If provided, assumes sequence is already features with this rate (Hz)
                     If None, assumes sequence is raw waveform
    
    Returns:
        buckets: Tensor of shape (batch_size, num_buckets, bucket_size, ...)
        num_buckets: Number of buckets created
    """
    # Determine if input is waveform or features
    if sequence.dim() == 1:
        # Single sequence, add batch dimension
        sequence = sequence.unsqueeze(0)
        was_1d = True
    else:
        was_1d = False
    
    batch_size = sequence.shape[0]
    
    if sequence.dim() == 2:
        # Waveform: (batch_size, n_samples)
        is_waveform = True
        seq_len = sequence.shape[1]
        if feature_rate is None:
            # Raw waveform - use sample_rate
            time_per_sample = 1000.0 / sample_rate  # ms per sample
            bucket_size_samples = int(bucket_duration_ms / time_per_sample)
            overlap_samples = int(overlap_ms / time_per_sample)
        else:
            # Features with known rate
            time_per_sample = 1000.0 / feature_rate
            bucket_size_samples = int(bucket_duration_ms / time_per_sample)
            overlap_samples = int(overlap_ms / time_per_sample)
    elif sequence.dim() == 3:
        # Features: (batch_size, seq_len, feature_dim)
        is_waveform = False
        seq_len = sequence.shape[1]
        # For features, we need to know the feature rate
        # Wav2Vec2 typically outputs features at ~50Hz (20ms per frame)
        # We'll use a default if not provided
        if feature_rate is None:
            feature_rate = 50.0  # Default: 50Hz (20ms per frame)
        time_per_sample = 1000.0 / feature_rate
        bucket_size_samples = int(bucket_duration_ms / time_per_sample)
        overlap_samples = int(overlap_ms / time_per_sample)
    else:
        raise ValueError(f"Unsupported sequence dimension: {sequence.dim()}")
    
    # Calculate number of buckets
    if overlap_samples >= bucket_size_samples:
        overlap_samples = bucket_size_samples // 2  # Cap overlap at 50%
    
    step_size = bucket_size_samples - overlap_samples
    num_buckets = max(1, (seq_len - overlap_samples) // step_size)
    
    # Create buckets
    buckets = []
    for i in range(num_buckets):
        start_idx = i * step_size
        end_idx = start_idx + bucket_size_samples
        
        if end_idx > seq_len:
            # Last bucket - pad if necessary
            if is_waveform:
                bucket = sequence[:, start_idx:].clone()
                # Pad to bucket_size_samples
                padding_size = bucket_size_samples - bucket.shape[1]
                if padding_size > 0:
                    if sequence.dim() == 2:
                        padding = torch.zeros(
                            batch_size, padding_size, device=sequence.device, dtype=sequence.dtype
                        )
                    else:
                        padding = torch.zeros(
                            batch_size, padding_size, sequence.shape[2],
                            device=sequence.device, dtype=sequence.dtype
                        )
                    bucket = torch.cat([bucket, padding], dim=1)
            else:
                bucket = sequence[:, start_idx:end_idx].clone()
                if bucket.shape[1] < bucket_size_samples:
                    padding_size = bucket_size_samples - bucket.shape[1]
                    padding = torch.zeros(
                        batch_size, padding_size, sequence.shape[2],
                        device=sequence.device, dtype=sequence.dtype
                    )
                    bucket = torch.cat([bucket, padding], dim=1)
        else:
            bucket = sequence[:, start_idx:end_idx].clone()
        
        buckets.append(bucket)
    
    if len(buckets) == 0:
        # Fallback: single bucket with entire sequence
        if is_waveform:
            if sequence.dim() == 2:
                buckets = [sequence]
            else:
                buckets = [sequence.unsqueeze(0)]
        else:
            buckets = [sequence]
        num_buckets = 1
    
    # Stack buckets
    buckets_tensor = torch.stack(buckets, dim=1)  # (batch_size, num_buckets, bucket_size, ...)
    
    if was_1d:
        buckets_tensor = buckets_tensor.squeeze(0)
    
    return buckets_tensor, num_buckets


def pool_bucket_features(
    bucket_features: torch.Tensor,
    method: str = "mean",
) -> torch.Tensor:
    """
    Pool features within each bucket.
    
    Args:
        bucket_features: (batch_size, num_buckets, bucket_seq_len, feature_dim)
        method: Pooling method ('mean', 'max', 'last')
    
    Returns:
        pooled: (batch_size, num_buckets, feature_dim)
    """
    if method == "mean":
        pooled = torch.mean(bucket_features, dim=2)  # (B, num_buckets, feature_dim)
    elif method == "max":
        pooled = torch.max(bucket_features, dim=2)[0]  # (B, num_buckets, feature_dim)
    elif method == "last":
        pooled = bucket_features[:, :, -1, :]  # (B, num_buckets, feature_dim)
    else:
        raise ValueError(f"Unknown pooling method: {method}")
    
    return pooled







