"""
OpenSMILE feature extractor for low-level audio features.

Extracts handcrafted features using OpenSMILE toolkit with eGeMAPSv02 configuration
(88 features for emotion recognition).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict
import warnings

try:
    import opensmile
except ImportError:
    opensmile = None
    warnings.warn(
        "opensmile not installed. Install with: pip install opensmile"
    )


class OpenSMILEExtractor(nn.Module):
    """
    Extracts OpenSMILE features from audio waveforms.
    
    Uses eGeMAPSv02 configuration which provides 88 low-level acoustic features
    designed for emotion recognition tasks.
    """
    
    def __init__(
        self,
        config: str = "eGeMAPSv02",
        sample_rate: int = 16000,
        normalize: bool = True,
        feature_dim: int = 88,
    ):
        """
        Args:
            config: OpenSMILE configuration name (default: "eGeMAPSv02")
            sample_rate: Audio sample rate (default: 16000)
            normalize: Whether to normalize features (default: True)
            feature_dim: Expected feature dimension (88 for eGeMAPSv02)
        """
        super().__init__()
        
        if opensmile is None:
            raise ImportError(
                "opensmile is required. Install with: pip install opensmile"
            )
        
        self.config = config
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.feature_dim = feature_dim
        
        # Initialize OpenSMILE feature extractor
        try:
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
        except Exception as e:
            # Fallback to basic configuration
            warnings.warn(f"Could not initialize eGeMAPSv02, using default: {e}")
            try:
                self.smile = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.ComParE_2016,
                    feature_level=opensmile.FeatureLevel.Functionals,
                )
                self.feature_dim = 6373  # ComParE has more features
                warnings.warn(f"Using ComParE_2016 instead, feature_dim={self.feature_dim}")
            except Exception as e2:
                raise RuntimeError(f"Could not initialize OpenSMILE: {e2}")
        
        # Normalization parameters (learned during training)
        if normalize:
            self.register_buffer("feature_mean", torch.zeros(feature_dim))
            self.register_buffer("feature_std", torch.ones(feature_dim))
            self.register_buffer("feature_count", torch.tensor(0))
    
    def _extract_features_numpy(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract OpenSMILE features from numpy array.
        
        Args:
            waveform: Audio waveform as numpy array (n_samples,)
            
        Returns:
            Features as numpy array (feature_dim,)
        """
        try:
            # OpenSMILE expects float32 in range [-1, 1]
            if waveform.dtype != np.float32:
                waveform = waveform.astype(np.float32)
            
            # Normalize to [-1, 1] range (peak normalization)
            max_val = np.abs(waveform).max()
            if max_val > 0:
                waveform = waveform / max_val
            
            # Extract features using process_signal
            # process_signal returns a pandas DataFrame
            features_df = self.smile.process_signal(waveform, self.sample_rate)
            
            # Convert DataFrame to numpy array
            # The DataFrame has shape (1, n_features) for functionals
            if hasattr(features_df, 'values'):
                # Get values as numpy array
                features = features_df.values
                # Flatten to 1D array
                if features.ndim > 1:
                    features = features.flatten()
                else:
                    features = features
            elif hasattr(features_df, 'to_numpy'):
                # Alternative: use to_numpy() method
                features = features_df.to_numpy().flatten()
            else:
                # Fallback: convert to array
                features = np.array(features_df).flatten()
            
            # Ensure correct dimension
            if len(features) != self.feature_dim:
                # Pad or truncate if necessary
                if len(features) > self.feature_dim:
                    features = features[:self.feature_dim]
                    warnings.warn(
                        f"OpenSMILE returned {len(features)} features, "
                        f"truncated to {self.feature_dim}"
                    )
                else:
                    padding = np.zeros(self.feature_dim - len(features), dtype=np.float32)
                    features = np.concatenate([features, padding])
                    warnings.warn(
                        f"OpenSMILE returned {len(features) - len(padding)} features, "
                        f"padded to {self.feature_dim}"
                    )
            
            return features.astype(np.float32)
            
        except Exception as e:
            warnings.warn(f"Error extracting OpenSMILE features: {e}")
            # Return zeros as fallback
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def _update_normalization_stats(self, features: torch.Tensor):
        """
        Update running statistics for normalization.
        
        Args:
            features: Batch of features (batch_size, feature_dim)
        """
        if not self.normalize:
            return
        
        batch_size = features.size(0)
        batch_mean = features.mean(dim=0)
        batch_std = features.std(dim=0) + 1e-8
        
        # Update running statistics
        total_count = self.feature_count + batch_size
        delta = batch_mean - self.feature_mean
        self.feature_mean += delta * (batch_size / total_count)
        
        # Update std using Welford's algorithm
        m2 = self.feature_std ** 2 * self.feature_count + batch_std ** 2 * batch_size
        self.feature_std = torch.sqrt(m2 / total_count + 1e-8)
        self.feature_count = total_count
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract OpenSMILE features from audio waveform.
        
        Args:
            waveform: Audio waveform tensor (batch_size, n_samples) or (n_samples,)
            
        Returns:
            Features tensor (batch_size, feature_dim)
        """
        # Handle single sample
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        batch_size = waveform.size(0)
        device = waveform.device
        
        # Extract features for each sample in batch
        features_list = []
        for i in range(batch_size):
            # Convert to numpy
            audio_np = waveform[i].cpu().detach().numpy()
            
            # Extract features
            features_np = self._extract_features_numpy(audio_np)
            features_list.append(features_np)
        
        # Stack into batch tensor
        features = torch.from_numpy(np.stack(features_list)).to(device)
        
        # Normalize if enabled
        if self.normalize:
            if self.training:
                # Update statistics during training
                self._update_normalization_stats(features)
            
            # Apply normalization
            features = (features - self.feature_mean) / (self.feature_std + 1e-8)
        
        return features
    
    def extract_features_batch(
        self,
        waveforms: torch.Tensor,
        return_tensor: bool = True
    ) -> torch.Tensor:
        """
        Extract features for a batch of waveforms.
        
        Args:
            waveforms: Audio waveforms (batch_size, n_samples)
            return_tensor: Whether to return as torch.Tensor (default: True)
            
        Returns:
            Features as tensor (batch_size, feature_dim) or list of arrays
        """
        return self.forward(waveforms)
    
    def get_feature_names(self) -> list:
        """
        Get names of extracted features.
        
        Returns:
            List of feature names
        """
        try:
            # Create dummy signal to get feature names
            # Use a short signal (1 second) for feature name extraction
            dummy_signal = np.random.randn(self.sample_rate).astype(np.float32)
            # Normalize
            dummy_signal = dummy_signal / (np.abs(dummy_signal).max() + 1e-8)
            
            features_df = self.smile.process_signal(dummy_signal, self.sample_rate)
            
            # Extract column names from DataFrame
            if hasattr(features_df, 'columns'):
                feature_names = features_df.columns.tolist()
                # Ensure we have the right number of names
                if len(feature_names) != self.feature_dim:
                    # If we have more names, truncate; if fewer, pad
                    if len(feature_names) > self.feature_dim:
                        feature_names = feature_names[:self.feature_dim]
                    else:
                        feature_names.extend([
                            f"feature_{i}" 
                            for i in range(len(feature_names), self.feature_dim)
                        ])
                return feature_names
            else:
                return [f"feature_{i}" for i in range(self.feature_dim)]
        except Exception as e:
            warnings.warn(f"Could not get feature names: {e}")
            return [f"feature_{i}" for i in range(self.feature_dim)]

