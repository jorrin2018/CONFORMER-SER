"""
Logging utilities for training and evaluation.
"""

import os
import sys
import io
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np


def setup_logger(
    name: str,
    log_dir: str,
    log_file: str = 'training.log',
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger for training.
    
    Args:
        name: Logger name
        log_dir: Directory to save logs
        log_file: Log file name
        level: Logging level
        
    Returns:
        Logger instance
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler with UTF-8 encoding for Windows compatibility
    file_handler = logging.FileHandler(
        os.path.join(log_dir, log_file),
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler with UTF-8 encoding for Windows compatibility
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    # Set encoding for stdout on Windows to handle Unicode characters
    if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            # Fallback for older Python versions
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


class TensorBoardLogger:
    """
    TensorBoard logger for training metrics and visualizations.
    """
    
    def __init__(self, log_dir: str, experiment_name: Optional[str] = None):
        """
        Args:
            log_dir: Base directory for logs
            experiment_name: Name of the experiment (default: timestamp)
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.log_dir = Path(log_dir) / experiment_name
        self.writer = SummaryWriter(str(self.log_dir))
        
        print(f"TensorBoard logging to: {self.log_dir}")
        print(f"To view, run: tensorboard --logdir={log_dir}")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars."""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ''):
        """
        Log multiple metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Global step
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{prefix}{name}", value, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log histogram of values."""
        self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """Log an image."""
        self.writer.add_image(tag, image, step)
    
    def log_figure(self, tag: str, figure, step: int):
        """Log a matplotlib figure."""
        self.writer.add_figure(tag, figure, step)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text."""
        self.writer.add_text(tag, text, step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """
        Log hyperparameters and final metrics.
        
        Args:
            hparams: Dictionary of hyperparameters
            metrics: Dictionary of final metrics
        """
        self.writer.add_hparams(hparams, metrics)
    
    def close(self):
        """Close the writer."""
        self.writer.close()


def convert_to_native_types(obj):
    """
    Convert numpy and torch types to native Python types for JSON serialization.
    
    Args:
        obj: Object to convert (dict, list, or scalar)
        
    Returns:
        Object with native Python types
    """
    if isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_native_types(obj.tolist())
    elif isinstance(obj, torch.Tensor):
        return convert_to_native_types(obj.item() if obj.numel() == 1 else obj.tolist())
    else:
        return obj


class ExperimentTracker:
    """
    Track experiment configurations and results.
    """
    
    def __init__(self, save_dir: str, experiment_name: Optional[str] = None):
        """
        Args:
            save_dir: Directory to save experiment info
            experiment_name: Name of the experiment
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = {}
        self.metrics = {}
        self.best_metrics = {}
        
        print(f"Experiment tracking at: {self.save_dir}")
    
    def log_config(self, config: Dict[str, Any]):
        """Save experiment configuration."""
        self.config = config
        config_path = self.save_dir / 'config.json'
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to: {config_path}")
    
    def log_metrics(self, metrics: Dict[str, float], epoch: int):
        """Log metrics for an epoch."""
        if epoch not in self.metrics:
            self.metrics[epoch] = {}
        
        # Convert numpy types to native Python types before storing
        metrics_native = convert_to_native_types(metrics)
        self.metrics[epoch].update(metrics_native)
        
        # Save metrics
        metrics_path = self.save_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def update_best_metrics(self, metrics: Dict[str, float]):
        """Update best metrics."""
        # Convert numpy types to native Python types
        self.best_metrics = convert_to_native_types(metrics)
        
        best_path = self.save_dir / 'best_metrics.json'
        with open(best_path, 'w') as f:
            json.dump(self.best_metrics, f, indent=2)
    
    def save_checkpoint_info(self, checkpoint_path: str, epoch: int, metrics: Dict[str, float]):
        """Save checkpoint information."""
        # Convert numpy types to native Python types
        metrics_native = convert_to_native_types(metrics)
        
        checkpoint_info = {
            'checkpoint_path': checkpoint_path,
            'epoch': epoch,
            'metrics': metrics_native,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoints_file = self.save_dir / 'checkpoints.json'
        
        # Load existing checkpoints
        if checkpoints_file.exists():
            with open(checkpoints_file, 'r') as f:
                checkpoints = json.load(f)
        else:
            checkpoints = []
        
        checkpoints.append(checkpoint_info)
        
        # Save updated checkpoints
        with open(checkpoints_file, 'w') as f:
            json.dump(checkpoints, f, indent=2)
    
    def save_summary(self, summary: str):
        """Save experiment summary."""
        summary_path = self.save_dir / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write(summary)


class MetricTracker:
    """
    Track metrics during training.
    """
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, **kwargs):
        """Update metrics."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            self.metrics[key].append(value)
    
    def get_average(self, key: str, last_n: Optional[int] = None) -> float:
        """Get average of a metric."""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        
        values = self.metrics[key]
        if last_n is not None:
            values = values[-last_n:]
        
        return sum(values) / len(values)
    
    def get_all_averages(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """Get averages of all metrics."""
        return {key: self.get_average(key, last_n) for key in self.metrics.keys()}
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
    
    def get_latest(self, key: str) -> Optional[float]:
        """Get the latest value of a metric."""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return None
        return self.metrics[key][-1]


def print_model_summary(model: torch.nn.Module, logger: Optional[logging.Logger] = None):
    """
    Print model summary.
    
    Args:
        model: PyTorch model
        logger: Logger instance
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
{'='*60}
MODEL SUMMARY
{'='*60}
Total Parameters:     {total_params:,}
Trainable Parameters: {trainable_params:,}
Non-trainable Parameters: {total_params - trainable_params:,}
{'='*60}
"""
    
    if logger:
        logger.info(summary)
    else:
        print(summary)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params
    }

