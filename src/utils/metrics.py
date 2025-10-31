"""
Evaluation metrics for emotion recognition.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class EmotionMetrics:
    """
    Compute emotion recognition metrics.
    """
    
    def __init__(self, emotion_labels: List[str]):
        """
        Args:
            emotion_labels: List of emotion label names
        """
        self.emotion_labels = emotion_labels
        self.num_classes = len(emotion_labels)
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and labels."""
        self.predictions = []
        self.labels = []
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor):
        """
        Update metrics with new batch.
        
        Args:
            predictions: Predicted labels (batch_size,)
            labels: Ground truth labels (batch_size,)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Ensure 1D arrays (flatten if needed)
        predictions = predictions.flatten()
        labels = labels.flatten()
        
        self.predictions.extend(predictions)
        self.labels.extend(labels)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric names and values
        """
        # Check if we have any predictions/labels
        if len(self.predictions) == 0 or len(self.labels) == 0:
            # Return zero metrics if no predictions/labels were collected
            return {
                'accuracy': 0.0,
                'weighted_accuracy': 0.0,
                'unweighted_accuracy': 0.0,
                'f1_weighted': 0.0,
                'f1_macro': 0.0,
                'precision_weighted': 0.0,
                'recall_weighted': 0.0,
                'precision_macro': 0.0,
                'recall_macro': 0.0,
                **{f'{emotion}_accuracy': 0.0 for emotion in self.emotion_labels},
                **{f'{emotion}_f1': 0.0 for emotion in self.emotion_labels}
            }
        
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        # Weighted Accuracy (WA)
        wa = accuracy_score(labels, predictions)
        
        # Unweighted Accuracy (UA) - average of per-class recalls
        cm = confusion_matrix(labels, predictions, labels=range(self.num_classes))
        per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-9)
        ua = per_class_acc.mean()
        
        # F1 scores
        f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        f1_per_class = f1_score(
            labels, predictions,
            average=None,
            labels=range(self.num_classes),
            zero_division=0
        )
        
        # Precision and Recall
        precision_weighted = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(labels, predictions, average='weighted', zero_division=0)
        
        precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
        
        # Per-class metrics
        per_class_metrics = {}
        for i, emotion in enumerate(self.emotion_labels):
            per_class_metrics[f'{emotion}_accuracy'] = per_class_acc[i]
            per_class_metrics[f'{emotion}_f1'] = f1_per_class[i]
        
        metrics = {
            'accuracy': wa,
            'weighted_accuracy': wa,
            'unweighted_accuracy': ua,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            **per_class_metrics
        }
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix.
        
        Returns:
            Confusion matrix (num_classes, num_classes)
        """
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        return confusion_matrix(labels, predictions, labels=range(self.num_classes))
    
    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        normalize: bool = True,
        figsize: tuple = (12, 10)
    ):
        """
        Plot confusion matrix.
        
        Args:
            save_path: Path to save figure
            normalize: Whether to normalize by true labels
            figsize: Figure size
        """
        cm = self.get_confusion_matrix()
        
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.emotion_labels,
            yticklabels=self.emotion_labels,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def get_classification_report(self) -> str:
        """
        Get classification report.
        
        Returns:
            Classification report string
        """
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        # Get unique classes present in the data
        unique_labels = np.unique(np.concatenate([labels, predictions]))
        
        # Only include labels that are present in the data
        present_labels = unique_labels[unique_labels < len(self.emotion_labels)]
        target_names = [self.emotion_labels[i] for i in present_labels]
        
        return classification_report(
            labels,
            predictions,
            labels=present_labels,
            target_names=target_names,
            zero_division=0
        )
    
    def print_report(self):
        """Print detailed classification report."""
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        
        # Get unique classes present in the data
        unique_labels = np.unique(np.concatenate([labels, predictions]))
        present_labels = unique_labels[unique_labels < len(self.emotion_labels)]
        
        metrics = self.compute()
        
        print("\n" + "="*60)
        print("EMOTION RECOGNITION EVALUATION REPORT")
        print("="*60)
        
        print(f"\nOverall Metrics:")
        print(f"  Weighted Accuracy (WA): {metrics['weighted_accuracy']:.4f}")
        print(f"  Unweighted Accuracy (UA): {metrics['unweighted_accuracy']:.4f}")
        print(f"  F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"  F1-Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"  Precision (Weighted): {metrics['precision_weighted']:.4f}")
        print(f"  Recall (Weighted): {metrics['recall_weighted']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        print(f"{'Emotion':<15} {'Accuracy':<10} {'F1-Score':<10}")
        print("-" * 40)
        for label_idx in present_labels:
            emotion = self.emotion_labels[label_idx]
            acc = metrics[f'{emotion}_accuracy']
            f1 = metrics[f'{emotion}_f1']
            print(f"{emotion:<15} {acc:<10.4f} {f1:<10.4f}")
        
        print("\n" + "="*60)
        
        print("\nDetailed Classification Report:")
        print(self.get_classification_report())


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    emotion_labels: List[str]
) -> Dict[str, float]:
    """
    Compute metrics from predictions and labels.
    
    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        emotion_labels: List of emotion label names
        
    Returns:
        Dictionary of metrics
    """
    metrics_calculator = EmotionMetrics(emotion_labels)
    metrics_calculator.update(predictions, labels)
    return metrics_calculator.compute()


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int,
    method: str = 'balanced'
) -> torch.Tensor:
    """
    Compute class weights for imbalanced dataset.
    
    Args:
        labels: All training labels
        num_classes: Number of classes
        method: Weighting method ('balanced' or 'effective')
        
    Returns:
        Class weights tensor
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    
    if method == 'balanced':
        # Inverse frequency
        total = len(labels)
        weights = total / (num_classes * class_counts + 1e-9)
    
    elif method == 'effective':
        # Effective number of samples
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / (effective_num + 1e-9)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize weights
    weights = weights / weights.sum() * num_classes
    
    return torch.tensor(weights, dtype=torch.float32)


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: List[float],
    val_metrics: List[float],
    metric_name: str = 'Accuracy',
    save_path: Optional[str] = None
):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_metrics: List of training metrics
        val_metrics: List of validation metrics
        metric_name: Name of the metric
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', marker='o', markersize=3)
    ax1.plot(val_losses, label='Val Loss', marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Metric curves
    ax2.plot(train_metrics, label=f'Train {metric_name}', marker='o', markersize=3)
    ax2.plot(val_metrics, label=f'Val {metric_name}', marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(metric_name)
    ax2.set_title(f'Training and Validation {metric_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

