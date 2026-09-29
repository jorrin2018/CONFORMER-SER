"""
Alternative viewer for training results without TensorBoard.
Shows metrics from checkpoint JSON files.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def load_experiment_metrics(experiment_dir):
    """Load metrics from an experiment directory."""
    metrics_file = Path(experiment_dir) / 'metrics.json'
    
    if not metrics_file.exists():
        return None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    return metrics

def plot_training_curves(metrics):
    """Plot training and validation curves."""
    epochs = sorted(metrics.keys(), key=lambda x: int(x))
    
    train_losses = [metrics[epoch].get('train_loss', 0) for epoch in epochs]
    val_losses = [metrics[epoch].get('val_loss', 0) for epoch in epochs]
    train_ua = [metrics[epoch].get('train_unweighted_accuracy', 0) for epoch in epochs]
    val_ua = [metrics[epoch].get('val_unweighted_accuracy', 0) for epoch in epochs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(train_ua, label='Train UA', marker='o')
    ax2.plot(val_ua, label='Val UA', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Unweighted Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def show_latest_results():
    """Show results from the most recent experiment."""
    # Anclado a la raiz del repositorio (padre de scripts/), no al directorio actual.
    logs_dir = Path(__file__).resolve().parent.parent / 'logs'
    
    if not logs_dir.exists():
        print("No logs directory found!")
        return
    
    # Find all experiment directories
    experiments = [d for d in logs_dir.iterdir() if d.is_dir() and d.name != '__pycache__']
    
    if not experiments:
        print("No experiments found in logs directory!")
        return
    
    # Sort by name (timestamp) to get most recent
    experiments.sort(key=lambda x: x.name, reverse=True)
    
    print("="*60)
    print("TRAINING RESULTS SUMMARY")
    print("="*60)
    
    for exp_dir in experiments[:3]:  # Show top 3 most recent
        metrics = load_experiment_metrics(exp_dir)
        
        if metrics is None:
            continue
        
        print(f"\nExperiment: {exp_dir.name}")
        print("-" * 60)
        
        # Get latest epoch
        epochs = sorted(metrics.keys(), key=lambda x: int(x))
        if not epochs:
            print("No epoch data found")
            continue
        
        latest_epoch = epochs[-1]
        latest_metrics = metrics[latest_epoch]
        
        print(f"Latest Epoch: {latest_epoch}")
        print(f"Train Loss: {latest_metrics.get('train_loss', 'N/A'):.4f}")
        print(f"Val Loss: {latest_metrics.get('val_loss', 'N/A'):.4f}")
        print(f"Train UA: {latest_metrics.get('train_unweighted_accuracy', 'N/A'):.2%}")
        print(f"Val UA: {latest_metrics.get('val_unweighted_accuracy', 'N/A'):.2%}")
        
        # Calculate overfitting gap
        train_ua = latest_metrics.get('train_unweighted_accuracy', 0)
        val_ua = latest_metrics.get('val_unweighted_accuracy', 0)
        gap = train_ua - val_ua
        print(f"Overfitting Gap: {gap:.2%}")
        
        # Plot if matplotlib is available
        try:
            fig = plot_training_curves(metrics)
            output_path = exp_dir / 'training_curves.png'
            fig.savefig(output_path)
            print(f"\nTraining curves saved to: {output_path}")
            plt.close(fig)
        except Exception as e:
            print(f"Could not generate plots: {e}")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    show_latest_results()

