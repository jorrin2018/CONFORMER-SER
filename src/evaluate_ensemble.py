"""
Ensemble evaluation script for combining multiple ablation models.
Implements weighted ensemble of top-performing models (Ablation 25, 34, 30).
"""

import os
import sys
import yaml
import torch
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (IEMOCAPDataset, AudioPreprocessor, TextPreprocessor, collate_fn,
                      build_dataset, corpus_from_config)
from scripts.run_ablation import create_ablation_model, load_ablation_config
from src.ablation_studies import load_base_config
from src.utils.metrics import EmotionMetrics
from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate ensemble of ablation models')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                        help='Paths to model checkpoints (space-separated)')
    parser.add_argument('--configs', type=str, nargs='+', required=True,
                        help='Paths to config files for each checkpoint (space-separated, same order)')
    parser.add_argument('--study_numbers', type=int, nargs='+', required=True,
                        help='Ablation study numbers for each checkpoint (space-separated, same order)')
    parser.add_argument('--weights', type=float, nargs='+', default=None,
                        help='Ensemble weights (optional, defaults to equal weights)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test', 'all'],
                        help='Which split to evaluate')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--method', type=str, default='weighted_average',
                        choices=['weighted_average', 'voting', 'stacking'],
                        help='Ensemble method to use')
    return parser.parse_args()


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: str,
    study_number: int,
    device: str
) -> torch.nn.Module:
    """Load ablation model from checkpoint."""
    print(f"\nLoading model from {checkpoint_path}...")
    print(f"  Config: {config_path}")
    print(f"  Study: {study_number}")
    
    # Load base config
    base_config = load_base_config()
    
    # Load ablation-specific config
    config = load_ablation_config(config_path, base_config)
    
    # Create model
    logger = setup_logger('ensemble', './logs/ensemble')
    model = create_ablation_model(study_number, config, device, logger)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Try loading directly (some checkpoints save model directly)
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"  ✓ Model loaded successfully!")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"  Metrics: {checkpoint['metrics']}")
    
    return model, config


def create_dataloader(config: dict, split: str):
    """Create dataloader for evaluation."""
    print(f"\nCreating {split} dataloader...")
    
    # Initialize preprocessors (no augmentation for evaluation)
    audio_preprocessor = AudioPreprocessor(
        sample_rate=config['dataset']['sample_rate'],
        use_mel_spectrogram=False,
        apply_spec_augment=False,
        apply_waveform_augment=False
    )
    audio_preprocessor.eval()
    
    text_preprocessor = TextPreprocessor(
        model_name=config['text']['model_name'],
        max_length=config['text']['max_length'],
        apply_text_augment=False
    )
    text_preprocessor.eval()
    
    # Select sessions
    if split == 'train':
        sessions = config['dataset']['train_sessions']
    elif split == 'val':
        sessions = config['dataset']['val_sessions']
    elif split == 'test':
        sessions = config['dataset']['test_sessions']
    else:  # 'all'
        sessions = (config['dataset']['train_sessions'] + 
                   config['dataset']['val_sessions'] + 
                   config['dataset']['test_sessions'])
    
    # Create dataset
    dataset = build_dataset(
        corpus=corpus_from_config(config),
        data_path=config['dataset']['data_path'],
        sessions=sessions,
        emotions=config['dataset']['emotions'],
        sample_rate=config['dataset']['sample_rate'],
        audio_preprocessor=audio_preprocessor,
        text_preprocessor=text_preprocessor,
        max_audio_length=config['audio'].get('max_length', 8.0),
        training=False,  # No augmentation during evaluation
        use_prosody_cues=config['text'].get('use_prosody_cues', False)
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['evaluation'].get('batch_size', 4),
        shuffle=False,
        num_workers=config['compute'].get('num_workers', 2),
        pin_memory=config['compute'].get('pin_memory', True),
        collate_fn=collate_fn
    )
    
    print(f"  Dataset size: {len(dataset)}")
    
    return dataloader, dataset


def get_predictions_and_logits(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get predictions and logits from model.
    
    Returns:
        predictions: (N,) array of predicted class indices
        logits: (N, num_classes) array of raw logits
    """
    model.eval()
    
    all_predictions = []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # Move to device
            audio = batch['audio'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            
            # Forward pass
            outputs = model(audio, text_input_ids, text_attention_mask)
            
            # Get predictions and logits
            predictions = outputs['predictions']
            logits = outputs['logits']
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_logits)


def ensemble_weighted_average(
    all_logits: List[np.ndarray],
    weights: List[float]
) -> np.ndarray:
    """
    Ensemble by weighted average of logits.
    
    Args:
        all_logits: List of (N, num_classes) logit arrays
        weights: List of weights for each model
        
    Returns:
        Ensemble predictions: (N,) array
    """
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Weighted average of logits
    ensemble_logits = np.zeros_like(all_logits[0])
    for logits, weight in zip(all_logits, weights):
        ensemble_logits += weight * logits
    
    # Get predictions from ensemble logits
    ensemble_predictions = np.argmax(ensemble_logits, axis=1)
    
    return ensemble_predictions


def ensemble_voting(
    all_predictions: List[np.ndarray],
    weights: List[float]
) -> np.ndarray:
    """
    Ensemble by weighted voting.
    
    Args:
        all_predictions: List of (N,) prediction arrays
        weights: List of weights for each model
        
    Returns:
        Ensemble predictions: (N,) array
    """
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    num_samples = len(all_predictions[0])
    num_classes = len(np.unique(np.concatenate(all_predictions)))
    
    # Weighted voting
    ensemble_predictions = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        votes = np.zeros(num_classes)
        for pred, weight in zip(all_predictions, weights):
            votes[pred[i]] += weight
        ensemble_predictions[i] = np.argmax(votes)
    
    return ensemble_predictions


def main():
    """Main ensemble evaluation function."""
    args = parse_args()
    
    # Validate inputs
    num_models = len(args.checkpoints)
    if len(args.configs) != num_models:
        raise ValueError(f"Number of configs ({len(args.configs)}) must match number of checkpoints ({num_models})")
    if len(args.study_numbers) != num_models:
        raise ValueError(f"Number of study numbers ({len(args.study_numbers)}) must match number of checkpoints ({num_models})")
    
    # Set default weights if not provided
    if args.weights is None:
        args.weights = [1.0 / num_models] * num_models
        print(f"\nUsing equal weights: {args.weights}")
    elif len(args.weights) != num_models:
        raise ValueError(f"Number of weights ({len(args.weights)}) must match number of checkpoints ({num_models})")
    else:
        # Normalize weights
        total_weight = sum(args.weights)
        args.weights = [w / total_weight for w in args.weights]
        print(f"\nUsing weights: {args.weights}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger('ensemble', str(output_dir))
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    logger.info(f"Evaluating ensemble on {device}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Models: {args.checkpoints}")
    logger.info(f"Weights: {args.weights}")
    
    # Load all models
    models = []
    configs = []
    for checkpoint, config_path, study_num in zip(args.checkpoints, args.configs, args.study_numbers):
        model, config = load_model_from_checkpoint(checkpoint, config_path, study_num, device)
        models.append(model)
        configs.append(config)
    
    # Use first config for dataloader (all should have same dataset structure)
    config = configs[0]
    
    # Create dataloader
    dataloader, dataset = create_dataloader(config, args.split)
    
    # Get predictions from all models
    logger.info("\nGetting predictions from all models...")
    all_predictions = []
    all_logits = []
    
    for i, model in enumerate(models):
        logger.info(f"Model {i+1}/{num_models}...")
        predictions, logits = get_predictions_and_logits(model, dataloader, device)
        all_predictions.append(predictions)
        all_logits.append(logits)
    
    # Ensemble predictions
    logger.info(f"\nEnsembling predictions using {args.method}...")
    if args.method == 'weighted_average':
        ensemble_predictions = ensemble_weighted_average(all_logits, args.weights)
    elif args.method == 'voting':
        ensemble_predictions = ensemble_voting(all_predictions, args.weights)
    else:
        raise ValueError(f"Unknown ensemble method: {args.method}")
    
    # Get labels
    all_labels = []
    all_utterance_ids = []
    all_emotions = []
    
    for batch in dataloader:
        all_labels.extend(batch['labels'].numpy())
        all_utterance_ids.extend(batch['utterance_ids'])
        all_emotions.extend(batch['emotions'])
    
    labels = np.array(all_labels)
    
    # Compute metrics
    emotion_labels = config['dataset']['emotions']
    metrics_tracker = EmotionMetrics(emotion_labels)
    metrics_tracker.update(ensemble_predictions, labels)
    
    # Print report
    print("\n" + "="*80)
    print("ENSEMBLE RESULTS")
    print("="*80)
    metrics_tracker.print_report()
    
    # Save confusion matrix
    cm_fig = metrics_tracker.plot_confusion_matrix(normalize=True)
    cm_path = output_dir / f'ensemble_confusion_matrix_{args.split}.png'
    cm_fig.savefig(cm_path, dpi=300, bbox_inches='tight')
    logger.info(f"Confusion matrix saved to: {cm_path}")
    
    # Save metrics
    metrics = metrics_tracker.compute()
    metrics_path = output_dir / f'ensemble_metrics_{args.split}.txt'
    with open(metrics_path, 'w') as f:
        f.write("ENSEMBLE EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Method: {args.method}\n")
        f.write(f"Models: {', '.join([Path(cp).name for cp in args.checkpoints])}\n")
        f.write(f"Weights: {args.weights}\n\n")
        f.write(metrics_tracker.get_classification_report())
        f.write("\n\nSummary:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    logger.info(f"Metrics saved to: {metrics_path}")
    
    # Save individual model results for comparison
    individual_results = {}
    for i, (predictions, checkpoint) in enumerate(zip(all_predictions, args.checkpoints)):
        individual_metrics = EmotionMetrics(emotion_labels)
        individual_metrics.update(predictions, labels)
        individual_results[Path(checkpoint).name] = individual_metrics.compute()
    
    comparison_path = output_dir / f'ensemble_comparison_{args.split}.json'
    with open(comparison_path, 'w') as f:
        json.dump({
            'ensemble': metrics,
            'individual_models': individual_results,
            'weights': args.weights,
            'method': args.method
        }, f, indent=2)
    logger.info(f"Comparison saved to: {comparison_path}")
    
    print("\n" + "="*80)
    print("Individual Model Results (for comparison):")
    print("="*80)
    for name, individual_metrics in individual_results.items():
        print(f"\n{name}:")
        print(f"  Unweighted Accuracy: {individual_metrics['unweighted_accuracy']:.4f}")
        print(f"  Weighted Accuracy: {individual_metrics['weighted_accuracy']:.4f}")
        print(f"  F1 Macro: {individual_metrics['f1_macro']:.4f}")
    
    print("\n" + "="*80)
    print(f"Ensemble Improvement:")
    print("="*80)
    best_individual = max([m['unweighted_accuracy'] for m in individual_results.values()])
    ensemble_ua = metrics['unweighted_accuracy']
    improvement = ensemble_ua - best_individual
    print(f"Best Individual: {best_individual:.4f}")
    print(f"Ensemble: {ensemble_ua:.4f}")
    print(f"Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
    
    return metrics


if __name__ == '__main__':
    main()






