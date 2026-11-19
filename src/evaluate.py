"""Evaluation script for Conformer multimodal SER.

Loads a checkpoint and config, runs inference on a specified split (train/val/test),
and reports metrics (e.g. WA, UA, per-class accuracy). Optionally saves predictions to CSV.
"""

import os
import sys
import yaml
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (IEMOCAPDataset, AudioPreprocessor, TextPreprocessor, collate_fn,
                      build_dataset, corpus_from_config)
from src.models import ConformerMultimodalModel
from src.utils.metrics import EmotionMetrics
from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Conformer multimodal SER model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test', 'all'],
                        help='Which split to evaluate')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions to CSV file')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Nested dict with model and dataset configuration.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path: str, config: dict, device: str):
    """Load model from checkpoint and move to device.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        config: Model/dataset config dict (e.g. from load_config).
        device: Target device string (e.g. 'cuda' or 'cpu').

    Returns:
        Model loaded with checkpoint weights, on device, in eval mode.
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # Create model
    model = ConformerMultimodalModel(
        audio_encoder_name=config['model']['audio_encoder'],
        text_encoder_name=config['model']['text_encoder'],
        num_classes=config['model']['num_classes'],
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout'],
        fusion_type=config['model']['fusion_type']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")
    
    return model


def create_dataloader(config: dict, split: str):
    """Create dataloader for evaluation."""
    print(f"Creating {split} dataloader...")
    
    # Initialize preprocessors
    audio_preprocessor = AudioPreprocessor(
        sample_rate=config['dataset']['sample_rate'],
        use_mel_spectrogram=False,
        apply_spec_augment=False
    )
    audio_preprocessor.eval()
    
    text_preprocessor = TextPreprocessor(
        model_name=config['text']['model_name'],
        max_length=config['text']['max_length']
    )
    
    # Select sessions
    if split == 'train':
        sessions = config['dataset']['train_sessions']
    elif split == 'val':
        sessions = config['dataset']['val_sessions']
    elif split == 'test':
        sessions = config['dataset']['test_sessions']
    elif split == 'all':
        sessions = (config['dataset']['train_sessions'] +
                   config['dataset']['val_sessions'] +
                   config['dataset']['test_sessions'])
    else:
        raise ValueError(f"Unknown split: {split}")
    
    # Create dataset
    dataset = build_dataset(
        corpus=corpus_from_config(config),
        data_path=config['dataset']['data_path'],
        sessions=sessions,
        emotions=config['dataset']['emotions'],
        sample_rate=config['dataset']['sample_rate'],
        audio_preprocessor=audio_preprocessor,
        text_preprocessor=text_preprocessor,
        max_audio_length=config['audio']['max_length'],
        training=False  # Use deterministic start for evaluation
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['compute']['num_workers'],
        pin_memory=config['compute']['pin_memory'],
        collate_fn=collate_fn
    )
    
    print(f"{split.capitalize()} samples: {len(dataset)}")
    
    return dataloader, dataset


def evaluate(model, dataloader, device, config):
    """
    Evaluate model on dataloader.
    
    Returns:
        predictions, labels, utterance_ids, emotions
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_utterance_ids = []
    all_emotions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            audio = batch['audio'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(audio, text_input_ids, text_attention_mask)
            predictions = outputs['predictions']
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_utterance_ids.extend(batch['utterance_ids'])
            all_emotions.extend(batch['emotions'])
    
    return (np.array(all_predictions),
            np.array(all_labels),
            all_utterance_ids,
            all_emotions)


def analyze_errors(
    predictions,
    labels,
    utterance_ids,
    emotions,
    emotion_labels,
    output_dir
):
    """Analyze and save error cases."""
    print("\nAnalyzing errors...")
    
    # Find misclassified samples
    errors = predictions != labels
    error_indices = np.where(errors)[0]
    
    print(f"Total errors: {len(error_indices)} / {len(predictions)} ({len(error_indices)/len(predictions)*100:.2f}%)")
    
    # Create error dataframe
    error_data = []
    for idx in error_indices:
        error_data.append({
            'utterance_id': utterance_ids[idx],
            'true_emotion': emotion_labels[labels[idx]],
            'predicted_emotion': emotion_labels[predictions[idx]],
            'true_label': labels[idx],
            'predicted_label': predictions[idx]
        })
    
    error_df = pd.DataFrame(error_data)
    
    # Save errors
    error_path = Path(output_dir) / 'errors.csv'
    error_df.to_csv(error_path, index=False)
    print(f"Errors saved to: {error_path}")
    
    # Analyze error patterns
    print("\nMost common confusion pairs:")
    error_pairs = error_df.groupby(['true_emotion', 'predicted_emotion']).size()
    error_pairs = error_pairs.sort_values(ascending=False)
    print(error_pairs.head(10))
    
    # Per-class error rates
    print("\nPer-class error rates:")
    for i, emotion in enumerate(emotion_labels):
        emotion_mask = labels == i
        if emotion_mask.sum() > 0:
            error_rate = errors[emotion_mask].mean()
            print(f"  {emotion:12s}: {error_rate*100:.2f}%")
    
    return error_df


def save_predictions(
    predictions,
    labels,
    utterance_ids,
    emotions,
    emotion_labels,
    output_dir
):
    """Save all predictions to CSV."""
    print("\nSaving predictions...")
    
    # Create dataframe
    data = []
    for i in range(len(predictions)):
        data.append({
            'utterance_id': utterance_ids[i],
            'true_emotion': emotions[i],
            'predicted_emotion': emotion_labels[predictions[i]],
            'true_label': labels[i],
            'predicted_label': predictions[i],
            'correct': predictions[i] == labels[i]
        })
    
    df = pd.DataFrame(data)
    
    # Save
    pred_path = Path(output_dir) / 'predictions.csv'
    df.to_csv(pred_path, index=False)
    print(f"Predictions saved to: {pred_path}")
    
    return df


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger('evaluate', str(output_dir))
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    logger.info(f"Evaluating on {device}")
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    
    # Create dataloader
    dataloader, dataset = create_dataloader(config, args.split)
    
    # Evaluate
    logger.info(f"Evaluating {args.split} split...")
    predictions, labels, utterance_ids, emotions = evaluate(
        model, dataloader, device, config
    )
    
    # Compute metrics
    emotion_labels = config['dataset']['emotions']
    metrics_tracker = EmotionMetrics(emotion_labels)
    metrics_tracker.update(predictions, labels)
    
    # Print report
    metrics_tracker.print_report()
    
    # Save confusion matrix
    cm_fig = metrics_tracker.plot_confusion_matrix(normalize=True)
    cm_path = output_dir / f'confusion_matrix_{args.split}.png'
    cm_fig.savefig(cm_path, dpi=300, bbox_inches='tight')
    logger.info(f"Confusion matrix saved to: {cm_path}")
    
    # Save metrics
    metrics = metrics_tracker.compute()
    metrics_path = output_dir / f'metrics_{args.split}.txt'
    with open(metrics_path, 'w') as f:
        f.write(metrics_tracker.get_classification_report())
        f.write("\n\nSummary:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    logger.info(f"Metrics saved to: {metrics_path}")
    
    # Analyze errors
    error_df = analyze_errors(
        predictions, labels, utterance_ids, emotions,
        emotion_labels, output_dir
    )
    
    # Save predictions if requested
    if args.save_predictions:
        pred_df = save_predictions(
            predictions, labels, utterance_ids, emotions,
            emotion_labels, output_dir
        )
    
    logger.info("\nEvaluation completed!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()

