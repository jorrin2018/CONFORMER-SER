"""
Master script to run all ablation studies.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import IEMOCAPDataset, AudioPreprocessor, TextPreprocessor, collate_fn
from src.ablation_studies import (
    load_base_config,
    Ablation01Model,
    Ablation02Model,
    Ablation03Model,
    Ablation04Model,
    create_ablation_optimizer
)
from src.utils.metrics import EmotionMetrics, compute_class_weights, plot_training_curves
from src.utils.logger import setup_logger, TensorBoardLogger, ExperimentTracker, print_model_summary
from src.utils.losses import create_loss_function
from src.train import (
    create_dataloaders,
    train_epoch,
    validate,
    create_optimizer_and_scheduler,
    detect_class_collapse
)


def load_ablation_config(config_path: str, base_config: dict) -> dict:
    """
    Load ablation-specific config and merge with base config.
    
    Args:
        config_path: Path to ablation YAML config
        base_config: Base configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    # Load ablation config
    with open(config_path, 'r') as f:
        ablation_config = yaml.safe_load(f)
    
    # Deep merge: ablation config overrides base config
    merged_config = base_config.copy()
    
    # Merge nested dictionaries
    for key, value in ablation_config.items():
        if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
            merged_config[key].update(value)
        else:
            merged_config[key] = value
    
    return merged_config


def create_ablation_model(study_number: int, config: dict, device: str, logger):
    """
    Create model for specific ablation study.
    
    Args:
        study_number: Ablation study number (1-4)
        config: Configuration dictionary
        device: Device to use
        logger: Logger instance
        
    Returns:
        Model instance
    """
    if study_number == 1:
        model = Ablation01Model(config)
    elif study_number == 2:
        model = Ablation02Model(config)
    elif study_number == 3:
        model = Ablation03Model(config)
    elif study_number == 4:
        model = Ablation04Model(config)
    else:
        raise ValueError(f"Invalid study number: {study_number}. Must be 1-4.")
    
    model = model.to(device)
    print_model_summary(model, logger)
    
    return model


def run_ablation_study(
    study_number: int,
    config_path: str,
    device: str = 'cuda',
    experiment_name: str = None
):
    """
    Run a single ablation study.
    
    Args:
        study_number: Ablation study number (1-4)
        config_path: Path to ablation config YAML
        device: Device to use
        experiment_name: Optional experiment name
    """
    print(f"\n{'='*60}")
    print(f"Starting Ablation Study {study_number}")
    print(f"{'='*60}\n")
    
    # Load base config
    base_config = load_base_config()
    
    # Load and merge ablation config
    config = load_ablation_config(config_path, base_config)
    
    # Override device if specified
    config['compute']['device'] = device
    device = config['compute']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if device == 'cuda':
        torch.cuda.manual_seed(config['seed'])
    
    # Generate experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"ablation_{study_number:02d}_{timestamp}"
    
    # Setup logging
    log_dir = config['logging']['log_dir']
    logger = setup_logger(f'ablation_{study_number}', log_dir)
    logger.info(f"Starting Ablation Study {study_number} on {device}")
    logger.info(f"Configuration: {config}")
    
    # Setup TensorBoard
    if config['logging']['use_tensorboard']:
        tb_logger = TensorBoardLogger(
            log_dir,
            experiment_name=experiment_name
        )
    else:
        tb_logger = None
    
    # Setup experiment tracker
    checkpoint_dir = config['logging']['checkpoint_dir']
    exp_tracker = ExperimentTracker(
        checkpoint_dir,
        experiment_name=experiment_name
    )
    exp_tracker.log_config(config)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        config, logger, audio_only=False
    )
    
    # Create model
    model = create_ablation_model(study_number, config, device, logger)
    
    # Create loss function
    loss_config = config['training'].get('loss', {})
    loss_type = loss_config.get('type', 'cross_entropy')
    gamma = loss_config.get('gamma', 2.0)
    label_smoothing = loss_config.get('label_smoothing', 0.0)
    
    # Use normalized and capped class weights as alpha for FocalLoss
    if loss_type == 'focal':
        num_classes = len(config['dataset']['emotions'])
        with torch.no_grad():
            norm_weights = class_weights / (class_weights.sum() + 1e-9) * num_classes
            norm_weights = torch.clamp(norm_weights, min=0.5, max=5.0)
        alpha = norm_weights
        logger.info(f"FocalLoss alpha (normalized & capped): {alpha.tolist()}")
    else:
        alpha = None
    
    criterion = create_loss_function(
        loss_type=loss_type,
        class_weights=class_weights if loss_type == 'cross_entropy' else None,
        alpha=alpha,
        gamma=gamma,
        label_smoothing=label_smoothing,
        device=device
    )
    
    # Create optimizer using ablation-specific function
    optimizer = create_ablation_optimizer(model, config, study_number)
    
    # Create scheduler (reuse logic from train.py)
    num_training_steps = len(train_loader) * config['training']['num_epochs']
    scheduler_type = config['training'].get('scheduler_type', 'cosine')
    
    if scheduler_type == 'linear':
        from transformers import get_linear_schedule_with_warmup
        num_warmup_steps = int(config['training'].get('warmup_ratio', 0.1) * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type == 'cosine':
        from transformers import get_cosine_schedule_with_warmup
        num_warmup_steps = int(config['training'].get('warmup_ratio', 0.1) * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type == 'plateau':
        scheduler_metric = config['training'].get('scheduler_metric', 'val_loss')
        scheduler_mode = config['training'].get('scheduler_mode', 'min')
        scheduler_factor = config['training'].get('scheduler_factor', 0.5)
        scheduler_patience = config['training'].get('scheduler_patience', 5)
        scheduler_threshold = float(config['training'].get('scheduler_threshold', 1e-4))
        scheduler_min_lr = config['training'].get('scheduler_min_lr', 1e-7)
        scheduler_cooldown = config['training'].get('scheduler_cooldown', 2)
        scheduler_verbose = config['training'].get('scheduler_verbose', True)
        
        # Calculate min_lr for each group
        min_lr_list = []
        for param_group in optimizer.param_groups:
            initial_lr = param_group['lr']
            param_group_min_lr = max(initial_lr * 0.01, scheduler_min_lr)
            min_lr_list.append(param_group_min_lr)
        
        min_lr = min_lr_list[0] if len(set(min_lr_list)) == 1 else min_lr_list
        
        scheduler_params = {
            'mode': scheduler_mode,
            'factor': scheduler_factor,
            'patience': scheduler_patience,
            'threshold': scheduler_threshold,
            'min_lr': min_lr,
            'cooldown': scheduler_cooldown
        }
        
        import inspect
        sig = inspect.signature(ReduceLROnPlateau.__init__)
        if 'verbose' in sig.parameters:
            scheduler_params['verbose'] = scheduler_verbose
        
        scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)
    else:
        scheduler = None
    
    # Initialize mixed precision scaler
    scaler = None
    if config['training']['mixed_precision']:
        if config['compute']['device'] == 'cuda':
            try:
                from torch.amp import GradScaler
                scaler = GradScaler('cuda')
            except (ImportError, TypeError):
                from torch.cuda.amp import GradScaler
                scaler = GradScaler()
        else:
            scaler = None
    
    # Training loop (reuse from train.py)
    best_val_ua = 0.0
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    train_metrics_list = []
    val_metrics_list = []
    
    logger.info("Starting training...")
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, config, epoch, logger, tb_logger, audio_only=False
        )
        
        train_losses.append(train_loss)
        train_metrics_list.append(train_metrics['unweighted_accuracy'])
        
        # Validate
        if epoch % config['logging']['val_every_n_epochs'] == 0:
            val_loss, val_metrics, val_metrics_tracker = validate(
                model, val_loader, criterion, device, config, logger, audio_only=False
            )
            
            val_losses.append(val_loss)
            val_metrics_list.append(val_metrics['unweighted_accuracy'])
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Train UA: {train_metrics['unweighted_accuracy']:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val UA: {val_metrics['unweighted_accuracy']:.4f}")
            
            # Log to TensorBoard
            if tb_logger:
                tb_logger.log_metrics(train_metrics, epoch, prefix='train/')
                tb_logger.log_metrics(val_metrics, epoch, prefix='val/')
                tb_logger.log_scalar('epoch/train_loss', train_loss, epoch)
                tb_logger.log_scalar('epoch/val_loss', val_loss, epoch)
            
            # Log to experiment tracker
            exp_tracker.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }, epoch)
            
            # Check if best model
            if val_metrics['unweighted_accuracy'] > best_val_ua:
                best_val_ua = val_metrics['unweighted_accuracy']
                logger.info(f"New best model! UA: {best_val_ua:.4f}")
                
                # Save best model
                best_checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
                os.makedirs(best_checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(best_checkpoint_dir, 'best_model.pt')
                
                checkpoint_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics,
                    'config': config
                }
                if scheduler is not None:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        # ReduceLROnPlateau doesn't have state_dict
                        checkpoint_data['scheduler_state'] = {
                            'best': getattr(scheduler, 'best', None),
                            'num_bad_epochs': getattr(scheduler, 'num_bad_epochs', None)
                        }
                    else:
                        checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                
                torch.save(checkpoint_data, checkpoint_path)
                
                exp_tracker.update_best_metrics(val_metrics)
                exp_tracker.save_checkpoint_info(checkpoint_path, epoch, val_metrics)
                
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"Patience: {patience_counter}/{config['training']['early_stopping_patience']}")
            
            # Step scheduler
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler_metric = config['training'].get('scheduler_metric', 'val_loss')
                    if scheduler_metric == 'val_loss':
                        metric_value = val_loss
                    elif scheduler_metric == 'val_unweighted_accuracy':
                        metric_value = val_metrics['unweighted_accuracy']
                    else:
                        metric_value = val_loss
                    scheduler.step(metric_value)
                else:
                    scheduler.step()
            
            # Early stopping
            if patience_counter >= config['training']['early_stopping_patience']:
                logger.info("Early stopping triggered!")
                break
    
    # Plot training curves
    if len(train_losses) > 0:
        fig = plot_training_curves(
            train_losses, val_losses,
            train_metrics_list, val_metrics_list,
            metric_name='Unweighted Accuracy'
        )
        if tb_logger:
            tb_logger.log_figure('training/curves', fig, epoch)
        fig.savefig(Path(checkpoint_dir) / experiment_name / 'training_curves.png')
    
    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    test_loss, test_metrics, test_metrics_tracker = validate(
        model, test_loader, criterion, device, config, logger, audio_only=False
    )
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    test_metrics_tracker.print_report()
    
    # Save confusion matrix
    cm_fig = test_metrics_tracker.plot_confusion_matrix(normalize=True)
    cm_fig.savefig(Path(checkpoint_dir) / experiment_name / 'confusion_matrix_test.png')
    
    if tb_logger:
        tb_logger.log_metrics(test_metrics, 0, prefix='test/')
        tb_logger.log_figure('test/confusion_matrix', cm_fig, 0)
    
    # Save final results
    results_path = os.path.join(checkpoint_dir, experiment_name, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'test_metrics': test_metrics,
            'best_val_ua': best_val_ua,
            'final_epoch': epoch
        }, f, indent=2)
    
    logger.info(f"Ablation Study {study_number} completed!")
    logger.info(f"Results saved to: {os.path.join(checkpoint_dir, experiment_name)}")
    
    if tb_logger:
        tb_logger.close()


def main():
    """Main function to run all ablation studies."""
    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument('--study', type=int, default=None,
                        help='Run specific study (1-4). If not specified, runs all.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name prefix')
    args = parser.parse_args()
    
    # Define all studies
    studies = [
        (1, "config/ablation/ablation_01_unfreeze_encoders.yaml"),
        (2, "config/ablation/ablation_02_lstm_head.yaml"),
        (3, "config/ablation/ablation_03_transformer_head.yaml"),
        (4, "config/ablation/ablation_04_cnn1d_head.yaml")
    ]
    
    # Filter studies if specific one requested
    if args.study is not None:
        studies = [s for s in studies if s[0] == args.study]
        if not studies:
            print(f"Error: Study {args.study} not found. Valid studies: 1-4")
            return
    
    # Run each study
    for study_number, config_path in studies:
        try:
            experiment_name = f"{args.experiment_name}_study_{study_number}" if args.experiment_name else None
            run_ablation_study(
                study_number=study_number,
                config_path=config_path,
                device=args.device,
                experiment_name=experiment_name
            )
        except Exception as e:
            print(f"Error running study {study_number}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("All ablation studies completed!")
    print("="*60)


if __name__ == '__main__':
    main()

