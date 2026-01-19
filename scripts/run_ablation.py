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
import copy
import gc

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import IEMOCAPDataset, AudioPreprocessor, TextPreprocessor, collate_fn
from src.ablation_studies import (
    load_base_config,
    Ablation01Model,
    Ablation02Model,
    Ablation03Model,
    Ablation04Model,
    Ablation05Model,
    Ablation06Model,
    Ablation07Model,
    Ablation08Model,
    Ablation09Model,
    Ablation10Model,
    Ablation11Model,
    Ablation13Model,
    Ablation21Model,
    Ablation22Model,
    Ablation23Model,
    Ablation24Model,
    Ablation25Model,
    Ablation26Model,
    Ablation27Model,
    Ablation28Model,
    Ablation31Model,
    Ablation32Model,
    Ablation33Model,
    Ablation38Model,
    Ablation34Model,
    Ablation35Model,
    Ablation36AModel,
    Ablation36BModel,
    Ablation36CModel,
    create_ablation_optimizer
)
from src.ablation_studies.ablation_30_dirichlet import Ablation30Model
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
    # Load ablation config (force UTF-8 to avoid Windows cp1252 issues)
    with open(config_path, 'r', encoding='utf-8') as f:
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
        study_number: Ablation study number (1-9)
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
    elif study_number == 5:
        model = Ablation05Model(config)
    elif study_number == 6:
        model = Ablation06Model(config)
    elif study_number == 7:
        model = Ablation07Model(config)
    elif study_number == 8:
        model = Ablation08Model(config)
    elif study_number == 9:
        model = Ablation09Model(config)
    elif study_number == 10:
        model = Ablation10Model(config)
    elif study_number == 11:
        model = Ablation11Model(config)
    elif study_number == 12:
        # Ablation 12 reuses the Ablation 10 architecture (SpeechCue-style prompts)
        model = Ablation10Model(config)
    elif study_number == 13:
        # Ablation 13: Ablation 11 + SpeechCue + ViT parcialmente descongelado
        model = Ablation13Model(config)
    elif study_number in (14, 15, 16, 17, 18, 19, 20):
        # Ablation 14 (13a), 15 (13b), 16 (13c), 17 (13d), 18 (13e), 19 (13f), 20 (13d kfold): Variantes de Ablation 13
        model = Ablation13Model(config)
    elif study_number == 21:
        model = Ablation21Model(config)
    elif study_number == 22:
        model = Ablation22Model(config)
    elif study_number == 23:
        model = Ablation23Model(config)
    elif study_number == 24:
        model = Ablation24Model(config)
    elif study_number == 25:
        model = Ablation25Model(config)
    elif study_number == 26:
        model = Ablation26Model(config)
    elif study_number == 27:
        model = Ablation27Model(config)
    elif study_number == 28:
        model = Ablation28Model(config)
    elif study_number == 29:
        # Ablation 29: Ablation 25 with K-Fold
        model = Ablation25Model(config)
    elif study_number == 30:
        model = Ablation30Model(config)
    elif study_number == 31:
        model = Ablation31Model(config)
    elif study_number == 32:
        model = Ablation32Model(config)
    elif study_number == 33:
        model = Ablation33Model(config)
    elif study_number == 34:
        model = Ablation34Model(config)
    elif study_number == 35:
        model = Ablation35Model(config)
    elif study_number == 36:  # 36A
        model = Ablation36AModel(config)
    elif study_number == 37:  # 37: Dirichlet + Conflict-Aware (similar to 36A but with different config)
        model = Ablation36AModel(config)
    elif study_number == 38:  # Ablation 38: OpenSMILE + Attention Gates + Residual
        model = Ablation38Model(config)
    elif study_number == 39:  # 36C
        model = Ablation36CModel(config)
    else:
        raise ValueError(f"Invalid study number: {study_number}. Must be 1-39.")
    
    model = model.to(device)
    print_model_summary(model, logger)
    
    return model


def _run_ablation_with_config(
    study_number: int,
    config: dict,
    device: str,
    experiment_name: str
):
    """
    Core training/evaluation loop shared by standard and LOSO ablations.
    Returns a small dict with key metrics for aggregation.
    """
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
    
    # Apply optimizations for RTX 4060 (8GB VRAM)
    model_config = config.get('model', {})
    
    # 1. Gradient Checkpointing (saves ~60% VRAM, costs ~20% speed)
    if model_config.get('gradient_checkpointing', False):
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            # Try to enable for encoder modules
            for module in model.modules():
                if hasattr(module, 'gradient_checkpointing_enable'):
                    module.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for encoder modules")
    
    # 2. torch.compile for kernel fusion (PyTorch 2.0+)
    # Note: Requires Triton for optimal performance. Install with: pip install triton
    # See: https://triton-lang.org/main/getting-started/installation.html
    if model_config.get('use_torch_compile', False) and hasattr(torch, 'compile'):
        compile_mode = model_config.get('compile_mode', 'reduce-overhead')
        
        # Check if Triton is available before attempting compilation
        triton_available = False
        try:
            import triton
            # Try a simple operation to verify Triton works
            triton_available = True
            logger.info("Triton is available for torch.compile")
        except ImportError:
            logger.warning("Triton not installed. torch.compile requires Triton for optimal performance.")
            logger.info("Install with: pip install triton")
            logger.info("Documentation: https://triton-lang.org/main/getting-started/installation.html")
            logger.info("Continuing without model compilation. Performance may be slightly lower.")
        except Exception as e:
            logger.warning(f"Triton import failed: {e}")
            logger.info("Continuing without model compilation.")
        
        if triton_available:
            try:
                # Try to compile the model
                # Use a more conservative mode if reduce-overhead fails
                try:
                    model = torch.compile(model, mode=compile_mode)
                    logger.info(f"Model compiled with mode: {compile_mode}")
                except Exception as compile_error:
                    # Try default mode if reduce-overhead fails
                    if compile_mode != "default":
                        logger.warning(f"Compilation with mode '{compile_mode}' failed: {compile_error}")
                        logger.info("Trying with default mode...")
                        try:
                            model = torch.compile(model, mode="default")
                            logger.info("Model compiled with default mode")
                        except Exception as default_error:
                            logger.warning(f"torch.compile failed even with default mode: {default_error}")
                            logger.info("Continuing without model compilation.")
                    else:
                        raise compile_error
            except Exception as e:
                error_msg = str(e)
                if 'TritonMissing' in error_msg or 'triton' in error_msg.lower():
                    logger.warning("Triton is not working properly. Install/update with: pip install triton")
                    logger.info("Documentation: https://triton-lang.org/main/getting-started/installation.html")
                else:
                    logger.warning(f"torch.compile failed: {error_msg}")
                logger.info("Continuing without model compilation.")
                # Model remains uncompiled, which is fine
        else:
            # Skip compilation if Triton is not available
            logger.info("Skipping torch.compile (Triton not available). Model will run normally.")
    
    # 3. Enable FlashAttention if available and configured
    if model_config.get('use_flash_attention', False):
        try:
            import xformers
            logger.info("xformers available - FlashAttention will be used if supported by model")
        except ImportError:
            logger.warning("xformers not available. Install with: pip install xformers")
    
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
        
        # For Ablation 36B (study 38), increase neutral weight by 2.0x
        if study_number == 38:  # Ablation 36B (Focal Loss)
            neutral_idx = config['dataset']['emotions'].index('neutral') if 'neutral' in config['dataset']['emotions'] else 3
            norm_weights[neutral_idx] = norm_weights[neutral_idx] * 2.0
            norm_weights = torch.clamp(norm_weights, min=0.5, max=10.0)  # Allow higher max for neutral
            logger.info(f"FocalLoss alpha (with 2.0x neutral weight): {norm_weights.tolist()}")
        
        alpha = norm_weights
        logger.info(f"FocalLoss alpha (normalized & capped): {alpha.tolist()}")
    else:
        alpha = None
    
    # For Dirichlet losses, need to pass num_classes
    num_classes = len(config['dataset']['emotions'])
    
    criterion = create_loss_function(
        loss_type=loss_type,
        class_weights=class_weights if loss_type == 'cross_entropy' else None,
        alpha=alpha,
        gamma=gamma,
        label_smoothing=label_smoothing,
        device=device,
        num_classes=num_classes
    )
    
    # For Dirichlet losses, need to update epoch in loss function
    if loss_type in ['dirichlet', 'dirichlet_neutral'] and hasattr(criterion, 'set_epoch'):
        # Will be updated during training loop
        pass
    
    # Create optimizer using ablation-specific function
    # Support for 8-bit optimizer (bitsandbytes) for RTX 4060
    training_config = config.get('training', {})
    use_8bit_optimizer = training_config.get('use_8bit_optimizer', False)
    
    if use_8bit_optimizer:
        try:
            import bitsandbytes as bnb
            logger.info("Using 8-bit optimizer (bitsandbytes) to reduce VRAM usage by ~75%")
            
            # Get parameter groups from standard optimizer creation
            training_config_temp = config.get('training', {})
            model_config = config.get('model', {})
            lr_audio = float(training_config_temp.get('learning_rate_audio', 1e-6))
            lr_text = float(training_config_temp.get('learning_rate_text', 2e-5))
            lr_fusion = float(training_config_temp.get('learning_rate_fusion', 5e-5))
            weight_decay = training_config_temp.get('weight_decay', 0.05)
            adam_epsilon = float(training_config_temp.get('adam_epsilon', 1e-8))
            
            # Create parameter groups
            audio_params = []
            text_params = []
            fusion_params = []
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'audio_encoder' in name:
                    audio_params.append(param)
                elif 'text_encoder' in name:
                    text_params.append(param)
                else:
                    fusion_params.append(param)
            
            param_groups = []
            if audio_params:
                param_groups.append({
                    'params': audio_params,
                    'lr': lr_audio,
                    'weight_decay': weight_decay
                })
            if text_params:
                param_groups.append({
                    'params': text_params,
                    'lr': lr_text,
                    'weight_decay': weight_decay
                })
            if fusion_params:
                param_groups.append({
                    'params': fusion_params,
                    'lr': lr_fusion,
                    'weight_decay': weight_decay
                })
            
            # Use 8-bit AdamW
            optimizer = bnb.optim.AdamW8bit(
                param_groups,
                eps=adam_epsilon
            )
        except ImportError:
            logger.warning("bitsandbytes not available. Install with: pip install bitsandbytes")
            logger.info("Falling back to standard optimizer")
            optimizer = create_ablation_optimizer(model, config, study_number)
    else:
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
    
    # Handle mixed precision: "bf16", "fp16", "fp8", or False
    mixed_precision = config['training'].get('mixed_precision', False)
    use_loss_scaling = config['training'].get('use_loss_scaling', True)  # Default True for FP16, False for BF16
    
    # Initialize scaler only if needed (FP16 requires it, BF16 doesn't)
    scaler = None
    if mixed_precision in ['fp16', 'fp8']:
        if config['compute']['device'] == 'cuda':
            try:
                from torch.amp import GradScaler
                scaler = GradScaler('cuda')
            except (ImportError, TypeError):
                from torch.cuda.amp import GradScaler
                scaler = GradScaler()
        else:
            scaler = None
    elif mixed_precision == 'bf16':
        # BF16 doesn't need loss scaling (same dynamic range as FP32)
        use_loss_scaling = False
        scaler = None
        logger.info("Using BFloat16 mixed precision (no loss scaling needed)")
    
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
        
        # Update epoch in Dirichlet loss function if applicable
        if hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(epoch)
        
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
        tb_logger.close()
    
    # Save final results for this run
    results_dir = Path(checkpoint_dir) / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'final_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'test_metrics': test_metrics,
            'best_val_ua': best_val_ua,
            'final_epoch': epoch
        }, f, indent=2)
    
    logger.info(f"Ablation Study {study_number} run completed!")
    logger.info(f"Results saved to: {results_dir}")
    
    # Return key metrics for potential aggregation (e.g., LOSO)
    test_ua = test_metrics.get('unweighted_accuracy', None)
    test_wa = test_metrics.get('weighted_accuracy', None)
    return {
        'best_val_ua': float(best_val_ua),
        'test_unweighted_accuracy': float(test_ua) if test_ua is not None else None,
        'test_weighted_accuracy': float(test_wa) if test_wa is not None else None,
        'final_epoch': int(epoch),
        'results_dir': str(results_dir)
    }


def run_ablation_study(
    study_number: int,
    config_path: str,
    device: str = 'cuda',
    experiment_name: str = None
):
    """
    Run a single ablation study (standard split).
    """
    print(f"\n{'='*60}")
    print(f"Starting Ablation Study {study_number}")
    print(f"{'='*60}\n")
    
    # Load base config
    base_config = load_base_config()
    
    # Load and merge ablation config
    config = load_ablation_config(config_path, base_config)
    
    # Generate experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"ablation_{study_number:02d}_{timestamp}"
    
    _run_ablation_with_config(study_number, config, device, experiment_name)


def run_ablation_study_loso(
    study_number: int,
    config_path: str,
    device: str = 'cuda',
    experiment_name: str = None
):
    """
    Run Ablation Study with 5-fold leave-one-session-out (LOSO) cross-validation.
    
    Specifically used for Ablation 12, which reuses the Ablation 10 architecture
    but cycles the IEMOCAP sessions so that in each fold one session is used as
    test set and the remaining sessions are split into train/val.
    """
    print(f"\n{'='*60}")
    print(f"Starting Ablation Study {study_number} with 5-fold LOSO")
    print(f"{'='*60}\n")
    
    # Load base + ablation config
    base_config = load_base_config()
    base_ablation_config = load_ablation_config(config_path, base_config)
    
    dataset_cfg = base_ablation_config.get('dataset', {})
    all_sessions = dataset_cfg.get('all_sessions', None)
    if all_sessions is None:
        # Fallback: infer from train/val/test if all_sessions not provided
        combined = []
        combined.extend(dataset_cfg.get('train_sessions', []))
        combined.extend(dataset_cfg.get('val_sessions', []))
        combined.extend(dataset_cfg.get('test_sessions', []))
        all_sessions = sorted(list(set(combined)))
    
    if len(all_sessions) < 2:
        raise ValueError("LOSO requires at least 2 sessions in 'dataset.all_sessions'.")
    
    # Base experiment name
    if experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_experiment_name = f"ablation_{study_number:02d}_loso_{timestamp}"
    else:
        base_experiment_name = experiment_name
    
    fold_results = []
    
    for fold_idx, test_session in enumerate(all_sessions):
        # Choose validation session as the next available session (cyclic),
        # different from the test session.
        remaining = [s for s in all_sessions if s != test_session]
        val_session = remaining[0]
        train_sessions = [s for s in all_sessions if s not in (test_session, val_session)]
        
        fold_config = copy.deepcopy(base_ablation_config)
        fold_config['dataset']['train_sessions'] = train_sessions
        fold_config['dataset']['val_sessions'] = [val_session]
        fold_config['dataset']['test_sessions'] = [test_session]
        
        # Create per-fold logging/checkpoint dirs to avoid collisions
        fold_suffix = f"fold_S{test_session}_test"
        fold_log_dir = os.path.join(fold_config['logging']['log_dir'], fold_suffix)
        fold_ckpt_dir = os.path.join(fold_config['logging']['checkpoint_dir'], fold_suffix)
        fold_config['logging']['log_dir'] = fold_log_dir
        fold_config['logging']['checkpoint_dir'] = fold_ckpt_dir
        
        fold_experiment_name = f"{base_experiment_name}_{fold_suffix}"
        
        print(f"\n--- LOSO Fold {fold_idx + 1}/{len(all_sessions)} ---")
        print(f"Train sessions: {train_sessions}")
        print(f"Val session:    {[val_session]}")
        print(f"Test session:   {[test_session]}")
        print(f"Experiment:     {fold_experiment_name}")
        
        fold_metrics = _run_ablation_with_config(
            study_number=study_number,
            config=fold_config,
            device=device,
            experiment_name=fold_experiment_name
        )
        fold_metrics['test_session'] = int(test_session)
        fold_metrics['val_session'] = int(val_session)
        fold_metrics['train_sessions'] = [int(s) for s in train_sessions]
        fold_results.append(fold_metrics)
    
    # Aggregate LOSO results across folds
    # We aggregate best_val_ua and test_unweighted_accuracy if present.
    best_val_uas = [fr['best_val_ua'] for fr in fold_results if fr.get('best_val_ua') is not None]
    test_uas = [fr['test_unweighted_accuracy'] for fr in fold_results if fr.get('test_unweighted_accuracy') is not None]
    
    summary = {
        'folds': fold_results,
        'num_folds': len(fold_results),
        'all_sessions': [int(s) for s in all_sessions],
    }
    if best_val_uas:
        summary['mean_best_val_ua'] = float(sum(best_val_uas) / len(best_val_uas))
    if test_uas:
        summary['mean_test_unweighted_accuracy'] = float(sum(test_uas) / len(test_uas))
    
    # Save LOSO summary next to the base checkpoint directory
    ckpt_root = base_ablation_config['logging']['checkpoint_dir']
    summary_path = Path(ckpt_root) / 'loso_summary_ablation_{:02d}.json'.format(study_number)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nLOSO summary saved to: {summary_path}")


def run_ablation_study_kfold(
    study_number: int,
    config_path: str,
    device: str = 'cuda',
    experiment_name: str = None
):
    """
    Run Ablation Study with K-Fold cross-validation.
    
    Similar to LOSO but with explicit cache clearing between folds to prevent
    training freezes. Each session is used as test set once, with remaining
    sessions split into train/val.
    
    Args:
        study_number: Ablation study number
        config_path: Path to ablation YAML config
        device: Device to use ('cuda' or 'cpu')
        experiment_name: Optional experiment name prefix
    """
    print(f"\n{'='*60}")
    print(f"Starting Ablation Study {study_number} with K-Fold Cross-Validation")
    print(f"{'='*60}\n")
    
    # Load base + ablation config
    base_config = load_base_config()
    base_ablation_config = load_ablation_config(config_path, base_config)
    
    dataset_cfg = base_ablation_config.get('dataset', {})
    all_sessions = dataset_cfg.get('all_sessions', None)
    if all_sessions is None:
        # Fallback: infer from train/val/test if all_sessions not provided
        combined = []
        combined.extend(dataset_cfg.get('train_sessions', []))
        combined.extend(dataset_cfg.get('val_sessions', []))
        combined.extend(dataset_cfg.get('test_sessions', []))
        all_sessions = sorted(list(set(combined)))
    
    if len(all_sessions) < 2:
        raise ValueError("K-Fold requires at least 2 sessions in 'dataset.all_sessions'.")
    
    # Base experiment name
    if experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_experiment_name = f"ablation_{study_number:02d}_kfold_{timestamp}"
    else:
        base_experiment_name = experiment_name
    
    fold_results = []
    
    for fold_idx, test_session in enumerate(all_sessions):
        print(f"\n{'='*60}")
        print(f"K-Fold Fold {fold_idx + 1}/{len(all_sessions)}")
        print(f"{'='*60}")
        
        # Clear cache before starting new fold
        print("Clearing cache before fold...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("Cache cleared.")
        
        # Choose validation session as the next available session (cyclic),
        # different from the test session.
        remaining = [s for s in all_sessions if s != test_session]
        val_session = remaining[0]
        train_sessions = [s for s in all_sessions if s not in (test_session, val_session)]
        
        fold_config = copy.deepcopy(base_ablation_config)
        fold_config['dataset']['train_sessions'] = train_sessions
        fold_config['dataset']['val_sessions'] = [val_session]
        fold_config['dataset']['test_sessions'] = [test_session]
        
        # Create per-fold logging/checkpoint dirs to avoid collisions
        fold_suffix = f"fold_S{test_session}_test"
        fold_log_dir = os.path.join(fold_config['logging']['log_dir'], fold_suffix)
        fold_ckpt_dir = os.path.join(fold_config['logging']['checkpoint_dir'], fold_suffix)
        fold_config['logging']['log_dir'] = fold_log_dir
        fold_config['logging']['checkpoint_dir'] = fold_ckpt_dir
        
        fold_experiment_name = f"{base_experiment_name}_{fold_suffix}"
        
        print(f"Train sessions: {train_sessions}")
        print(f"Val session:    {[val_session]}")
        print(f"Test session:   {[test_session]}")
        print(f"Experiment:     {fold_experiment_name}")
        print()
        
        try:
            fold_metrics = _run_ablation_with_config(
                study_number=study_number,
                config=fold_config,
                device=device,
                experiment_name=fold_experiment_name
            )
            fold_metrics['test_session'] = int(test_session)
            fold_metrics['val_session'] = int(val_session)
            fold_metrics['train_sessions'] = [int(s) for s in train_sessions]
            fold_results.append(fold_metrics)
        except Exception as e:
            print(f"Error in fold {fold_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            # Clear cache even on error
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        finally:
            # Always clear cache after each fold
            print(f"\nClearing cache after fold {fold_idx + 1}...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print("Cache cleared.\n")
    
    # Aggregate K-Fold results across folds
    best_val_uas = [fr['best_val_ua'] for fr in fold_results if fr.get('best_val_ua') is not None]
    test_uas = [fr['test_unweighted_accuracy'] for fr in fold_results if fr.get('test_unweighted_accuracy') is not None]
    test_was = [fr['test_weighted_accuracy'] for fr in fold_results if fr.get('test_weighted_accuracy') is not None]
    
    summary = {
        'folds': fold_results,
        'num_folds': len(fold_results),
        'all_sessions': [int(s) for s in all_sessions],
    }
    if best_val_uas:
        summary['mean_best_val_ua'] = float(sum(best_val_uas) / len(best_val_uas))
        summary['std_best_val_ua'] = float(np.std(best_val_uas))
    if test_uas:
        summary['mean_test_unweighted_accuracy'] = float(sum(test_uas) / len(test_uas))
        summary['std_test_unweighted_accuracy'] = float(np.std(test_uas))
    if test_was:
        summary['mean_test_weighted_accuracy'] = float(sum(test_was) / len(test_was))
        summary['std_test_weighted_accuracy'] = float(np.std(test_was))
    
    # Save K-Fold summary next to the base checkpoint directory
    ckpt_root = base_ablation_config['logging']['checkpoint_dir']
    summary_path = Path(ckpt_root) / 'kfold_summary_ablation_{:02d}.json'.format(study_number)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("K-Fold Cross-Validation Summary")
    print(f"{'='*60}")
    if best_val_uas:
        print(f"Mean Best Val UA: {summary['mean_best_val_ua']:.4f} ± {summary['std_best_val_ua']:.4f}")
    if test_uas:
        print(f"Mean Test UA:     {summary['mean_test_unweighted_accuracy']:.4f} ± {summary['std_test_unweighted_accuracy']:.4f}")
    if test_was:
        print(f"Mean Test WA:     {summary['mean_test_weighted_accuracy']:.4f} ± {summary['std_test_weighted_accuracy']:.4f}")
    print(f"\nK-Fold summary saved to: {summary_path}")


def main():
    """Main function to run all ablation studies."""
    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument('--study', type=int, default=None,
                        help='Run specific study (1-39). If not specified, runs all.')
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
        (4, "config/ablation/ablation_04_cnn1d_head.yaml"),
        (5, "config/ablation/ablation_05_unfreeze_lstm.yaml"),
        (6, "config/ablation/ablation_06_unfreeze_transformer.yaml"),
        (7, "config/ablation/ablation_07_unfreeze_mid_high_lstm.yaml"),
        (8, "config/ablation/ablation_08_deep_lstm.yaml"),
        (9, "config/ablation/ablation_09_gia_lstm.yaml"),
        (10, "config/ablation/ablation_10_speechcue_internal.yaml"),
        (11, "config/ablation/ablation_11_melspec_vit.yaml"),
        (12, "config/ablation/ablation_12_speechcue_loso.yaml"),
        (13, "config/ablation/ablation_13_melspec_vit_speechcue_unfreeze_vit.yaml"),
        (14, "config/ablation/ablation_13a_regularized.yaml"),
        (15, "config/ablation/ablation_13b_no_speechcue.yaml"),
        (16, "config/ablation/ablation_13c_frozen_vit.yaml"),
        (17, "config/ablation/ablation_13d_high_reg.yaml"),
        (18, "config/ablation/ablation_13e_advanced_reg.yaml"),
        (19, "config/ablation/ablation_13f_focal_loss.yaml"),
        (20, "config/ablation/ablation_13d_kfold.yaml"),
        (21, "config/ablation/ablation_21_hierarchical_fusion.yaml"),
        (22, "config/ablation/ablation_22_tensor_fusion.yaml"),
        (23, "config/ablation/ablation_23_bilinear_fusion.yaml"),
        (24, "config/ablation/ablation_24_adaptive_weighted_fusion.yaml"),
        (25, "config/ablation/ablation_25_enhanced_cross_attention.yaml"),
        (26, "config/ablation/ablation_26_gated_fusion.yaml"),
        (27, "config/ablation/ablation_27_late_fusion.yaml"),
        (28, "config/ablation/ablation_28_glu_fusion.yaml"),
        (29, "config/ablation/ablation_25_kfold.yaml"),
        (30, "config/ablation/ablation_30_dirichlet.yaml"),
        (31, "config/ablation/ablation_31_temporal_alignment.yaml"),
        (32, "config/ablation/ablation_32_temporal_bucketing.yaml"),
        (33, "config/ablation/ablation_33_noise_aware_speechcue.yaml"),
        (34, "config/ablation/ablation_34_conflict_aware_attention.yaml"),
        (35, "config/ablation/ablation_35_modal_stratified_al.yaml"),
        (36, "config/ablation/ablation_36a_dirichlet_conflict.yaml"),  # 36A
        (37, "config/ablation/ablation_37_dirichlet_conflict_aware.yaml"),  # 37: Dirichlet + Conflict-Aware
        (38, "config/ablation/ablation_38_opensmile_residual.yaml"),  # Ablation 38: OpenSMILE + Attention Gates + Residual
        (39, "config/ablation/ablation_36c_hybrid_conflict.yaml"),  # 36C
    ]
    
    # Filter studies if specific one requested
    if args.study is not None:
        studies = [s for s in studies if s[0] == args.study]
        if not studies:
            print(f"Error: Study {args.study} not found. Valid studies: 1-39")
            return
    
    # Run each study
    for study_number, config_path in studies:
        try:
            experiment_name = f"{args.experiment_name}_study_{study_number}" if args.experiment_name else None
            if study_number == 12:
                # Special case: Ablation 12 runs 5-fold LOSO instead of a single split
                run_ablation_study_loso(
                    study_number=study_number,
                    config_path=config_path,
                    device=args.device,
                    experiment_name=experiment_name
                )
            elif study_number in (20, 29):
                # Special case: Ablation 20 and 29 run K-Fold cross-validation with cache clearing
                run_ablation_study_kfold(
                    study_number=study_number,
                    config_path=config_path,
                    device=args.device,
                    experiment_name=experiment_name
                )
            else:
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

