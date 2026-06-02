"""
Standalone training script for the lightweight multimodal experiment.

Reuses training utilities from src.train (dataloaders, train_epoch, validate,
save_checkpoint, etc.) but swaps the model for LightweightMultimodalModel and
uses a 4-group optimizer (audio / text / vision / fusion+head).
"""

import os
import sys

# Fix for Windows HF cache issues: redirect HuggingFace cache to project-local dir
os.environ.setdefault('HF_HOME', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.cache', 'hf'))
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', os.path.join(os.environ['HF_HOME'], 'hub'))

# Fix for corrupted HF token file on Windows: monkey-patch get_token to avoid reading file
import huggingface_hub.utils._auth as _auth_module
_auth_module.get_token = lambda: _auth_module._get_token_from_google_colab() or _auth_module._get_token_from_environment()

import yaml
import torch
import torch.nn as nn
import argparse
import math
import numpy as np
from pathlib import Path

# Ensure project root is on path (__file__ = src/experiments/train_lightweight.py)
_sys_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _sys_dir)

from src.experiments.lightweight_encoders import LightweightMultimodalModel
from src.train import (
    create_dataloaders,
    train_epoch,
    validate,
    save_checkpoint,
    load_config,
    create_loss_function,
    log_loss_contract_traceability,
    plot_training_curves,
)
from src.utils.logger import setup_logger, TensorBoardLogger, ExperimentTracker, print_model_summary
from src.utils.metrics import compute_class_weights


def parse_args():
    parser = argparse.ArgumentParser(description='Train lightweight multimodal SER model')
    parser.add_argument('--config', type=str, default='config/experiment_lightweight.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from the latest checkpoint in the experiment checkpoint directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Override training.num_epochs from config')
    return parser.parse_args()


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the latest checkpoint_epoch_*.pt in the checkpoint directory."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob('checkpoint_epoch_*.pt'))
    if not checkpoints:
        return None
    return str(checkpoints[-1])


def create_model(config: dict, device: str, logger):
    """Build LightweightMultimodalModel and move to device."""
    logger.info("Creating lightweight multimodal model...")
    model = LightweightMultimodalModel(config)
    model = model.to(device)
    print_model_summary(model, logger)
    return model


def create_optimizer_and_scheduler(model, config: dict, logger=None):
    """AdamW with 4 LR groups: audio, text, vision, fusion+head."""
    training_config = config.get('training', {})
    adam_epsilon = float(training_config.get('adam_epsilon', 1e-8))
    weight_decay = training_config.get('weight_decay', 0.05)

    lr_audio = float(training_config.get('learning_rate_audio', 1e-4))
    lr_text = float(training_config.get('learning_rate_text', 1e-4))
    lr_vision = float(training_config.get('learning_rate_vision', lr_audio))
    lr_fusion = float(training_config.get('learning_rate_fusion', 5e-5))

    audio_params, text_params, vision_params, fusion_params = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'audio_encoder' in name:
            audio_params.append(param)
        elif 'text_encoder' in name:
            text_params.append(param)
        elif 'vision_encoder' in name:
            vision_params.append(param)
        else:
            fusion_params.append(param)

    param_groups = []
    if audio_params:
        param_groups.append({'params': audio_params, 'lr': lr_audio})
    if text_params:
        param_groups.append({'params': text_params, 'lr': lr_text})
    if vision_params:
        param_groups.append({'params': vision_params, 'lr': lr_vision})
    if fusion_params:
        param_groups.append({'params': fusion_params, 'lr': lr_fusion})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay, eps=adam_epsilon)

    if logger is not None:
        logger.info(
            f"Optimizer groups: audio={len(audio_params)} params @ {lr_audio}, "
            f"text={len(text_params)} params @ {lr_text}, "
            f"vision={len(vision_params)} params @ {lr_vision}, "
            f"fusion={len(fusion_params)} params @ {lr_fusion}"
        )

    # Scheduler
    scheduler_type = training_config.get('scheduler_type', 'plateau')
    scheduler = None
    if scheduler_type == 'plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=training_config.get('scheduler_mode', 'min'),
            factor=training_config.get('scheduler_factor', 0.5),
            patience=training_config.get('scheduler_patience', 3),
            threshold=float(training_config.get('scheduler_threshold', 1e-4)),
            min_lr=float(training_config.get('scheduler_min_lr', 1e-7)),
            cooldown=training_config.get('scheduler_cooldown', 1),
            verbose=training_config.get('scheduler_verbose', True),
        )
    elif scheduler_type in ('cosine', 'cosine_with_warmup'):
        from transformers import get_cosine_schedule_with_warmup
        # We need num_training_steps; this will be set by caller if needed
        # For simplicity, we create it after we know steps in main()
        pass
    else:
        if logger is not None:
            logger.info(f"No scheduler (type={scheduler_type})")

    return optimizer, scheduler


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    if args.num_epochs is not None:
        config['training']['num_epochs'] = int(args.num_epochs)

    # Override device
    if args.device:
        config['compute']['device'] = args.device
    device = config['compute']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    # Seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if device == 'cuda':
        torch.cuda.manual_seed(config['seed'])

    # Logging
    logger = setup_logger('train_lightweight', config['logging']['log_dir'])
    logger.info(f"Starting lightweight training on {device}")
    logger.info(f"Configuration: {config}")

    # TensorBoard
    if config['logging']['use_tensorboard']:
        tb_logger = TensorBoardLogger(
            config['logging']['log_dir'],
            experiment_name=args.experiment_name
        )
    else:
        tb_logger = None

    # Experiment tracker
    exp_tracker = ExperimentTracker(
        config['logging']['checkpoint_dir'],
        experiment_name=args.experiment_name
    )
    exp_tracker.log_config(config)

    # Fallback to local tokenizer if HF Hub is unreachable (Windows cache issues)
    local_tokenizer = r'C:\temp\hf_tokenizer'
    if os.path.isdir(local_tokenizer) and os.path.isfile(os.path.join(local_tokenizer, 'tokenizer.json')):
        config['text']['model_name'] = local_tokenizer
        logger.info(f"Using local tokenizer from {local_tokenizer}")

    # Dataloaders
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        config, logger, audio_only=False
    )

    # Model
    model = create_model(config, device, logger)

    # Loss function
    loss_config = config['training'].get('loss', {})
    loss_type = loss_config.get('type', 'cross_entropy')
    gamma = loss_config.get('gamma', 2.0)
    label_smoothing = loss_config.get('label_smoothing', 0.0)

    if loss_type == 'focal':
        num_classes = len(config['dataset']['emotions'])
        with torch.no_grad():
            norm_weights = class_weights / (class_weights.sum() + 1e-9) * num_classes
            norm_weights = torch.clamp(norm_weights, min=0.5, max=5.0)
        alpha = norm_weights
        logger.info(f"FocalLoss alpha: {alpha.tolist()}")
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
    log_loss_contract_traceability(logger, loss_type, gamma, label_smoothing)

    # Optimizer and scheduler
    accum_steps = max(1, int(config['training'].get('gradient_accumulation_steps', 1)))
    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    num_training_steps = steps_per_epoch * int(config['training']['num_epochs'])

    optimizer, scheduler = create_optimizer_and_scheduler(model, config, logger=logger)

    # Handle cosine scheduler which needs num_training_steps
    scheduler_type = config['training'].get('scheduler_type', 'plateau')
    if scheduler_type in ('cosine', 'cosine_with_warmup'):
        from transformers import get_cosine_schedule_with_warmup
        warmup_ratio = float(config['training'].get('warmup_ratio', 0.1))
        num_warmup_steps = int(warmup_ratio * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        logger.info(f"Cosine scheduler: {num_warmup_steps} warmup, {num_training_steps} total steps")

    if accum_steps > 1:
        logger.info(
            f"Gradient accumulation: {accum_steps} micro-batches per step; "
            f"~{steps_per_epoch} optimizer steps/epoch; total scheduler steps ≈ {num_training_steps}"
        )

    # Mixed precision scaler
    scaler = None
    if config['training'].get('mixed_precision', False):
        if device == 'cuda':
            try:
                from torch.amp import GradScaler
                scaler = GradScaler('cuda')
            except (ImportError, TypeError):
                from torch.cuda.amp import GradScaler
                scaler = GradScaler()
            logger.info("Using mixed precision (AMP)")
        else:
            logger.info("Mixed precision disabled on CPU")

    # Resume from checkpoint if requested
    start_epoch = 1
    best_val_ua = 0.0
    if args.resume:
        resume_path = args.checkpoint
        if resume_path is None:
            resume_path = find_latest_checkpoint(config['logging']['checkpoint_dir'])
        if resume_path and Path(resume_path).exists():
            logger.info(f"Resuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_ua = checkpoint.get('metrics', {}).get('unweighted_accuracy', 0.0)
            logger.info(f"Resumed at epoch {start_epoch}, best_val_ua={best_val_ua:.4f}")
        else:
            logger.warning(f"No checkpoint found to resume from (looked for: {resume_path}). Starting from scratch.")

    # Training loop state
    patience_counter = 0
    train_losses, val_losses = [], []
    train_metrics_list, val_metrics_list = [], []

    logger.info("Starting training...")
    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{config['training']['num_epochs']}")

        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, config, epoch, logger, tb_logger, audio_only=False
        )

        # Validate
        if epoch % config['logging']['val_every_n_epochs'] == 0:
            val_loss, val_metrics, val_metrics_tracker = validate(
                model, val_loader, criterion, device, config, logger, audio_only=False
            )

            logger.info(f"Train Loss: {train_loss:.4f}, Train UA: {train_metrics['unweighted_accuracy']:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val UA: {val_metrics['unweighted_accuracy']:.4f}")

            if tb_logger:
                tb_logger.log_metrics(train_metrics, epoch, prefix='train/')
                tb_logger.log_metrics(val_metrics, epoch, prefix='val/')
                tb_logger.log_scalar('epoch/train_loss', train_loss, epoch)
                tb_logger.log_scalar('epoch/val_loss', val_loss, epoch)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_metrics_list.append(train_metrics['unweighted_accuracy'])
            val_metrics_list.append(val_metrics['unweighted_accuracy'])

            exp_tracker.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }, epoch)

            # Save checkpoint
            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                config, config['logging']['checkpoint_dir'], logger,
                filename=f'checkpoint_epoch_{epoch}.pt'
            )

            # Best model
            if val_metrics['unweighted_accuracy'] > best_val_ua:
                best_val_ua = val_metrics['unweighted_accuracy']
                logger.info(f"New best model! UA: {best_val_ua:.4f}")
                best_checkpoint_path = save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics,
                    config, config['logging']['checkpoint_dir'], logger,
                    filename='best_model.pt'
                )
                exp_tracker.update_best_metrics(val_metrics)
                exp_tracker.save_checkpoint_info(best_checkpoint_path, epoch, val_metrics)
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"Patience: {patience_counter}/{config['training']['early_stopping_patience']}")

            # Plateau scheduler step
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_metric = config['training'].get('scheduler_metric', 'val_loss')
                metric_value = val_loss if scheduler_metric == 'val_loss' else val_metrics['unweighted_accuracy']
                lr_before = [g['lr'] for g in optimizer.param_groups]
                scheduler.step(metric_value)
                lr_after = [g['lr'] for g in optimizer.param_groups]
                if any(lb > la for lb, la in zip(lr_before, lr_after)):
                    logger.info(f"Learning rate reduced! Before: {lr_before}, After: {lr_after}")
                else:
                    logger.info(f"Current learning rates: {lr_after}")
                if tb_logger:
                    for idx, lr in enumerate(lr_after):
                        group_name = optimizer.param_groups[idx].get('name', f'group_{idx}')
                        tb_logger.log_scalar(f'train/lr_{group_name}', lr, epoch)

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
        fig.savefig(Path(config['logging']['checkpoint_dir']) / 'training_curves.png')

    # Load best checkpoint for final test
    best_ckpt = Path(config['logging']['checkpoint_dir']) / 'best_model.pt'
    if best_ckpt.exists():
        logger.info(f"\nLoading best checkpoint from {best_ckpt} for test evaluation...")
        checkpoint = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logger.info(f"Best model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        logger.warning("No best_model.pt found; evaluating with last-epoch weights.")

    # Final test evaluation
    logger.info("\nEvaluating on test set...")
    test_loss, test_metrics, test_metrics_tracker = validate(
        model, test_loader, criterion, device, config, logger, audio_only=False
    )
    logger.info(f"Test Loss: {test_loss:.4f}")
    test_metrics_tracker.print_report()

    cm_fig = test_metrics_tracker.plot_confusion_matrix(normalize=True)
    cm_fig.savefig(Path(config['logging']['checkpoint_dir']) / 'confusion_matrix_test.png')

    if tb_logger:
        tb_logger.log_metrics(test_metrics, 0, prefix='test/')
        tb_logger.log_figure('test/confusion_matrix', cm_fig, 0)
        tb_logger.close()

    logger.info("Training completed!")


if __name__ == '__main__':
    main()
