"""Training script for Conformer multimodal speech emotion recognition.

Loads config and dataset, builds the model (or ablation model), and runs
training with optional mixed precision, gradient accumulation, and
learning-rate scheduling. Supports resuming from checkpoint.
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
import math
import warnings
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import IEMOCAPDataset, AudioPreprocessor, TextPreprocessor, collate_fn, collate_fn_audio_only
from src.models import ConformerMultimodalModel, AudioOnlyModel
from src.utils.metrics import EmotionMetrics, compute_class_weights, plot_training_curves
from src.utils.logger import setup_logger, TensorBoardLogger, ExperimentTracker, print_model_summary
from src.utils.losses import create_loss_function, log_loss_contract_traceability


def parse_args():
    """Parse command line arguments for training.

    Returns:
        Namespace with config path, optional checkpoint, device, and experiment_name.
    """
    parser = argparse.ArgumentParser(description='Train Conformer multimodal SER model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Override training.num_epochs from config (e.g. 1 for smoke tests)')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Nested dict with dataset, model, training, logging, etc.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict, logger, audio_only: bool = False):
    """Create train, validation, and test dataloaders."""
    logger.info(f"Creating dataloaders for {'audio-only' if audio_only else 'multimodal'} model...")
    
    # Initialize preprocessors with augmentation config
    audio_config = config.get('audio', {})
    
    # Audio augmentation config
    apply_audio_augment = audio_config.get('apply_waveform_augment', False)
    audio_augment_config = audio_config.get('augmentation', {})
    
    audio_preprocessor = AudioPreprocessor(
        sample_rate=config['dataset']['sample_rate'],
        use_mel_spectrogram=False,  # Use raw waveform for Conformer
        apply_spec_augment=False,
        apply_waveform_augment=apply_audio_augment,
        augmentation_config=audio_augment_config
    )
    
    # Text / prosody configuration
    text_config = config.get('text', {})
    use_prosody_cues = text_config.get('use_prosody_cues', False) if not audio_only else False

    # Text preprocessor only if not audio-only
    text_preprocessor = None
    if not audio_only:
        apply_text_augment = text_config.get('apply_text_augment', False)
        text_augment_config = text_config.get('augmentation', {})
        
        text_preprocessor = TextPreprocessor(
            model_name=text_config.get('model_name', config['text']['model_name']),
            max_length=text_config.get('max_length', config['text']['max_length']),
            apply_text_augment=apply_text_augment,
            text_augmentation_config=text_augment_config
        )
        text_preprocessor.train()
    
    # Set training mode for train dataset (augmentations enabled)
    audio_preprocessor.train()
    
    # Create datasets
    train_dataset = IEMOCAPDataset(
        data_path=config['dataset']['data_path'],
        sessions=config['dataset']['train_sessions'],
        emotions=config['dataset']['emotions'],
        sample_rate=config['dataset']['sample_rate'],
        audio_preprocessor=audio_preprocessor,
        text_preprocessor=text_preprocessor,
        max_audio_length=config['audio']['max_length'],
        training=True,  # Enable random segment sampling for training
        use_prosody_cues=use_prosody_cues
    )
    
    # Create validation preprocessors (eval mode, no augmentation)
    val_audio_preprocessor = AudioPreprocessor(
        sample_rate=config['dataset']['sample_rate'],
        use_mel_spectrogram=False,
        apply_spec_augment=False,
        apply_waveform_augment=False  # No augmentation for val
    )
    val_audio_preprocessor.eval()
    
    val_text_preprocessor = None
    if not audio_only:
        val_text_preprocessor = TextPreprocessor(
            model_name=text_config.get('model_name', config['text']['model_name']),
            max_length=text_config.get('max_length', config['text']['max_length']),
            apply_text_augment=False  # No augmentation for val
        )
        val_text_preprocessor.eval()
    
    val_dataset = IEMOCAPDataset(
        data_path=config['dataset']['data_path'],
        sessions=config['dataset']['val_sessions'],
        emotions=config['dataset']['emotions'],
        sample_rate=config['dataset']['sample_rate'],
        audio_preprocessor=val_audio_preprocessor,
        text_preprocessor=val_text_preprocessor,
        max_audio_length=config['audio']['max_length'],
        training=False,  # Use deterministic start for validation
        use_prosody_cues=use_prosody_cues
    )
    
    # Create test preprocessors (eval mode, no augmentation)
    test_audio_preprocessor = AudioPreprocessor(
        sample_rate=config['dataset']['sample_rate'],
        use_mel_spectrogram=False,
        apply_spec_augment=False,
        apply_waveform_augment=False  # No augmentation for test
    )
    test_audio_preprocessor.eval()
    
    test_text_preprocessor = None
    if not audio_only:
        test_text_preprocessor = TextPreprocessor(
            model_name=text_config.get('model_name', config['text']['model_name']),
            max_length=text_config.get('max_length', config['text']['max_length']),
            apply_text_augment=False  # No augmentation for test
        )
        test_text_preprocessor.eval()
    
    test_dataset = IEMOCAPDataset(
        data_path=config['dataset']['data_path'],
        sessions=config['dataset']['test_sessions'],
        emotions=config['dataset']['emotions'],
        sample_rate=config['dataset']['sample_rate'],
        audio_preprocessor=test_audio_preprocessor,
        text_preprocessor=test_text_preprocessor,
        max_audio_length=config['audio']['max_length'],
        training=False,  # Use deterministic start for test
        use_prosody_cues=use_prosody_cues
    )
    
    # Choose collate function based on model type
    collate_fn_to_use = collate_fn_audio_only if audio_only else collate_fn
    
    # Get DataLoader optimization settings
    compute_config = config.get('compute', {})
    num_workers = compute_config.get('num_workers', 2)
    pin_memory = compute_config.get('pin_memory', True)
    prefetch_factor = compute_config.get('prefetch_factor', 2)
    persistent_workers = compute_config.get('persistent_workers', True)
    non_blocking = compute_config.get('non_blocking', True)

    # Windows: multiprocessing workers + tensor sharing often raises
    # RuntimeError: Couldn't open shared file mapping (WinError 1455). Single-process
    # loading avoids it unless compute.windows_keep_dataloader_workers is true.
    if (
        sys.platform == 'win32'
        and num_workers > 0
        and not compute_config.get('windows_keep_dataloader_workers', False)
    ):
        logger.warning(
            "Windows: forcing DataLoader num_workers from %s to 0 to avoid shared-memory "
            "failures between worker processes (e.g. RuntimeError 1455). "
            "Set compute.num_workers: 0 in YAML to silence this, or "
            "compute.windows_keep_dataloader_workers: true to keep workers (not recommended).",
            num_workers,
        )
        num_workers = 0
        persistent_workers = False
    
    # Create dataloaders with optimizations for PCIe x8 (RTX 4060)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=collate_fn_to_use
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=collate_fn_to_use
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=collate_fn_to_use
    )
    
    # Store non_blocking setting for use in training loop
    train_loader.non_blocking = non_blocking
    val_loader.non_blocking = non_blocking
    test_loader.non_blocking = non_blocking
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Compute class weights
    class_weights = train_dataset.get_class_weights()
    logger.info(f"Class weights: {class_weights}")
    
    return train_loader, val_loader, test_loader, class_weights


def create_model(config: dict, device: str, logger, audio_only: bool = False):
    """Create and initialize model."""
    logger.info(f"Creating {'audio-only' if audio_only else 'multimodal'} model...")
    
    if audio_only:
        model = AudioOnlyModel(
            audio_encoder_name=config['model']['audio_encoder'],
            num_classes=config['model']['num_classes'],
            hidden_dim=config['model']['hidden_dim'],
            dropout=config['model']['dropout'],
            freeze_encoder=config['model'].get('freeze_audio_encoder', True),
            audio_unfreeze_last_n_layers=config['model'].get('audio_unfreeze_last_n_layers', 0)
        )
    else:
        model = ConformerMultimodalModel(
            audio_encoder_name=config['model']['audio_encoder'],
            text_encoder_name=config['model']['text_encoder'],
            num_classes=config['model']['num_classes'],
            hidden_dim=config['model']['hidden_dim'],
            dropout=config['model']['dropout'],
            fusion_type=config['model']['fusion_type'],
            freeze_audio_encoder=config['model'].get('freeze_audio_encoder', True),
            audio_unfreeze_last_n_layers=config['model'].get('audio_unfreeze_last_n_layers', 0),
            freeze_text_encoder=config['model'].get('freeze_text_encoder', True)
        )
    
    model = model.to(device)
    
    # Print model summary
    print_model_summary(model, logger)
    
    return model


def create_optimizer_and_scheduler(
    model,
    config: dict,
    num_training_steps: int,
    audio_only: bool = False,
    logger=None,
):
    """Create optimizer and learning rate scheduler."""
    
    if audio_only:
        # For audio-only, separate parameters for encoder (if unfrozen) and classifier
        audio_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'audio_encoder' in name:
                audio_params.append(param)
            else:
                classifier_params.append(param)
        
        # Use different learning rates if both groups exist, otherwise use single LR
        # Ensure adam_epsilon is a float (YAML may parse 1e-07 as string)
        adam_epsilon = float(config['training'].get('adam_epsilon', 1e-8))
        
        # Ensure learning rates are floats (YAML may parse 1e-06 as string)
        lr_audio = float(config['training']['learning_rate_audio'])
        lr_classifier = float(config['training'].get('learning_rate_classifier', lr_audio))
        
        if audio_params and classifier_params:
            optimizer = torch.optim.AdamW([
                {'params': audio_params, 'lr': lr_audio},
                {'params': classifier_params, 'lr': lr_classifier}
            ], weight_decay=config['training']['weight_decay'],
               eps=adam_epsilon)
        else:
            # Single group if only one type of parameters is trainable
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                [{'params': params}],
                lr=lr_audio,
                weight_decay=config['training']['weight_decay'],
                eps=adam_epsilon
            )
    else:
        # Separate parameters for different learning rates (multimodal)
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
        
        # Ensure adam_epsilon is a float (YAML may parse 1e-07 as string)
        adam_epsilon = float(config['training'].get('adam_epsilon', 1e-8))
        
        # Ensure learning rates are floats (YAML may parse 1e-06 as string)
        lr_audio = float(config['training']['learning_rate_audio'])
        lr_text = float(config['training']['learning_rate_text'])
        lr_fusion = float(config['training']['learning_rate_fusion'])
        
        # Create optimizer with parameter groups
        optimizer = torch.optim.AdamW([
            {'params': audio_params, 'lr': lr_audio},
            {'params': text_params, 'lr': lr_text},
            {'params': fusion_params, 'lr': lr_fusion}
        ], weight_decay=config['training']['weight_decay'],
           eps=adam_epsilon)
    
    # Create scheduler
    scheduler_type = config['training'].get('scheduler_type', 'cosine')
    warmup_ratio = float(config['training'].get('warmup_ratio', 0.0) or 0.0)

    if scheduler_type == 'plateau' and warmup_ratio > 0:
        msg = (
            "training.warmup_ratio is set but scheduler_type is 'plateau': "
            "ReduceLROnPlateau does not use warmup_ratio; warmup is ignored. "
            "Use cosine/linear scheduler for warmup, or remove warmup_ratio from config."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        if logger is not None:
            logger.warning(msg)

    if scheduler_type == 'linear':
        from transformers import get_linear_schedule_with_warmup
        num_warmup_steps = int(config['training'].get('warmup_ratio', 0.1) * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type in ('cosine', 'cosine_with_warmup'):
        from transformers import get_cosine_schedule_with_warmup
        num_warmup_steps = int(config['training'].get('warmup_ratio', 0.1) * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type == 'plateau':
        # ReduceLROnPlateau scheduler - reduces LR based on validation metric
        
        scheduler_metric = config['training'].get('scheduler_metric', 'val_loss')
        scheduler_mode = config['training'].get('scheduler_mode', 'min')
        scheduler_factor = config['training'].get('scheduler_factor', 0.5)
        scheduler_patience = config['training'].get('scheduler_patience', 5)
        # Ensure threshold is a float (YAML might parse 1e-5 as string)
        scheduler_threshold = float(config['training'].get('scheduler_threshold', 1e-4))
        scheduler_min_lr = config['training'].get('scheduler_min_lr', 1e-7)
        scheduler_cooldown = config['training'].get('scheduler_cooldown', 2)
        scheduler_verbose = config['training'].get('scheduler_verbose', True)
        
        # For ReduceLROnPlateau with multiple parameter groups, we need to set min_lr for each group
        # Calculate min_lr for each group based on initial LR
        min_lr_list = []
        for param_group in optimizer.param_groups:
            initial_lr = param_group['lr']
            # Set min_lr as a fraction of initial LR or absolute minimum
            # Use 1% of initial LR as minimum, but not less than scheduler_min_lr
            param_group_min_lr = max(initial_lr * 0.01, scheduler_min_lr)
            min_lr_list.append(param_group_min_lr)
        
        # If all groups have same min_lr, use single value, otherwise use list
        if len(set(min_lr_list)) == 1:
            min_lr = min_lr_list[0]
        else:
            min_lr = min_lr_list
        
        # Create scheduler
        # Note: verbose parameter is only available in PyTorch 1.4+, so we check dynamically
        scheduler_params = {
            'mode': scheduler_mode,
            'factor': scheduler_factor,
            'patience': scheduler_patience,
            'threshold': scheduler_threshold,
            'min_lr': min_lr,
            'cooldown': scheduler_cooldown
        }
        
        # Try to add verbose parameter if supported
        import inspect
        sig = inspect.signature(ReduceLROnPlateau.__init__)
        if 'verbose' in sig.parameters:
            scheduler_params['verbose'] = scheduler_verbose
        
        scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)
    else:
        scheduler = None
    
    return optimizer, scheduler


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    config,
    epoch,
    logger,
    tb_logger,
    audio_only: bool = False
):
    """Train for one epoch with optional gradient accumulation."""
    model.train()

    metrics_tracker = EmotionMetrics(config['dataset']['emotions'])
    total_loss = 0.0
    num_loss_batches = 0

    accum_steps = max(1, int(config['training'].get('gradient_accumulation_steps', 1)))
    mixed_precision = config['training'].get('mixed_precision', False)
    use_amp_scaler = scaler is not None
    num_batches = len(train_loader)
    steps_per_epoch = math.ceil(num_batches / accum_steps) if num_batches else 0

    optimizer.zero_grad(set_to_none=True)
    accum_count = 0
    optimizer_step_idx = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    def _optimizer_step(batch_idx: int) -> bool:
        """Clip, step optimizer, zero grad, step scheduler. Returns True if stepped."""
        nonlocal optimizer_step_idx
        max_norm = float(config['training']['max_grad_norm'])

        if use_amp_scaler:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if not torch.isfinite(grad_norm.clone().detach()):
                logger.warning(
                    f"NaN/Inf gradients at batch {batch_idx}, skipping optimizer step."
                )
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                return False
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if not torch.isfinite(grad_norm.clone().detach()):
                logger.warning(
                    f"NaN/Inf gradients at batch {batch_idx}, skipping optimizer step."
                )
                optimizer.zero_grad(set_to_none=True)
                return False
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step()
        optimizer_step_idx += 1
        return True

    for batch_idx, batch in enumerate(pbar):
        non_blocking = getattr(train_loader, 'non_blocking', False)
        audio = batch['audio'].to(device, non_blocking=non_blocking)
        labels = batch['labels'].to(device, non_blocking=non_blocking)

        if not torch.isfinite(audio).all():
            logger.error(f"NaN/Inf detected in audio input at batch {batch_idx}!")
            continue

        if mixed_precision:
            dtype = None
            if mixed_precision == 'bf16':
                dtype = torch.bfloat16
            elif mixed_precision in ('fp16', True):
                dtype = torch.float16
            elif mixed_precision == 'fp8':
                dtype = torch.bfloat16

            with autocast(device_type='cuda', dtype=dtype):
                if audio_only:
                    outputs = model(audio, labels)
                else:
                    text_input_ids = batch['text_input_ids'].to(
                        device, non_blocking=non_blocking
                    )
                    text_attention_mask = batch['text_attention_mask'].to(
                        device, non_blocking=non_blocking
                    )
                    outputs = model(
                        audio, text_input_ids, text_attention_mask, labels
                    )
            if 'alphas' in outputs:
                loss = criterion(outputs['alphas'], labels)
            else:
                loss = criterion(outputs['logits'], labels)
            if outputs.get('loss_contrastive') is not None:
                cw = config.get('training', {}).get('loss', {}).get(
                    'contrastive_weight', 0.0
                )
                if cw > 0:
                    loss = loss + cw * outputs['loss_contrastive']
        else:
            if audio_only:
                outputs = model(audio, labels)
            else:
                text_input_ids = batch['text_input_ids'].to(
                    device, non_blocking=non_blocking
                )
                text_attention_mask = batch['text_attention_mask'].to(
                    device, non_blocking=non_blocking
                )
                outputs = model(
                    audio, text_input_ids, text_attention_mask, labels
                )
            if 'alphas' in outputs:
                loss = criterion(outputs['alphas'], labels)
            else:
                loss = criterion(outputs['logits'], labels)
            if outputs.get('loss_contrastive') is not None:
                cw = config.get('training', {}).get('loss', {}).get(
                    'contrastive_weight', 0.0
                )
                if cw > 0:
                    loss = loss + cw * outputs['loss_contrastive']

        if not torch.isfinite(loss):
            logger.error(f"NaN/Inf loss detected at batch {batch_idx}!")
            if 'logits' in outputs:
                lo = outputs['logits']
                logger.error(
                    f"Logits stats: min={lo.min().item()}, max={lo.max().item()}, "
                    f"mean={lo.mean().item()}, std={lo.std().item()}"
                )
            logger.warning("Skipping this batch and continuing...")
            continue

        loss_scaled = loss / accum_steps

        if use_amp_scaler:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        accum_count += 1

        should_step = accum_count >= accum_steps or (batch_idx + 1 == num_batches)
        if should_step and accum_count > 0:
            stepped = _optimizer_step(batch_idx)
            accum_count = 0
            if (
                stepped
                and tb_logger
                and scheduler is not None
                and not isinstance(scheduler, ReduceLROnPlateau)
            ):
                tb_logger.log_scalar(
                    'train/lr',
                    scheduler.get_last_lr()[0],
                    (epoch - 1) * steps_per_epoch + optimizer_step_idx - 1,
                )

        loss_value = loss.item()
        if torch.isfinite(loss):
            total_loss += loss_value
            num_loss_batches += 1
            metrics_tracker.update(outputs['predictions'], labels)

            if batch_idx == 0 and epoch == 1 and 'logits' in outputs:
                predictions_np = outputs['predictions'].cpu().numpy()
                unique_preds = torch.unique(outputs['predictions']).cpu().numpy()
                lo = outputs['logits']
                logger.info(f"First training batch - Predictions: {predictions_np[:10]}")
                logger.info(f"First training batch - Labels: {labels.cpu().numpy()[:10]}")
                logger.info(
                    f"First training batch - Logits range: [{lo.min().item():.4f}, {lo.max().item():.4f}]"
                )
                logger.info(
                    f"First training batch - Logits mean: {lo.mean().item():.4f}, std: {lo.std().item():.4f}"
                )
                logger.info(f"First training batch - Unique predictions: {unique_preds}")
                if len(unique_preds) == 1:
                    logger.warning(
                        f"[WARNING] Model is predicting only class {unique_preds[0]} for all samples!"
                    )

        pbar.set_postfix({'loss': loss_value})

        global_step = epoch * num_batches + batch_idx
        if batch_idx % config['logging']['log_every_n_steps'] == 0:
            if torch.isfinite(loss) and tb_logger:
                tb_logger.log_scalar('train/loss_step', loss.item(), global_step)

    avg_loss = total_loss / max(num_loss_batches, 1)
    metrics = metrics_tracker.compute()
    return avg_loss, metrics


def detect_class_collapse(predictions: np.ndarray, num_classes: int, threshold: float = 0.9) -> tuple:
    """
    Detect if model is collapsing to a single class.
    
    Args:
        predictions: Array of predicted class indices
        num_classes: Total number of classes
        threshold: Threshold for considering a class as dominant (default: 0.9 = 90%)
    
    Returns:
        (is_collapsed, dominant_class, class_distribution)
    """
    unique, counts = np.unique(predictions, return_counts=True)
    total = len(predictions)
    
    # Calculate class distribution
    class_dist = np.zeros(num_classes)
    for cls, count in zip(unique, counts):
        class_dist[cls] = count / total
    
    # Check if any class dominates
    max_freq = class_dist.max()
    dominant_class = class_dist.argmax()
    is_collapsed = max_freq >= threshold
    
    return is_collapsed, dominant_class, class_dist


def validate(model, val_loader, criterion, device, config, logger, audio_only: bool = False):
    """Validate model."""
    model.eval()
    
    metrics_tracker = EmotionMetrics(config['dataset']['emotions'])
    total_loss = 0
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
            # Move to device
            # Use non_blocking for validation too
            non_blocking = getattr(val_loader, 'non_blocking', False)
            audio = batch['audio'].to(device, non_blocking=non_blocking)
            labels = batch['labels'].to(device, non_blocking=non_blocking)
            
            # Forward pass
            if audio_only:
                outputs = model(audio, labels)
            else:
                text_input_ids = batch['text_input_ids'].to(
                    device, non_blocking=non_blocking
                )
                text_attention_mask = batch['text_attention_mask'].to(
                    device, non_blocking=non_blocking
                )
                outputs = model(audio, text_input_ids, text_attention_mask, labels)
            # Handle Dirichlet models that output alphas instead of logits
            if 'alphas' in outputs:
                loss = criterion(outputs['alphas'], labels)
            else:
                loss = criterion(outputs['logits'], labels)
            
            # Check for NaN/Inf in loss
            if not torch.isfinite(loss):
                logger.warning(f"NaN/Inf loss detected in validation at batch {batch_idx}")
                continue
            
            # Update metrics
            total_loss += loss.item()
            predictions_np = outputs['predictions'].cpu().numpy()
            metrics_tracker.update(outputs['predictions'], labels)
            all_predictions.extend(predictions_np)
            
            # Debug: Log predictions distribution for first batch
            if batch_idx == 0:
                unique_preds = torch.unique(outputs['predictions']).cpu().numpy()
                logger.info(f"First validation batch - Predictions: {predictions_np[:10]}")
                logger.info(f"First validation batch - Labels: {labels.cpu().numpy()[:10]}")
                logger.info(f"First validation batch - Logits range: [{outputs['logits'].min().item():.4f}, {outputs['logits'].max().item():.4f}]")
                logger.info(f"First validation batch - Logits mean: {outputs['logits'].mean().item():.4f}, std: {outputs['logits'].std().item():.4f}")
                logger.info(f"First validation batch - Unique predictions: {unique_preds}")
                if len(unique_preds) == 1:
                    logger.warning(f"WARNING: Model is predicting only class {unique_preds[0]} for all samples!")
    
    # Compute metrics
    avg_loss = total_loss / len(val_loader)
    metrics = metrics_tracker.compute()
    
    # Check for class collapse across all validation batches
    all_predictions = np.array(all_predictions)
    num_classes = len(config['dataset']['emotions'])
    is_collapsed, dominant_class, class_dist = detect_class_collapse(all_predictions, num_classes, threshold=0.8)
    
    if is_collapsed:
        emotion_labels = config['dataset']['emotions']
        dominant_emotion = emotion_labels[dominant_class]
        logger.error(f"[CLASS COLLAPSE DETECTED] Model is predicting '{dominant_emotion}' (class {dominant_class}) for {class_dist[dominant_class]*100:.1f}% of samples")
        logger.error(f"   Class distribution: {dict(zip(emotion_labels, [f'{d*100:.1f}%' for d in class_dist]))}")
        logger.error(f"   This indicates the model has collapsed. VAL UA will be stuck around {1.0/num_classes:.3f}")
        logger.error(f"   Check: dropout (should be <=0.1), learning_rate_audio (should be <=1e-5), label_smoothing (should be 0.0), weight_decay (should be <=0.01)")
    else:
        unique_classes = len(np.unique(all_predictions))
        logger.info(f"[OK] Model predicts {unique_classes}/{num_classes} classes (healthy diversity)")
    
    return avg_loss, metrics, metrics_tracker


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    metrics,
    config,
    checkpoint_dir,
    logger,
    filename='checkpoint.pt'
):
    """Save checkpoint with error handling."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    checkpoint_path = checkpoint_dir / filename
    
    try:
        # Try to save checkpoint
        torch.save(checkpoint, checkpoint_path)
    except RuntimeError as e:
        # If save fails, clear GPU cache and retry
        logger.warning(f"Failed to save checkpoint: {e}")
        logger.warning("Clearing GPU cache and retrying...")
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Retry save with shorter filename if path too long
        if len(str(checkpoint_path)) > 255:  # Windows path limit
            short_name = f"ckpt_ep{epoch}.pt"
            checkpoint_path = checkpoint_dir / short_name
            logger.warning(f"Using shorter filename: {short_name}")
        
        torch.save(checkpoint, checkpoint_path)
        logger.info("Checkpoint saved successfully after retry")
    
    return str(checkpoint_path)


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    if args.num_epochs is not None:
        config['training']['num_epochs'] = int(args.num_epochs)
    
    # Override device if specified
    if args.device:
        config['compute']['device'] = args.device
    
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
    logger = setup_logger('train', config['logging']['log_dir'])
    logger.info(f"Starting training on {device}")
    logger.info(f"Configuration: {config}")
    
    # Setup TensorBoard
    if config['logging']['use_tensorboard']:
        tb_logger = TensorBoardLogger(
            config['logging']['log_dir'],
            experiment_name=args.experiment_name
        )
    else:
        tb_logger = None
    
    # Check if audio-only model
    audio_only = config['model'].get('model_type', 'multimodal') == 'audio_only'
    
    # Setup experiment tracker
    exp_tracker = ExperimentTracker(
        config['logging']['checkpoint_dir'],
        experiment_name=args.experiment_name
    )
    exp_tracker.log_config(config)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(config, logger, audio_only=audio_only)
    
    # Create model
    model = create_model(config, device, logger, audio_only=audio_only)
    
    # Create loss function
    loss_config = config['training'].get('loss', {})
    loss_type = loss_config.get('type', 'cross_entropy')
    gamma = loss_config.get('gamma', 2.0)
    label_smoothing = loss_config.get('label_smoothing', 0.0)
    
    # Use normalized and capped class weights as alpha for FocalLoss
    if loss_type == 'focal':
        # Normalize to sum = num_classes and cap extremes to avoid collapse
        num_classes = len(config['dataset']['emotions'])
        with torch.no_grad():
            norm_weights = class_weights / (class_weights.sum() + 1e-9) * num_classes
            # Cap to a reasonable range to prevent extreme bias in early epochs
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
    log_loss_contract_traceability(logger, loss_type, gamma, label_smoothing)
    
    # Create optimizer and scheduler (steps = optimizer updates per epoch * epochs)
    accum_steps = max(1, int(config['training'].get('gradient_accumulation_steps', 1)))
    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    num_training_steps = steps_per_epoch * int(config['training']['num_epochs'])
    if accum_steps > 1:
        logger.info(
            f"Gradient accumulation: {accum_steps} micro-batches per step; "
            f"~{steps_per_epoch} optimizer steps/epoch; total scheduler steps ≈ {num_training_steps}"
        )
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, config, num_training_steps, audio_only=audio_only, logger=logger
    )
    
    # Initialize mixed precision scaler
    scaler = None
    if config['training']['mixed_precision']:
        if config['compute']['device'] == 'cuda':
            # Use the recommended API for PyTorch 2.0+
            try:
                from torch.amp import GradScaler
                scaler = GradScaler('cuda')
            except (ImportError, TypeError):
                # Fallback for older PyTorch versions
                from torch.cuda.amp import GradScaler
                scaler = GradScaler()
        else:
            # CPU doesn't benefit from mixed precision in the same way
            scaler = None
    
    # Training loop
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
            scaler, device, config, epoch, logger, tb_logger, audio_only=audio_only
        )
        
        # Validate
        if epoch % config['logging']['val_every_n_epochs'] == 0:
            val_loss, val_metrics, val_metrics_tracker = validate(
                model, val_loader, criterion, device, config, logger, audio_only=audio_only
            )
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Train UA: {train_metrics['unweighted_accuracy']:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val UA: {val_metrics['unweighted_accuracy']:.4f}")
            
            if tb_logger:
                tb_logger.log_metrics(train_metrics, epoch, prefix='train/')
                tb_logger.log_metrics(val_metrics, epoch, prefix='val/')
                tb_logger.log_scalar('epoch/train_loss', train_loss, epoch)
                tb_logger.log_scalar('epoch/val_loss', val_loss, epoch)
            
            # Track metrics
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
            
            # Check if best model
            if val_metrics['unweighted_accuracy'] > best_val_ua:
                best_val_ua = val_metrics['unweighted_accuracy']
                logger.info(f"New best model! UA: {best_val_ua:.4f}")
                
                # Save best model
                best_checkpoint_path = save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics,
                    config, config['logging']['checkpoint_dir'], logger,
                    filename='best_model.pt'
                )
                
                exp_tracker.update_best_metrics(val_metrics)
                exp_tracker.save_checkpoint_info(best_checkpoint_path, epoch, val_metrics)
                
                # Reset patience
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"Patience: {patience_counter}/{config['training']['early_stopping_patience']}")
            
            # Step plateau scheduler based on validation metric (if using plateau scheduler)
            if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
                scheduler_metric = config['training'].get('scheduler_metric', 'val_loss')
                
                if scheduler_metric == 'val_loss':
                    metric_value = val_loss
                elif scheduler_metric == 'val_unweighted_accuracy':
                    # For accuracy, we want to maximize, so pass positive value
                    # The scheduler mode should be 'max' in config
                    metric_value = val_metrics['unweighted_accuracy']
                else:
                    # Fallback to val_loss
                    metric_value = val_loss
                
                # Get LR before scheduler step
                lr_before = [group['lr'] for group in optimizer.param_groups]
                
                # Get scheduler's internal best value (if available)
                scheduler_best = getattr(scheduler, 'best', None)
                scheduler_num_bad_epochs = getattr(scheduler, 'num_bad_epochs', None)
                
                # Step scheduler with metric
                scheduler.step(metric_value)
                
                # Get LR after scheduler step
                lr_after = [group['lr'] for group in optimizer.param_groups]
                
                # Check if LR was reduced
                lr_reduced = any(lr_before[i] > lr_after[i] for i in range(len(lr_before)))
                
                # Get updated scheduler state after step
                scheduler_best_after = getattr(scheduler, 'best', None)
                scheduler_num_bad_epochs_after = getattr(scheduler, 'num_bad_epochs', None)
                
                # Log current learning rates after scheduler step
                logger.info(f"Current learning rates: {lr_after}")
                if lr_reduced:
                    logger.info(f"Learning rate reduced! Before: {lr_before}, After: {lr_after}")
                    logger.info(f"Scheduler metric ({scheduler_metric}): {metric_value:.6f}")
                else:
                    # Log scheduler state for debugging
                    if scheduler_best is not None:
                        logger.info(f"Scheduler state - Metric: {metric_value:.6f}, Best: {scheduler_best:.6f}, Bad epochs: {scheduler_num_bad_epochs_after}/{scheduler.patience}")
                    else:
                        logger.info(f"Scheduler metric ({scheduler_metric}): {metric_value:.6f}")
                
                # Log to TensorBoard
                if tb_logger:
                    for idx, lr in enumerate(lr_after):
                        param_group_name = optimizer.param_groups[idx].get('name', f'group_{idx}')
                        tb_logger.log_scalar(f'train/lr_{param_group_name}', lr, epoch)
            
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
    
    # Load best checkpoint before final test evaluation (same logic as run_ablation.py)
    best_checkpoint_path = Path(config['logging']['checkpoint_dir']) / 'best_model.pt'
    if best_checkpoint_path.exists():
        logger.info(f"\nLoading best checkpoint from {best_checkpoint_path} for test evaluation...")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logger.info(f"Best model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        logger.warning("No best_model.pt found; evaluating with last-epoch weights.")
    
    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    test_loss, test_metrics, test_metrics_tracker = validate(
        model, test_loader, criterion, device, config, logger, audio_only=audio_only
    )
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    test_metrics_tracker.print_report()
    
    # Save confusion matrix
    cm_fig = test_metrics_tracker.plot_confusion_matrix(normalize=True)
    cm_fig.savefig(Path(config['logging']['checkpoint_dir']) / 'confusion_matrix_test.png')
    
    if tb_logger:
        tb_logger.log_metrics(test_metrics, 0, prefix='test/')
        tb_logger.log_figure('test/confusion_matrix', cm_fig, 0)
        tb_logger.close()
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()

