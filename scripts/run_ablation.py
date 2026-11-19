"""Master script to run ablation studies (studies 1-50, 52-57).

Loads study-specific YAML config, builds the corresponding ablation model,
and runs training with the same training loop as src.train. Use --study N
to select the experiment (e.g. 34 for Conflict-Aware, 45 for MultiScale head,
55-57 for Ablation-45 recipe with alternate audio encoders).
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
import math
import warnings
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime
import copy
import gc
import shutil

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
    Ablation43Model,
    Ablation47Model,
    Ablation48Model,
    Ablation35Model,
    Ablation36AModel,
    Ablation36BModel,
    Ablation36CModel,
    Ablation39Model,
    create_ablation_optimizer
)
from src.ablation_studies.ablation_30_dirichlet import Ablation30Model
from src.utils.metrics import EmotionMetrics, compute_class_weights, plot_training_curves
from src.utils.logger import setup_logger, TensorBoardLogger, ExperimentTracker, print_model_summary
from src.utils.losses import create_loss_function, log_loss_contract_traceability
from src.train import (
    create_dataloaders,
    train_epoch,
    validate,
    detect_class_collapse,
)


def load_ablation_config(config_path: str, base_config: dict) -> dict:
    """Load ablation-specific YAML config and merge with base config.

    Ablation keys override base_config. Used to get dataset, model, and
    training settings for the selected study.

    Args:
        config_path: Path to the ablation YAML config file.
        base_config: Base configuration dict (e.g. from load_base_config).

    Returns:
        Merged configuration dictionary.
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
    elif study_number == 39:
        model = Ablation39Model(config)
    elif study_number == 40:
        model = Ablation34Model(config)
    elif study_number == 41:
        # Ablation 41: WavLM-base + ViT + Conflict-Aware, K-Fold
        model = Ablation34Model(config)
    elif study_number == 42:
        model = Ablation34Model(config)
    elif study_number == 43:
        model = Ablation43Model(config)
    elif study_number in (66, 67, 68, 69, 70, 71, 73, 78, 80, 82, 84, 86):
        # Lineas base unimodales con el mismo regimen que el 45.
        from src.ablation_studies.unimodal_baseline import UnimodalBaselineModel
        model = UnimodalBaselineModel(config)
    elif study_number in (44, 45, 46, 55, 56, 57, 64, 65, 72, 74, 75, 76, 77, 79, 81, 83, 85):
        # Ablations 44-46, 55-57: Conflict-Aware Cross-Attention (trimodal) with alternative heads.
        # 55-57: mismo diseño que 45 (MultiScaleTemporal) con distinto audio_encoder (WavLM, HuBERT, Emotion2Vec).
        model = Ablation34Model(config)
    elif study_number == 47:
        # Ablation 47: Conflict-Aware + SupCon pre-fusión (2 views)
        model = Ablation47Model(config)
    elif study_number == 48:
        # Ablation 48: Conflict-Aware + SupCon pre-fusión (3 views: audio + text + vision)
        model = Ablation48Model(config)
    elif study_number == 49:
        # Ablation 49: Conflict-Aware + SupCon (3 views) + WavLM-base
        model = Ablation48Model(config)
    elif study_number == 50:
        # Ablation 50: Conflict-Aware + SupCon (3 views) + Emotion2Vec audio encoder
        model = Ablation48Model(config)
    elif study_number == 52:
        # Ablation 52: like 49 + no waveform aug + SpecAugment on mel for ViT
        model = Ablation48Model(config)
    elif study_number == 53:
        # Ablation 53: like run 174035 of 52 (freeze flags) + WavLM weighted hidden states
        model = Ablation48Model(config)
    elif study_number == 54:
        # Ablation 54: same as 53 but microsoft/wavlm-base-plus
        model = Ablation48Model(config)
    elif study_number == 58:
        # Ablation 58: MultiScaleTemporal head + SupCon (3 views) + Wav2Vec2
        model = Ablation48Model(config)
    elif study_number == 59:
        # Ablation 59: MultiScaleTemporal + WavLM + SpecAugment on ViT (no waveform aug)
        model = Ablation34Model(config)
    elif study_number == 60:
        # Ablation 60: SOTA Combo — WavLM + MultiScaleTemporal + SupCon + SpecAugment
        model = Ablation48Model(config)
    elif study_number == 61:
        # Ablation 61: SOTA Combo + CompositeLoss with neutral focus
        model = Ablation48Model(config)
    elif study_number == 62:
        # Ablation 62: SOTA Combo + attention-based audio pooling
        model = Ablation34Model(config)
    elif study_number == 63:
        # Ablation 63: Fine-tuned Study 45 (lower dropout, lower label smoothing)
        model = Ablation34Model(config)
    else:
        raise ValueError(f"Invalid study number: {study_number}. Must be 1-50 or 52-86 (study 51 does not exist).")
    
    model = model.to(device)
    print_model_summary(model, logger)
    
    return model


def _run_ablation_with_config(
    study_number: int,
    config: dict,
    device: str,
    experiment_name: str,
    seed_override: int = None,
    resume: bool = False
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
    
    # Set random seed (allow CLI override)
    seed = seed_override if seed_override is not None else config['seed']
    config['seed'] = seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
    
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
        class_weights=class_weights if loss_type in ('cross_entropy', 'composite_loss') else None,
        alpha=alpha,
        gamma=gamma,
        label_smoothing=label_smoothing,
        device=device,
        num_classes=num_classes,
        loss_config=loss_config if loss_type == 'composite_loss' else None,
    )
    log_loss_contract_traceability(logger, loss_type, gamma, label_smoothing)
    
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
    accum_steps = max(1, int(config['training'].get('gradient_accumulation_steps', 1)))
    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    num_training_steps = steps_per_epoch * int(config['training']['num_epochs'])
    if accum_steps > 1:
        logger.info(
            f"Gradient accumulation: {accum_steps} micro-batches/step; "
            f"scheduler total steps ≈ {num_training_steps}"
        )

    scheduler_type = config['training'].get('scheduler_type', 'cosine')
    warmup_ratio = float(config['training'].get('warmup_ratio', 0.0) or 0.0)
    if scheduler_type == 'plateau' and warmup_ratio > 0:
        msg = (
            "training.warmup_ratio is set but scheduler_type is 'plateau': "
            "warmup_ratio is ignored. Use cosine/linear for warmup."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
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
    elif scheduler_type == 'cosine_with_restarts':
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        sched_cfg = config['training'].get('scheduler_config', {})
        T_0 = sched_cfg.get('T_0', 15)
        T_mult = sched_cfg.get('T_mult', 2)
        eta_min = float(sched_cfg.get('eta_min', 1e-7))
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
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
    
    # Handle mixed precision: True/"fp16"/"fp8" (need GradScaler), "bf16" (no scaler), or False
    mixed_precision = config['training'].get('mixed_precision', False)
    use_loss_scaling = config['training'].get('use_loss_scaling', True)  # Default True for FP16, False for BF16

    # Initialize scaler when FP16/FP8 or when mixed_precision is True (treat as fp16)
    scaler = None
    if mixed_precision in ['fp16', 'fp8'] or mixed_precision is True:
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
    monitor_metric = config['training'].get('monitor_metric', 'val_ua')
    monitor_mode = config['training'].get('monitor_mode', 'max')
    def _get_monitor_value(metrics):
        if monitor_metric == 'val_wa':
            return metrics.get('weighted_accuracy', metrics.get('unweighted_accuracy', 0.0))
        return metrics.get('unweighted_accuracy', 0.0)
    best_val_metric = -1.0 if monitor_mode == 'max' else float('inf')
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_metrics_list = []
    val_metrics_list = []

    start_epoch = 1
    resume_points = []
    if resume:
        restaurado = _load_resume_state(
            checkpoint_dir, experiment_name, model, optimizer, scheduler,
            device, monitor_metric, monitor_mode, logger, config_actual=config)
        if restaurado is not None:
            (start_epoch, best_val_metric, patience_counter,
             resume_points, historias) = restaurado
            train_losses, val_losses, train_metrics_list, val_metrics_list = historias
        else:
            logger.info("[reanudacion] no hay checkpoint previo; se entrena desde cero.")

    epocas_previas = start_epoch - 1     # para no reportar una epoca final inventada
    if start_epoch > config['training']['num_epochs']:
        logger.info("El fold ya habia agotado las epocas; se pasa directo a evaluar en test.")
    if patience_counter >= config['training']['early_stopping_patience']:
        logger.info("El fold ya habia disparado early stopping; se pasa directo a evaluar en test.")
        start_epoch = config['training']['num_epochs'] + 1

    logger.info("Starting training...")
    
    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
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
        train_metrics_list.append(_get_monitor_value(train_metrics))

        # Validate
        if epoch % config['logging']['val_every_n_epochs'] == 0:
            val_loss, val_metrics, val_metrics_tracker = validate(
                model, val_loader, criterion, device, config, logger, audio_only=False
            )

            val_losses.append(val_loss)
            val_metrics_list.append(_get_monitor_value(val_metrics))
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Train {monitor_metric}: {train_metrics_list[-1]:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val {monitor_metric}: {val_metrics_list[-1]:.4f}")
            
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
            
            # Check if best model (by monitor_metric)
            val_monitor = _get_monitor_value(val_metrics)
            is_best = (monitor_mode == 'max' and val_monitor > best_val_metric) or (
                monitor_mode == 'min' and val_monitor < best_val_metric
            )
            if is_best:
                best_val_metric = val_monitor
                logger.info(f"New best model! {monitor_metric}: {best_val_metric:.4f}")
                
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

            # Step plateau scheduler only (step-based schedulers step inside train_epoch)
            if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
                scheduler_metric = config['training'].get('scheduler_metric', 'val_loss')
                if scheduler_metric == 'val_loss':
                    metric_value = val_loss
                elif scheduler_metric in ('val_ua', 'val_unweighted_accuracy'):
                    metric_value = val_metrics.get('unweighted_accuracy', val_loss)
                elif scheduler_metric == 'val_wa':
                    metric_value = val_metrics.get('weighted_accuracy', val_metrics.get('unweighted_accuracy', val_loss))
                else:
                    metric_value = val_loss
                scheduler.step(metric_value)

            # Estado completo de ESTA epoca, mejore o no, y DESPUES de que el
            # scheduler haya dado su paso: best_model.pt solo se escribe al
            # mejorar, asi que por si solo no sirve para reanudar -- tras una
            # caida se perderian hasta 'patience' epocas y el LR volveria atras.
            _save_resume_state(
                _resume_state_path(checkpoint_dir, experiment_name),
                epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler,
                best_val_metric=best_val_metric, patience_counter=patience_counter,
                resume_points=resume_points, config=config, logger=logger)

            # Early stopping
            if patience_counter >= config['training']['early_stopping_patience']:
                logger.info("Early stopping triggered!")
                break
    
    # Plot training curves
    ultima_epoca = locals().get('epoch', epocas_previas)
    if len(train_losses) > 0:
        fig = plot_training_curves(
            train_losses, val_losses,
            train_metrics_list, val_metrics_list,
            metric_name='Unweighted Accuracy'
        )
        if tb_logger:
            tb_logger.log_figure('training/curves', fig, ultima_epoca)
        fig.savefig(Path(checkpoint_dir) / experiment_name / 'training_curves.png')

    # Load best checkpoint so test evaluation uses the best model, not the last epoch
    best_checkpoint_path = os.path.join(checkpoint_dir, experiment_name, 'best_model.pt')
    if os.path.isfile(best_checkpoint_path):
        logger.info(f"Loading best model from {best_checkpoint_path} for test evaluation...")
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        logger.warning("No best_model.pt found; evaluating on last-epoch weights.")

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

    # Alias for aggregation/LOSO code that expects best_val_ua
    best_val_ua = float(best_val_metric)

    # Save final results for this run
    results_dir = Path(checkpoint_dir) / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'final_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'test_metrics': test_metrics,
            'best_val_ua': best_val_ua,
            'final_epoch': ultima_epoca,
            # Un fold reanudado no es bit-identico a uno corrido de un tiron.
            # Queda marcado para poder rehacerlo limpio si hace falta.
            'resumed': bool(resume_points),
            'resume_points': resume_points,
        }, f, indent=2)

    logger.info(f"Ablation Study {study_number} run completed!")
    logger.info(f"Results saved to: {results_dir}")

    # Return key metrics for potential aggregation (e.g., LOSO)
    test_ua = test_metrics.get('unweighted_accuracy', None)
    test_wa = test_metrics.get('weighted_accuracy', None)
    return {
        'best_val_ua': best_val_ua,
        'test_unweighted_accuracy': float(test_ua) if test_ua is not None else None,
        'test_weighted_accuracy': float(test_wa) if test_wa is not None else None,
        'final_epoch': int(ultima_epoca),
        'resumed': bool(resume_points),
        'resume_points': resume_points,
        'results_dir': str(results_dir)
    }


def run_ablation_study(
    study_number: int,
    config_path: str,
    device: str = 'cuda',
    experiment_name: str = None,
    seed: int = None
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
    
    _run_ablation_with_config(study_number, config, device, experiment_name, seed_override=seed)


def run_ablation_study_loso(
    study_number: int,
    config_path: str,
    device: str = 'cuda',
    experiment_name: str = None,
    seed: int = None
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
            experiment_name=fold_experiment_name,
            seed_override=seed
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


# ---------------------------------------------------------------------------
# Reanudacion a mitad de fold
# ---------------------------------------------------------------------------
# Un fold cuesta 3-4.5 h. Sin esto, cualquier caida (reset del driver, reinicio
# de Windows, corte de luz) devuelve el fold a la epoca 0. El checkpoint ya
# guardaba modelo, optimizador y scheduler; lo que faltaba era guardarlo TODAS
# las epocas -- no solo al mejorar -- y poder volver a entrarle.
#
# AVISO METODOLOGICO: un fold reanudado NO es bit-identico a uno corrido de un
# tiron. El orden de shuffle del DataLoader arranca de nuevo en la epoca de
# reanudacion, asi que la semilla ya no reproduce ese fold exactamente. Por eso
# cada reanudacion queda registrada en 'resume_points' y el fold se marca con
# 'resumed': True en su final_results.json. Un fold marcado asi se puede volver
# a correr limpio si el revisor lo pide; lo que no se puede es ocultarlo.

def _resume_state_path(checkpoint_dir, experiment_name):
    return os.path.join(checkpoint_dir, experiment_name, 'last_model.pt')


def _scheduler_state(scheduler):
    if scheduler is None:
        return None
    try:
        return {'kind': 'state_dict', 'data': scheduler.state_dict()}
    except Exception:
        return {'kind': 'plateau_attrs', 'data': {
            'best': getattr(scheduler, 'best', None),
            'num_bad_epochs': getattr(scheduler, 'num_bad_epochs', None),
            'cooldown_counter': getattr(scheduler, 'cooldown_counter', None),
        }}


def _restore_scheduler(scheduler, state, logger):
    if scheduler is None or not state:
        return
    try:
        if state.get('kind') == 'state_dict':
            scheduler.load_state_dict(state['data'])
        else:
            for k, v in (state.get('data') or {}).items():
                if v is not None:
                    setattr(scheduler, k, v)
    except Exception as e:
        logger.warning(f"No se pudo restaurar el scheduler ({e}); sigue con su estado inicial.")


def _save_resume_state(path, *, epoch, model, optimizer, scheduler, best_val_metric,
                       patience_counter, resume_points, config, logger):
    """Guarda el estado completo de la epoca. Escritura atomica: si la maquina
    muere a mitad del torch.save, el last_model.pt anterior sigue intacto."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state': _scheduler_state(scheduler),
        'best_val_metric': best_val_metric,
        'patience_counter': patience_counter,
        'resume_points': resume_points,
        'config': config,
    }
    try:
        torch.save(data, tmp)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning(f"No se pudo guardar el estado de reanudacion: {e}")
        try:
            os.remove(tmp)
        except OSError:
            pass


def _rebuild_histories(checkpoint_dir, experiment_name, last_epoch, monitor_metric):
    """Reconstruye las curvas de entrenamiento desde metrics.json.

    Solo alimentan la grafica final, pero sin esto un fold reanudado pierde la
    mitad de su curva y la figura del articulo mentiria por omision.
    """
    acc_key = 'weighted_accuracy' if monitor_metric == 'val_wa' else 'unweighted_accuracy'
    tl, vl, tm, vm = [], [], [], []
    mpath = os.path.join(checkpoint_dir, experiment_name, 'metrics.json')
    if not os.path.isfile(mpath):
        return tl, vl, tm, vm
    try:
        hist = json.load(open(mpath))
    except Exception:
        return tl, vl, tm, vm
    for e in range(1, last_epoch + 1):
        row = hist.get(str(e))
        if not isinstance(row, dict):
            continue
        if 'train_loss' in row:
            tl.append(row['train_loss'])
            tm.append(row.get(f'train_{acc_key}', 0.0))
        if 'val_loss' in row:
            vl.append(row['val_loss'])
            vm.append(row.get(f'val_{acc_key}', 0.0))
    return tl, vl, tm, vm


def _config_experimental(cfg):
    """Los campos que definen el experimento. Si dos corridas coinciden en esto,
    una es continuacion de la otra; si no, reanudar mezclaria experimentos."""
    ds = (cfg or {}).get('dataset', {})
    return {
        'train_sessions': ds.get('train_sessions'),
        'val_sessions': ds.get('val_sessions'),
        'test_sessions': ds.get('test_sessions'),
        'val_speaker': ds.get('val_speaker'),
        'val_fraction': ds.get('val_fraction'),
        'model': (cfg or {}).get('model'),
        'seed': (cfg or {}).get('seed'),
    }


def _buscar_checkpoint_hermano(checkpoint_dir, experiment_name, config_actual, logger):
    """Checkpoint de una corrida anterior del MISMO fold, en otra carpeta.

    Devuelve (ruta, exacto) o None. 'exacto' es True para last_model.pt.
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    esperado = _config_experimental(config_actual)
    candidatos = []
    for nombre in sorted(os.listdir(checkpoint_dir)):
        if nombre == experiment_name:
            continue
        d = os.path.join(checkpoint_dir, nombre)
        if not os.path.isdir(d):
            continue
        for fichero, exacto in (('last_model.pt', True), ('best_model.pt', False)):
            ruta = os.path.join(d, fichero)
            if not os.path.isfile(ruta):
                continue
            cfg_prev = None
            cfg_path = os.path.join(d, 'config.json')
            if os.path.isfile(cfg_path):
                try:
                    cfg_prev = json.load(open(cfg_path))
                except Exception:
                    cfg_prev = None
            if cfg_prev is None or _config_experimental(cfg_prev) != esperado:
                logger.info(f"[reanudacion] se ignora {nombre}: su config no coincide "
                            f"con el de esta corrida")
                break
            if os.path.isfile(os.path.join(d, 'final_results.json')):
                logger.info(f"[reanudacion] se ignora {nombre}: ese fold ya termino")
                break
            candidatos.append((os.path.getmtime(ruta), ruta, exacto))
            break
    if not candidatos:
        return None
    _, ruta, exacto = max(candidatos)
    logger.warning(f"[reanudacion] no hay checkpoint en {experiment_name}; se continua "
                   f"la corrida anterior {os.path.basename(os.path.dirname(ruta))} "
                   f"({os.path.basename(ruta)})")
    return ruta, exacto


def _load_resume_state(checkpoint_dir, experiment_name, model, optimizer, scheduler,
                       device, monitor_metric, monitor_mode, logger,
                       config_actual=None):
    """Devuelve (start_epoch, best_val_metric, patience, resume_points, histories)
    o None si no hay nada que reanudar.

    Prefiere last_model.pt (estado exacto de la ultima epoca). Si solo existe
    best_model.pt -- una corrida anterior a este cambio -- reanuda desde ahi y
    reconstruye el mejor valor y la paciencia desde metrics.json.
    """
    base = os.path.join(checkpoint_dir, experiment_name)
    last_path = os.path.join(base, 'last_model.pt')
    best_path = os.path.join(base, 'best_model.pt')

    if os.path.isfile(last_path):
        path, exacto = last_path, True
    elif os.path.isfile(best_path):
        path, exacto = best_path, False
    else:
        # Sin --experiment_name el nombre del experimento lleva la marca de
        # tiempo del lanzamiento, asi que cada relanzamiento estrena carpeta y
        # no encontraria nada que reanudar. Se busca en las carpetas hermanas
        # del mismo fold, que son corridas anteriores de ESTE fold, y se exige
        # que el config coincida en lo que define el experimento.
        hermano = _buscar_checkpoint_hermano(checkpoint_dir, experiment_name, config_actual, logger)
        if hermano is None:
            return None
        path, exacto = hermano
        base = os.path.dirname(path)

    try:
        ck = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        logger.warning(f"No se pudo leer {path} para reanudar ({e}); se entrena desde cero.")
        return None

    model.load_state_dict(ck['model_state_dict'], strict=True)
    if 'optimizer_state_dict' in ck:
        try:
            optimizer.load_state_dict(ck['optimizer_state_dict'])
        except Exception as e:
            logger.warning(f"No se pudo restaurar el optimizador ({e}); sigue desde su estado inicial.")
    _restore_scheduler(scheduler, ck.get('scheduler_state'), logger)

    done = int(ck.get('epoch', 0))
    resume_points = list(ck.get('resume_points') or [])
    resume_points.append({'restored_from': os.path.basename(path),
                          'exact': exacto,
                          'resumed_after_epoch': done})

    if exacto and ck.get('best_val_metric') is not None:
        best_val_metric = ck['best_val_metric']
        patience = int(ck.get('patience_counter', 0))
    else:
        # Reconstruir desde metrics.json: el best_model.pt es de la mejor epoca,
        # no de la ultima, asi que la paciencia hay que recontarla.
        best_val_metric = -1.0 if monitor_mode == 'max' else float('inf')
        patience = 0
        acc_key = 'weighted_accuracy' if monitor_metric == 'val_wa' else 'unweighted_accuracy'
        mpath = os.path.join(base, 'metrics.json')
        ultima = done
        if os.path.isfile(mpath):
            try:
                hist = json.load(open(mpath))
                epochs = sorted(int(k) for k in hist)
                ultima = max(epochs) if epochs else done
                for e in epochs:
                    v = hist[str(e)].get(f'val_{acc_key}')
                    if v is None:
                        continue
                    mejor = (monitor_mode == 'max' and v > best_val_metric) or \
                            (monitor_mode == 'min' and v < best_val_metric)
                    if mejor:
                        best_val_metric, patience = v, 0
                    else:
                        patience += 1
            except Exception as e:
                logger.warning(f"No se pudo reconstruir la paciencia desde metrics.json ({e}).")
        logger.warning(
            f"Reanudando desde best_model.pt (epoca {done}); las epocas {done + 1}-{ultima} "
            f"de la corrida anterior se repiten porque sus pesos no se guardaron.")
        done = min(done, ultima)

    # El mejor checkpoint tiene que quedar en la carpeta de ESTA corrida: al final
    # se evalua en test cargando <experiment_name>/best_model.pt. Si se reanuda con
    # la paciencia ya avanzada y no se vuelve a mejorar, esa carpeta no tendria
    # best_model.pt y el test se haria con los pesos de la ultima epoca -- un
    # numero peor que el real, sin ningun aviso.
    destino_dir = os.path.join(checkpoint_dir, experiment_name)
    destino_best = os.path.join(destino_dir, 'best_model.pt')
    origen_best = os.path.join(base, 'best_model.pt')
    if not os.path.isfile(destino_best) and os.path.isfile(origen_best):
        try:
            os.makedirs(destino_dir, exist_ok=True)
            shutil.copy2(origen_best, destino_best)
            logger.info(f"[reanudacion] se copia el mejor checkpoint previo a "
                        f"{experiment_name}/best_model.pt para la evaluacion en test")
        except Exception as e:
            logger.warning(f"No se pudo copiar el mejor checkpoint previo ({e}). "
                           f"REVISA que el test no acabe evaluando la ultima epoca.")

    histories = _rebuild_histories(os.path.dirname(base), os.path.basename(base),
                                   done, monitor_metric)
    logger.info(f"[reanudacion] {os.path.basename(path)}: se reanuda en la epoca {done + 1}, "
                f"mejor {monitor_metric}={best_val_metric:.4f}, paciencia {patience}")
    return done + 1, best_val_metric, patience, resume_points, histories


def _clear_cuda_cache(quiet: bool = False) -> bool:
    """Libera cache de CUDA. Devuelve False si el contexto esta muerto.

    Tras un 'CUDA error: unknown error' todas las llamadas al runtime siguen
    fallando hasta reiniciar el proceso. Esta funcion lo detecta en vez de
    propagar la excepcion y tumbar la agregacion de resultados.
    """
    gc.collect()
    if not torch.cuda.is_available():
        return True
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return True
    except RuntimeError as e:
        if not quiet:
            print(f"[aviso] CUDA no responde al limpiar cache: {e}")
        return False


def run_ablation_study_kfold(
    study_number: int,
    config_path: str,
    device: str = 'cuda',
    experiment_name: str = None,
    seed: int = None,
    protocol: str = 'standard',
    folds: list = None,
    resume: bool = False
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

    # Reanudacion: --folds 4 5 corre solo esos folds y FUSIONA con el resumen que ya
    # exista en disco, para no repetir folds que ya costaron GPU. El fold_idx sigue
    # siendo el indice global dentro de all_sessions, porque de el dependen el
    # hablante de validacion del protocolo 'speaker' y la semilla por fold.
    requested = [int(x) for x in folds] if folds else None
    if requested:
        desconocidas = [s for s in requested if s not in all_sessions]
        if desconocidas:
            raise ValueError(f"--folds {desconocidas} no estan en all_sessions={all_sessions}")

    # 'fixed': el corpus trae un reparto fijo (MELD: train/dev/test) y NO hay nada
    # que rotar. Se corre UN solo fold con el reparto que diga el config, y la
    # variabilidad se mide con varias semillas en vez de entre folds. Se pasa por
    # el mismo camino que el k-fold a proposito, para conservar el sello de semilla,
    # la reanudacion y el formato del resumen.
    if protocol == 'fixed':
        fijos = dataset_cfg.get('test_sessions') or []
        if len(fijos) != 1:
            raise ValueError(
                "--protocol fixed necesita exactamente una sesion en "
                f"dataset.test_sessions; hay {fijos!r}.")
        sesiones_a_rotar = [int(fijos[0])]
    else:
        sesiones_a_rotar = list(all_sessions)

    for fold_idx, test_session in enumerate(sesiones_a_rotar):
        if requested and test_session not in requested:
            print(f"\n[skip] fold {fold_idx + 1}/{len(sesiones_a_rotar)} (test S{test_session}) "
                  f"-- no esta en --folds")
            continue
        print(f"\n{'='*60}")
        print(f"K-Fold Fold {fold_idx + 1}/{len(sesiones_a_rotar)}")
        print(f"{'='*60}")
        
        # Clear cache before starting new fold
        print("Clearing cache before fold...")
        _clear_cuda_cache()
        print("Cache cleared.")
        
        # -------------------------------------------------- reparto de sesiones
        # 'standard': LOSO como lo usa la literatura de IEMOCAP. Se entrena con las
        #   CUATRO sesiones que no son test y la validacion sale de UN hablante de
        #   dentro, rotando por fold. Sigue siendo independiente de hablante en los
        #   tres conjuntos y aprovecha ~87 % de las cuatro sesiones para entrenar.
        # 'legacy': 3 sesiones de entrenamiento + 1 de validacion + 1 de test. Es lo
        #   que se corrio hasta el 12/09/2026; usa un 25 % menos de datos y NO es
        #   comparable con las cifras publicadas.
        val_speaker = None
        val_fraction = None
        random_folds = None
        if protocol == 'fixed':
            # Reparto fijo del corpus. No se infiere nada: lo que diga el YAML.
            train_sessions = [int(s) for s in dataset_cfg.get('train_sessions', [])]
            val_cfg = [int(s) for s in dataset_cfg.get('val_sessions', [])]
            if not train_sessions or len(val_cfg) != 1:
                raise ValueError(
                    "--protocol fixed necesita dataset.train_sessions no vacio y "
                    f"exactamente una dataset.val_sessions; hay {train_sessions!r} y "
                    f"{val_cfg!r}.")
            if test_session in train_sessions or test_session in val_cfg:
                raise ValueError(
                    f"--protocol fixed: la sesion de test {test_session} tambien "
                    f"aparece en train={train_sessions} o val={val_cfg}. El reparto "
                    f"fijo tiene que ser disjunto.")
            val_session = val_cfg[0]
        elif protocol == 'random':
            # Reparto ALEATORIO estratificado sobre las cinco sesiones (Yoon et al.
            # 2018). DEPENDIENTE DE HABLANTE a proposito: los mismos actores aparecen
            # en train y test. No es el numero honesto; existe para medir, con el mismo
            # modelo y el mismo corpus, cuanto vale el reparto por si solo frente a
            # LOSO. Reportarlo SIEMPRE junto al 'standard', nunca solo.
            train_sessions = list(all_sessions)
            val_session = None
            random_folds = {'n_folds': len(all_sessions),
                            'fold_index': fold_idx,
                            'val_fraction': 0.06}
        elif protocol in ('standard', 'speaker'):
            train_sessions = [x for x in all_sessions if x != test_session]
            val_session = None
            if protocol == 'speaker':
                # Un hablante de la sesion siguiente a la de test (ciclico),
                # alternando genero por fold: cinco hablantes distintos.
                # Medido: la validacion de un solo hablante es muy ruidosa
                # (sigma de val UA 8.0 pp frente a 1.7) y para el entrenamiento
                # demasiado pronto. Se conserva como variante, no por defecto.
                nxt = all_sessions[(all_sessions.index(test_session) + 1) % len(all_sessions)]
                val_speaker = f"Ses{nxt:02d}_{'F' if fold_idx % 2 == 0 else 'M'}"
            else:
                # Muestra estratificada por clase dentro de las cuatro sesiones de
                # entrenamiento: todos los hablantes representados, senal de
                # validacion mucho mas estable para elegir el checkpoint.
                val_fraction = 0.12
        else:
            remaining = [x for x in all_sessions if x != test_session]
            val_session = remaining[0]
            train_sessions = [x for x in all_sessions
                              if x not in (test_session, val_session)]

        fold_config = copy.deepcopy(base_ablation_config)
        fold_config['dataset']['train_sessions'] = train_sessions
        fold_config['dataset']['val_sessions'] = (
            train_sessions if val_speaker else [val_session])
        fold_config['dataset']['test_sessions'] = [test_session]
        fold_config['dataset'].pop('val_speaker', None)
        fold_config['dataset'].pop('val_fraction', None)
        fold_config['dataset'].pop('random_folds', None)
        if random_folds:
            fold_config['dataset']['random_folds'] = random_folds
            fold_config['dataset']['test_sessions'] = list(all_sessions)
        if val_speaker:
            fold_config['dataset']['val_speaker'] = val_speaker
        if val_fraction:
            fold_config['dataset']['val_fraction'] = val_fraction
        
        # Create per-fold logging/checkpoint dirs to avoid collisions
        fold_suffix = f"fold_S{test_session}_test"
        fold_log_dir = os.path.join(fold_config['logging']['log_dir'], fold_suffix)
        fold_ckpt_dir = os.path.join(fold_config['logging']['checkpoint_dir'], fold_suffix)
        fold_config['logging']['log_dir'] = fold_log_dir
        fold_config['logging']['checkpoint_dir'] = fold_ckpt_dir
        
        fold_experiment_name = f"{base_experiment_name}_{fold_suffix}"
        
        print(f"Protocolo:      {protocol}")
        print(f"Train sessions: {train_sessions}")
        print("Validacion:     " + (
            f"hablante {val_speaker} (dentro de train)" if val_speaker
            else f"{100*val_fraction:.0f} % estratificado de train" if val_fraction
            else f"sesion {val_session}"))
        print(f"Test session:   {[test_session]}")
        print(f"Experiment:     {fold_experiment_name}")
        print()
        
        try:
            fold_metrics = _run_ablation_with_config(
                study_number=study_number,
                config=fold_config,
                device=device,
                experiment_name=fold_experiment_name,
                seed_override=seed,
                resume=resume
            )
            fold_metrics['test_session'] = int(test_session)
            fold_metrics['protocol'] = protocol
            fold_metrics['val_speaker'] = val_speaker
            fold_metrics['val_fraction'] = val_fraction
            fold_metrics['random_folds'] = random_folds
            fold_metrics['val_session'] = (
                int(val_session) if val_session is not None else None)
            fold_metrics['train_sessions'] = [int(s) for s in train_sessions]
            # Sello de procedencia. Sin esto, el rescate de folds del resumen previo
            # puede mezclar semillas distintas en una misma media sin avisar.
            fold_metrics['seed'] = int(seed) if seed is not None else None
            fold_metrics['experiment_name'] = fold_experiment_name
            fold_results.append(fold_metrics)
        except Exception as e:
            print(f"Error in fold {fold_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            # Soltar el traceback ANTES de recolectar. El objeto de excepcion
            # referencia todos los frames de la pila, y esos frames referencian el
            # modelo, el optimizador y los tensores del fold que acaba de fallar: con
            # `e` vivo, el gc.collect() de _clear_cuda_cache no libera nada y el fold
            # muerto se arrastra al siguiente. El 29/09/2026 tres de cinco folds de la
            # semilla 123 murieron con 'bad allocation' (memoria de HOST, no CUDA).
            e.__traceback__ = None
            del e
            _clear_cuda_cache(quiet=True)
            continue
        finally:
            # Always clear cache after each fold. Si el contexto de CUDA ha muerto
            # (error 'unknown error': reset del driver, TDR, GPU caida del bus),
            # empty_cache() vuelve a lanzar. Antes eso escapaba del bucle y se perdia
            # el resumen de los folds que SI habian terminado.
            print(f"\nClearing cache after fold {fold_idx + 1}...")
            if not _clear_cuda_cache():
                print("[aviso] el contexto de CUDA no responde: los folds restantes no "
                      "pueden correr en este proceso. Se guarda el resumen parcial y se "
                      "sale; reanuda con --folds sobre las sesiones que falten.")
                break
            print("Cache cleared.\n")
    
    # --- fusion con el resumen previo -------------------------------------------
    # Con --folds solo se han corrido algunas sesiones. Se recuperan del resumen en
    # disco las que no se pidieron, siempre que su protocolo coincida: mezclar folds
    # de protocolos distintos produciria una media sin sentido.
    ckpt_root = base_ablation_config['logging']['checkpoint_dir']
    # El nombre del resumen lleva la semilla cuando se pide una explicitamente. Sin
    # esto, correr 42/123/456 sobre el mismo estudio sobrescribe el resumen anterior
    # cada vez y la multi-semilla se destruye a si misma en silencio.
    sufijo_semilla = '' if seed is None else '_seed{}'.format(int(seed))
    summary_path = (Path(ckpt_root) /
                    'kfold_summary_ablation_{:02d}{}.json'.format(study_number,
                                                                  sufijo_semilla))
    if requested and summary_path.exists():
        try:
            previo = json.load(open(summary_path))
        except Exception as e:
            print(f"[aviso] no se pudo leer el resumen previo ({e}); se guarda solo lo corrido ahora")
            previo = None
        if previo:
            nuevas = {fr.get('test_session') for fr in fold_results}
            recuperados = []
            for fr in previo.get('folds', []):
                if fr.get('test_session') in nuevas:
                    continue                      # lo de ahora manda
                if fr.get('protocol') != protocol:
                    print(f"[aviso] fold S{fr.get('test_session')} del resumen previo es "
                          f"protocolo '{fr.get('protocol')}' y no '{protocol}': se descarta")
                    continue
                semilla_previa = fr.get('seed')
                semilla_actual = int(seed) if seed is not None else None
                if 'seed' in fr and semilla_previa != semilla_actual:
                    print(f"[aviso] fold S{fr.get('test_session')} del resumen previo "
                          f"se corrio con semilla {semilla_previa} y ahora se pide "
                          f"{semilla_actual}: se descarta. Mezclar semillas en una "
                          f"media la invalida.")
                    continue
                recuperados.append(fr)
            if recuperados:
                print(f"[reanudacion] se recuperan del resumen previo los folds "
                      f"{sorted(fr['test_session'] for fr in recuperados)}")
            fold_results = sorted(fold_results + recuperados,
                                  key=lambda fr: fr.get('test_session', 0))

    # Lo esperado son las sesiones que este protocolo ROTA, no todas las del corpus.
    # Con --protocol fixed solo hay una, y comparar contra all_sessions hacia creer
    # que faltaban dos folds y aconsejaba correr train y dev COMO TEST, que habria
    # producido numeros sin sentido.
    esperadas = set(int(s) for s in sesiones_a_rotar)
    if len(fold_results) < len(esperadas):
        faltan = sorted(esperadas - {fr.get('test_session') for fr in fold_results})
        print(f"\n[PARCIAL] {len(fold_results)}/{len(esperadas)} folds. Faltan {faltan}. "
              f"Reanuda con:  --folds {' '.join(str(s) for s in faltan)} --protocol {protocol}")
    if protocol == 'fixed':
        print("\n[protocol fixed] UN solo fold con el reparto del config: la sigma que "
              "sale abajo es 0 por construccion y NO mide nada.\n"
              "                 La variabilidad se mide repitiendo con otra semilla "
              "(--seed 123) y comparando\n"
              "                 contra la banda de +-0.86 pp de la seccion 7 de "
              "REWRITE_NOTES.")

    # Aggregate K-Fold results across folds
    best_val_uas = [fr['best_val_ua'] for fr in fold_results if fr.get('best_val_ua') is not None]
    test_uas = [fr['test_unweighted_accuracy'] for fr in fold_results if fr.get('test_unweighted_accuracy') is not None]
    test_was = [fr['test_weighted_accuracy'] for fr in fold_results if fr.get('test_weighted_accuracy') is not None]
    
    summary = {
        'folds': fold_results,
        'protocol': protocol,
        'num_folds': len(fold_results),
        # 'complete' se juzga contra las sesiones que ESTE protocolo rota. Contra
        # all_sessions, un resumen de --protocol fixed quedaria marcado incompleto
        # para siempre aunque su unico fold haya terminado.
        'complete': len(fold_results) == len(sesiones_a_rotar),
        'all_sessions': [int(s) for s in all_sessions],
        'sesiones_rotadas': [int(s) for s in sesiones_a_rotar],
        'seed': int(seed) if seed is not None else None,
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
                        help='Run specific study (1-50, 52-86; there is no study 51). If not specified, runs all.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name prefix')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config seed)')
    parser.add_argument('--protocol',
                        choices=['standard', 'speaker', 'legacy', 'random', 'fixed'],
                        default='standard',
                        help="Reparto de sesiones en k-fold. 'standard': 4 sesiones de "
                             "entrenamiento y validacion estratificada del 12%% de ellas. "
                             "'speaker': 4 sesiones y validacion de un hablante de dentro "
                             "(disjunta por hablante pero ruidosa: sigma de val UA 8 pp). "
                             "'legacy': 3+1+1, con 25%% menos datos y NO comparable con "
                             "las cifras publicadas. 'random': folds aleatorios sobre las "
                             "cinco sesiones al estilo de Yoon et al. 2018, DEPENDIENTE DE "
                             "HABLANTE -- solo para medir cuanto vale el reparto; nunca "
                             "como numero principal. 'fixed': UN solo fold con el "
                             "reparto que diga el config, para corpus con split fijo "
                             "como MELD (1=train, 2=dev, 3=test). Sin folds no hay sigma "
                             "entre folds: la variabilidad se mide con varias semillas.")
    parser.add_argument('--resume', action='store_true',
                        help='Reanudar cada fold desde su ultimo checkpoint en vez de '
                             'empezar en la epoca 0. Usa last_model.pt (estado exacto de '
                             'la ultima epoca); si solo hay best_model.pt de una corrida '
                             'anterior, reanuda desde ahi. AVISO: un fold reanudado no es '
                             'bit-identico a uno corrido de un tiron -- queda marcado con '
                             'resumed:true en su final_results.json.')
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                        help='Correr solo estos folds (por sesion de test), p. ej. '
                             '--folds 4 5. Los demas se recuperan del resumen previo '
                             'si su protocolo coincide. Para reanudar tras un fallo de '
                             'GPU sin repetir folds ya terminados.')
    parser.add_argument('--loso', '--kfold', action='store_true', dest='loso',
                        help='Force 5-fold leave-one-session-out for the requested '
                             'study, regardless of its number. Required for any '
                             'result that goes in a paper: a single split on '
                             'IEMOCAP has a between-fold sigma of 4-5 pp. '
                             '--kfold es un alias: el nombre de la bandera dice LOSO '
                             'pero el camino que toma es el de k-fold con limpieza de '
                             'cache entre folds, que es el que se usa desde el 12/09.')
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
        (40, "config/ablation/ablation_40_wavlm_conflict_aware.yaml"),  # WavLM-base + Conflict-Aware
        (41, "config/ablation/ablation_41_wavlm_base_vit_kfold.yaml"),  # WavLM-base + ViT + Conflict-Aware, K-Fold
        (42, "config/ablation/ablation_42_wavlm_base_vit.yaml"),  # WavLM-base + ViT + Conflict-Aware, single split
        (43, "config/ablation/ablation_43_training_improvements.yaml"),  # Focal + Contrastive + Augmentation
        (44, "config/ablation/ablation_44_transformer_head.yaml"),  # Conflict-Aware + Transformer Head
        (45, "config/ablation/ablation_45_multiscale_head.yaml"),  # Conflict-Aware + MultiScaleTemporal Head
        (46, "config/ablation/ablation_46_cnn1d_head.yaml"),  # Conflict-Aware + CNN1D Head
        (47, "config/ablation/ablation_47_supcon.yaml"),  # Conflict-Aware + SupCon pre-fusión (2 views)
        (48, "config/ablation/ablation_48_supcon_3views.yaml"),  # Conflict-Aware + SupCon (3 views: audio+text+vision)
        (49, "config/ablation/ablation_49_supcon_wavlm.yaml"),  # Conflict-Aware + SupCon (3 views) + WavLM-base
        (50, "config/ablation/ablation_50_emotion2vec.yaml"),  # Conflict-Aware + SupCon (3 views) + Emotion2Vec
        (52, "config/ablation/ablation_52_supcon_wavlm_specaug_vit.yaml"),  # Like 49: no wf aug, SpecAugment on ViT mel
        (53, "config/ablation/ablation_53_wavlm_weighted_hidden_states.yaml"),  # 52/174035-style + WavLM layer weights
        (54, "config/ablation/ablation_54_wavlm_base_plus_weighted.yaml"),  # Like 53 + wavlm-base-plus
        (55, "config/ablation/ablation_55_45_wavlm_multiscale.yaml"),  # Ablation 45 recipe + WavLM-base
        (56, "config/ablation/ablation_56_45_hubert_multiscale.yaml"),  # Ablation 45 recipe + HuBERT-base
        (57, "config/ablation/ablation_57_45_emotion2vec_multiscale.yaml"),  # Ablation 45 recipe + Emotion2Vec
        (58, "config/ablation/ablation_58_supcon_multiscale.yaml"),  # MultiScaleTemporal + SupCon (3 views)
        (59, "config/ablation/ablation_59_wavlm_multiscale_specaug.yaml"),  # WavLM + MultiScaleTemporal + SpecAugment
        (60, "config/ablation/ablation_60_sota_combo.yaml"),  # SOTA Combo: WavLM + MultiScaleTemporal + SupCon + SpecAugment
        (61, "config/ablation/ablation_61_composite_neutral.yaml"),  # SOTA Combo + CompositeLoss with neutral focus
        (62, "config/ablation/ablation_62_attention_pooling.yaml"),  # SOTA Combo + attention-based audio pooling
        (63, "config/ablation/ablation_63_finetuned_45.yaml"),  # Fine-tuned Study 45
        (64, "config/ablation/ablation_64_speechcue_real.yaml"),  # 45 + SpeechCue del articulo (F0 pYIN + terciles + 9 tokens)
        (65, "config/ablation/ablation_65_vit_unfrozen.yaml"),  # 45 con el ViT descongelado (un solo factor)
        (66, "config/ablation/ablation_66_audio_only.yaml"),  # linea base unimodal de audio, regimen del 45
        (67, "config/ablation/ablation_67_text_only.yaml"),  # linea base unimodal de texto, regimen del 45
        (68, "config/ablation/ablation_68_audio_weighted_layers.yaml"),  # 66 + suma ponderada de capas (Pepino)
        (69, "config/ablation/ablation_69_audio_simple_head.yaml"),  # 66 + cabeza simple: la MST son 10M param sobre un vector
        (70, "config/ablation/ablation_70_pepino_replication.yaml"),  # REPLICACION de Pepino 2021 (congelado + capas + cabeza simple)
        (71, "config/ablation/ablation_71_param_groups.yaml"),  # 66 con las tasas por grupo que el articulo describe
        (72, "config/ablation/ablation_72_trimodal_param_groups.yaml"),  # 45 con las tasas por grupo -- NUMERO PRINCIPAL
        (73, "config/ablation/ablation_73_text_param_groups.yaml"),  # 67 con las tasas por grupo
        (74, "config/ablation/ablation_74_no_conflict_gate.yaml"),  # 72 SIN compuerta de conflicto -- DECIDE EL TITULO
        (75, "config/ablation/ablation_75_conflict_gate_trained.yaml"),  # 72 + perdida auxiliar: la compuerta con detector que SI entrena
        (76, "config/ablation/ablation_76_confidence_calibrated.yaml"),  # 75 + objetivo de calibracion sobre la confianza por rama
        (77, "config/ablation/ablation_77_cremad_trimodal.yaml"),  # el modelo del 74, sin cambios, sobre CREMA-D (segundo corpus)
        (78, "config/ablation/ablation_78_cremad_audio_only.yaml"),  # solo audio sobre CREMA-D: el par del 77, replica del claim central
        (79, "config/ablation/ablation_79_meld_trimodal.yaml"),  # el modelo del 74, sin cambios, sobre MELD 4 clases (tercer corpus; --protocol fixed)
        (80, "config/ablation/ablation_80_meld_audio_only.yaml"),  # solo audio sobre MELD: el par del 79 en el regimen texto-dominante
        (81, "config/ablation/ablation_81_meld_trimodal_7class.yaml"),  # el 79 a 7 clases: la fila comparable con la literatura de MELD (--protocol fixed)
        (82, "config/ablation/ablation_82_meld_audio_only_7class.yaml"),  # el 80 a 7 clases: el par solo-audio de esa fila
        (83, "config/ablation/ablation_83_iemocap_trimodal_7class.yaml"),  # el 74 con las SIETE clases propias de IEMOCAP, excited separado de happy (--loso)
        (84, "config/ablation/ablation_84_iemocap_audio_only_7class.yaml"),  # el 71 a 7 clases: el par solo-audio del 83
        (85, "config/ablation/ablation_85_ravdess_trimodal.yaml"),  # el modelo del 74 sobre RAVDESS: replica del extremo texto-nulo (2 frases fijas)
        (86, "config/ablation/ablation_86_ravdess_audio_only.yaml"),  # solo audio sobre RAVDESS: el par del 85
    ]
    
    # Filter studies if specific one requested
    if args.study is not None:
        studies = [s for s in studies if s[0] == args.study]
        if not studies:
            print(f"Error: Study {args.study} not found. Valid studies: 1-50, 52-86 (there is no study 51)")
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
                    experiment_name=experiment_name,
                    seed=args.seed
                )
            elif args.loso or args.protocol == 'fixed' or study_number in (20, 29, 41):
                # Special case: Ablation 20, 29 and 41 run K-Fold cross-validation with cache clearing.
                # --protocol fixed entra aqui SIN pedir --loso: pedir las dos cosas a
                # la vez se lee como una contradiccion ("leave-one-session-out con
                # reparto fijo") y es una trampa de uso. Con reparto fijo el camino
                # correcto es este igualmente, porque es el que conserva el sello de
                # semilla, la reanudacion y el formato del resumen.
                run_ablation_study_kfold(
                    study_number=study_number,
                    config_path=config_path,
                    device=args.device,
                    experiment_name=experiment_name,
                    seed=args.seed,
                    protocol=args.protocol,
                    folds=args.folds,
                    resume=args.resume
                )
            else:
                run_ablation_study(
                    study_number=study_number,
                    config_path=config_path,
                    device=args.device,
                    experiment_name=experiment_name,
                    seed=args.seed
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

