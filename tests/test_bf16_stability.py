"""
Quick stability test for BFloat16 mixed precision on RTX 4060.
Runs 5 epochs of the smallest viable config to verify no NaN/Inf gradients.

Usage:
    python tests	est_bf16_stability.py --study 45 --epochs 5 --device cuda
"""

import os
import sys
import argparse
import torch
import yaml

# El repositorio es el padre de tests/: necesario para importar `scripts.*` y `src.*`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_ablation import (
    load_base_config,
    load_ablation_config,
    create_ablation_model,
    create_dataloaders,
)
from src.utils.losses import create_loss_function
from src.train import train_epoch, validate
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='BF16 stability smoke test')
    parser.add_argument('--study', type=int, default=45,
                        help='Study number to test (default: 45)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs for smoke test (default: 5)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for test')
    return parser.parse_args()


def main():
    args = parse_args()
    study = args.study
    device = args.device

    # Check BF16 support
    if device == 'cuda' and torch.cuda.is_available():
        if not torch.cuda.is_bf16_supported():
            print("[ERROR] This GPU does not support BFloat16. Aborting.")
            sys.exit(1)
        print(f"[OK] BF16 supported on {torch.cuda.get_device_name(0)}")
    else:
        print("[WARNING] Running on CPU or CUDA not available. BF16 test skipped.")
        sys.exit(0)

    # Load config
    base_config = load_base_config()
    config_path = f"config/ablation/ablation_{study:02d}_multiscale_head.yaml"
    if study == 58:
        config_path = "config/ablation/ablation_58_supcon_multiscale.yaml"
    elif study == 59:
        config_path = "config/ablation/ablation_59_wavlm_multiscale_specaug.yaml"

    if not os.path.exists(config_path):
        # fallback: try generic names
        candidates = [
            f"config/ablation/ablation_{study:02d}_multiscale_head.yaml",
            f"config/ablation/ablation_{study:02d}_wavlm_multiscale.yaml",
            f"config/ablation/ablation_{study:02d}_supcon_3views.yaml",
            f"config/ablation/ablation_{study}.yaml",
        ]
        for c in candidates:
            if os.path.exists(c):
                config_path = c
                break

    config = load_ablation_config(config_path, base_config)
    config['compute']['device'] = device
    config['training']['num_epochs'] = args.epochs
    config['training']['batch_size'] = args.batch_size
    config['training']['mixed_precision'] = 'bf16'
    config['training']['gradient_accumulation_steps'] = 2  # faster test
    config['logging']['log_dir'] = "./logs/bf16_test"
    config['logging']['checkpoint_dir'] = "./checkpoints/bf16_test"

    logger = setup_logger('bf16_test', config['logging']['log_dir'])
    logger.info("=" * 60)
    logger.info("BF16 Stability Smoke Test")
    logger.info(f"Study: {study}, Epochs: {args.epochs}, Device: {device}")
    logger.info("=" * 60)

    # Create dataloaders
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        config, logger, audio_only=False
    )

    # Create model
    model = create_ablation_model(study, config, device, logger)

    # Create optimizer
    from src.ablation_studies.utils import create_ablation_optimizer
    optimizer = create_ablation_optimizer(model, config, study)

    # Loss
    criterion = create_loss_function(
        loss_type=config['training']['loss'].get('type', 'cross_entropy'),
        device=device
    )

    # Scaler: BF16 does NOT need GradScaler
    scaler = None
    logger.info("[BF16] Using BFloat16 autocast without GradScaler")

    # Training loop (short)
    nan_detected = False
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n--- Test Epoch {epoch}/{args.epochs} ---")
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, None,
            scaler, device, config, epoch, logger, None, audio_only=False
        )
        logger.info(f"Train Loss: {train_loss:.4f}, UA: {train_metrics.get('unweighted_accuracy', 0):.4f}")

        if not torch.isfinite(torch.tensor(train_loss)):
            logger.error("[FAIL] NaN/Inf loss detected!")
            nan_detected = True
            break

    # Final verdict
    logger.info("\n" + "=" * 60)
    if nan_detected:
        logger.info("[RESULT] BF16 FAILED — NaN/Inf detected during training.")
        logger.info("Recommendation: Keep mixed_precision: false for this study.")
    else:
        logger.info("[RESULT] BF16 PASSED — No NaN/Inf after {} epochs.".format(args.epochs))
        logger.info("Recommendation: You can safely set mixed_precision: 'bf16' in your configs.")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
