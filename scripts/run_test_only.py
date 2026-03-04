"""
Run test evaluation only on an existing ablation checkpoint.
Loads config and best_model.pt from the checkpoint directory, runs validate on test set,
and writes final_results.json and confusion_matrix_test.png to the same directory.
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.train import create_dataloaders, validate
from src.utils.losses import create_loss_function
from src.utils.logger import setup_logger
from scripts.run_ablation import create_ablation_model


def main():
    parser = argparse.ArgumentParser(description="Run test evaluation on an ablation checkpoint")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the run directory or to best_model.pt (e.g. checkpoints/ablation_34/ablation_34_20251210_132712 or .../best_model.pt)",
    )
    parser.add_argument("--study", type=int, default=34, help="Ablation study number (default 34)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if checkpoint_dir.is_file():
        # User passed path to best_model.pt — use parent directory
        checkpoint_dir = checkpoint_dir.parent
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    best_pt = checkpoint_dir / "best_model.pt"
    config_path = checkpoint_dir / "config.json"
    if not best_pt.is_file():
        raise FileNotFoundError(f"best_model.pt not found in {checkpoint_dir}")
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found in {checkpoint_dir}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")

    # Load config (saved at training time)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    # Ensure required keys exist
    config.setdefault("evaluation", {})
    config["evaluation"].setdefault("batch_size", config.get("training", {}).get("batch_size", 8))
    config.setdefault("compute", {})
    config["compute"].setdefault("num_workers", 4)
    config["compute"].setdefault("pin_memory", True)
    config["compute"]["device"] = device

    # setup_logger(name, log_dir, log_file='training.log', level=...) — no 'verbose' argument
    logger = setup_logger(
        "test_only",
        log_dir=str(checkpoint_dir),
        log_file="test_eval.log",
    )

    # #region agent log
    try:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _logpath = os.path.join(_root, ".cursor", "debug.log")
        with open(_logpath, "a", encoding="utf-8") as _f:
            import time
            _f.write(json.dumps({"id": "log_setup_ok", "timestamp": int(time.time() * 1000), "location": "run_test_only.py:main", "message": "setup_logger succeeded", "data": {"checkpoint_dir": str(checkpoint_dir)}, "hypothesisId": "H1"}) + "\n")
    except Exception:
        pass
    # #endregion

    # Dataloaders (we only use test_loader)
    _, _, test_loader, class_weights = create_dataloaders(config, logger, audio_only=False)
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Model
    model = create_ablation_model(args.study, config, device, logger)
    checkpoint = torch.load(best_pt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    # Criterion for validate (only used for loss value)
    loss_config = config.get("training", {}).get("loss", {})
    loss_type = loss_config.get("type", "cross_entropy")
    criterion = create_loss_function(
        loss_type=loss_type,
        class_weights=class_weights if loss_type in ("cross_entropy", "composite_loss") else None,
        device=device,
        num_classes=config["model"]["num_classes"],
        loss_config=loss_config if loss_type == "composite_loss" else None,
    )

    # Run test
    logger.info("Evaluating on test set...")
    test_loss, test_metrics, test_metrics_tracker = validate(
        model, test_loader, criterion, device, config, logger, audio_only=False
    )
    logger.info(f"Test Loss: {test_loss:.4f}")
    test_metrics_tracker.print_report()

    # Best val metric and epoch from checkpoint
    best_val_ua = None
    if "metrics" in checkpoint:
        m = checkpoint["metrics"]
        monitor = config.get("training", {}).get("monitor_metric", "val_ua")
        if monitor == "val_wa":
            best_val_ua = m.get("weighted_accuracy", m.get("unweighted_accuracy"))
        else:
            best_val_ua = m.get("unweighted_accuracy")
    if best_val_ua is None and (checkpoint_dir / "best_metrics.json").is_file():
        with open(checkpoint_dir / "best_metrics.json") as f:
            bm = json.load(f)
        best_val_ua = bm.get("weighted_accuracy", bm.get("unweighted_accuracy"))
    final_epoch = checkpoint.get("epoch", 0)

    # Save final_results.json
    results = {
        "test_metrics": test_metrics,
        "best_val_ua": float(best_val_ua) if best_val_ua is not None else None,
        "final_epoch": final_epoch,
    }
    with open(checkpoint_dir / "final_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved {checkpoint_dir / 'final_results.json'}")

    # Save confusion matrix
    cm_fig = test_metrics_tracker.plot_confusion_matrix(normalize=True)
    cm_fig.savefig(checkpoint_dir / "confusion_matrix_test.png")
    logger.info(f"Saved {checkpoint_dir / 'confusion_matrix_test.png'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
