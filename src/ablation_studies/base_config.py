"""Load and adjust base configuration for ablation studies.

Used by studies that build on a reference training setup (4-class IEMOCAP:
angry, happy, sad, neutral). The base can come from a checkpoint ``config.json``,
the project ``config/config.yaml``, or paths set via environment variables.

Paths that are not absolute are resolved first against the **repository root**
(the parent of ``src/``), so running ``python scripts/run_ablation.py`` from
``scripts/`` still finds ``checkpoints/`` and ``config/`` at the project root.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml

# Repository root: .../src/ablation_studies/base_config.py -> parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_CHECKPOINT_REL = Path("checkpoints/20251029_190659/config.json")
_PROJECT_CONFIG_REL = Path("config/config.yaml")


def _apply_four_class_setup(config: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure dataset emotions and num_classes match the standard 4-class ablation setup."""
    if "dataset" not in config:
        config["dataset"] = {}
    if "model" not in config:
        config["model"] = {}
    config["dataset"]["emotions"] = ["angry", "happy", "sad", "neutral"]
    config["model"]["num_classes"] = 4
    return config


def _resolve_existing_path(p: Path) -> Path | None:
    """Return p if it exists as file, else None."""
    return p if p.is_file() else None


def _candidate_from_env_file(raw: str) -> Path | None:
    """Resolve ABLATION_BASE_CONFIG to an existing file."""
    path = Path(raw.strip())
    if not path.is_absolute():
        for base in (Path.cwd(), _REPO_ROOT):
            hit = _resolve_existing_path((base / path).resolve())
            if hit is not None:
                return hit
    return _resolve_existing_path(path.resolve())


def _candidate_from_env_checkpoint(raw: str) -> Path | None:
    """Resolve ABLATION_BASE_CHECKPOINT to config.json (file, dir, or id under checkpoints/)."""
    s = raw.strip()
    path = Path(s)
    if path.is_absolute():
        if path.is_file() and path.name == "config.json":
            return path
        if path.is_dir():
            return _resolve_existing_path(path / "config.json")
        return None
    # Relative: try cwd, then repo checkpoints/<s>/config.json, then repo/<s>
    cj = Path.cwd() / path
    if cj.is_file() and cj.name == "config.json":
        return cj
    dir_cwd = Path.cwd() / path
    if dir_cwd.is_dir():
        hit = _resolve_existing_path(dir_cwd / "config.json")
        if hit is not None:
            return hit
    id_under_ckpt = _REPO_ROOT / "checkpoints" / s / "config.json"
    return _resolve_existing_path(id_under_ckpt)


def _resolve_base_config_path() -> Path:
    """Choose which base config file to load (JSON from checkpoint or YAML)."""
    env_cfg = os.environ.get("ABLATION_BASE_CONFIG", "").strip()
    if env_cfg:
        hit = _candidate_from_env_file(env_cfg)
        if hit is None:
            raise FileNotFoundError(
                f"ABLATION_BASE_CONFIG={env_cfg!r} does not point to an existing file "
                f"(tried relative to cwd and repo root {_REPO_ROOT})."
            )
        return hit

    env_ckpt = os.environ.get("ABLATION_BASE_CHECKPOINT", "").strip()
    if env_ckpt:
        hit = _candidate_from_env_checkpoint(env_ckpt)
        if hit is None:
            raise FileNotFoundError(
                f"ABLATION_BASE_CHECKPOINT={env_ckpt!r} did not resolve to an existing "
                "config.json (file, directory with config.json, or checkpoints/<id>/config.json under repo root)."
            )
        return hit

    default_ckpt = _resolve_existing_path(_REPO_ROOT / _DEFAULT_CHECKPOINT_REL)
    if default_ckpt is not None:
        return default_ckpt

    fallback_yaml = _resolve_existing_path(_REPO_ROOT / _PROJECT_CONFIG_REL)
    if fallback_yaml is not None:
        return fallback_yaml

    raise FileNotFoundError(
        "No base ablation config found. Use one of:\n"
        "  • ABLATION_BASE_CONFIG: path to a config.json or .yaml\n"
        "  • ABLATION_BASE_CHECKPOINT: checkpoint folder name (e.g. 20251029_190659) or path to config.json\n"
        f"  • Place reference config at: {_REPO_ROOT / _DEFAULT_CHECKPOINT_REL}\n"
        f"  • Or keep the project default: {_REPO_ROOT / _PROJECT_CONFIG_REL}"
    )


def load_base_config() -> Dict[str, Any]:
    """Load base configuration and normalize to 4 emotion classes.

    Resolution order:
        1. ``ABLATION_BASE_CONFIG`` — explicit path to JSON or YAML.
        2. ``ABLATION_BASE_CHECKPOINT`` — ``config.json`` path, directory containing it,
           or checkpoint id under ``<repo>/checkpoints/<id>/``.
        3. Default ``<repo>/checkpoints/20251029_190659/config.json`` if present.
        4. Fallback ``<repo>/config/config.yaml`` if present.

    Returns:
        Deep-copied configuration dict with ``dataset.emotions`` and
        ``model.num_classes`` set for the 4-class setup.

    Raises:
        FileNotFoundError: If no candidate file exists.
        ValueError: If the file does not contain a JSON/YAML object at root.
    """
    path = _resolve_base_config_path()
    with open(path, encoding="utf-8") as f:
        if path.suffix.lower() in (".yaml", ".yml"):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Base config at {path} must be a mapping (object) at the root.")

    config = copy.deepcopy(config)
    return _apply_four_class_setup(config)
