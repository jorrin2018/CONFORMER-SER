"""Load and adjust base configuration from a reference checkpoint.

Used by ablation studies that need a 4-class (angry, happy, sad, neutral) setup
derived from an existing checkpoint config.
"""

import json
from pathlib import Path
from typing import Dict, Any


def load_base_config() -> Dict[str, Any]:
    """Load base configuration from checkpoint 20251029_190659 and adapt to 4 classes.

    Adjusts num_classes to 4 and emotions to ['angry', 'happy', 'sad', 'neutral'].
    Fails if the checkpoint config path does not exist.

    Returns:
        Dictionary with base configuration (dataset, model, etc.).

    Raises:
        FileNotFoundError: If the base checkpoint config file is missing.
    """
    # Path to base checkpoint config
    base_config_path = Path("checkpoints/20251029_190659/config.json")
    
    if not base_config_path.exists():
        raise FileNotFoundError(
            f"Base config not found at {base_config_path}. "
            "Please ensure checkpoint 20251029_190659 exists."
        )
    
    # Load base config
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Adjust emotions to the standard 4-class setup:
    # angry, happy (happy+excited), sad, neutral
    original_emotions = config['dataset']['emotions']
    new_emotions = ['angry', 'happy', 'sad', 'neutral']

    config['dataset']['emotions'] = new_emotions
    config['model']['num_classes'] = 4
    
    return config

