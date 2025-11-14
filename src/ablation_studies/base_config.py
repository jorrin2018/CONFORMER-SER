"""
Load and adjust base configuration from checkpoint 20251029_190659.
"""

import json
from pathlib import Path
from typing import Dict, Any


def load_base_config() -> Dict[str, Any]:
    """
    Load base configuration from checkpoint 20251029_190659.
    
    Adjusts:
    - num_classes from 10 to 7 (removes fearful, surprised, other)
    - emotions list to match 7 classes
    
    Returns:
        Dictionary with base configuration
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
    
    # Adjust emotions: remove fearful, surprised, other
    original_emotions = config['dataset']['emotions']
    new_emotions = [
        'angry', 'happy', 'sad', 'neutral', 
        'frustrated', 'excited', 'disgusted'
    ]
    
    config['dataset']['emotions'] = new_emotions
    config['model']['num_classes'] = 7
    
    return config

