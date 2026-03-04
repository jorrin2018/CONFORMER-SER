"""
Utility functions for ablation studies.
"""

import torch
import torch.optim as optim
from typing import Dict, Any, List, Tuple
from src.models.conformer import ConformerMultimodalModel


def create_ablation_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any],
    study_number: int = 1
) -> optim.Optimizer:
    """
    Create optimizer for ablation studies with appropriate parameter groups.
    
    For Ablation 01, 05, 06, 07, 08 (unfreeze encoders), creates separate groups for:
    - Audio encoder parameters
    - Text encoder parameters  
    - Fusion/classification parameters
    
    For other studies, uses standard grouping.
    
    Args:
        model: Model instance
        config: Configuration dictionary
        study_number: Ablation study number (1-8)
        
    Returns:
        Optimizer with appropriate parameter groups
    """
    training_config = config.get('training', {})
    model_config = config.get('model', {})
    
    # Get learning rates
    lr_audio = float(training_config.get('learning_rate_audio', 1e-6))
    lr_text = float(training_config.get('learning_rate_text', 2e-5))
    lr_fusion = float(training_config.get('learning_rate_fusion', 5e-5))
    lr_classifier = float(training_config.get('learning_rate_classifier', lr_fusion))
    weight_decay = training_config.get('weight_decay', 0.1)
    adam_epsilon = float(training_config.get('adam_epsilon', 1e-8))

    studies_with_separate_lr = (
        1, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39
    )
    if study_number in studies_with_separate_lr:
        audio_params = []
        text_params = []
        fusion_params = []
        classifier_params = []

        use_classifier_lr = training_config.get('learning_rate_classifier') is not None

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'audio_encoder' in name:
                audio_params.append(param)
            elif 'text_encoder' in name:
                text_params.append(param)
            elif use_classifier_lr and (
                'classifier' in name or 'fusion.classifier' in name or '_aux' in name
            ):
                classifier_params.append(param)
            else:
                fusion_params.append(param)

        param_groups = []
        if audio_params:
            param_groups.append({'params': audio_params, 'lr': lr_audio, 'weight_decay': weight_decay})
        if text_params:
            param_groups.append({'params': text_params, 'lr': lr_text, 'weight_decay': weight_decay})
        if fusion_params:
            param_groups.append({'params': fusion_params, 'lr': lr_fusion, 'weight_decay': weight_decay})
        if classifier_params:
            param_groups.append({
                'params': classifier_params,
                'lr': lr_classifier,
                'weight_decay': weight_decay
            })

        optimizer = optim.AdamW(param_groups, eps=adam_epsilon)
    else:
        # Other studies: Standard grouping (fusion only, encoders frozen)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        optimizer = optim.AdamW(
            trainable_params,
            lr=lr_fusion,
            weight_decay=weight_decay,
            eps=adam_epsilon
        )
    
    return optimizer

