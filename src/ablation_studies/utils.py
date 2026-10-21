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
    weight_decay = float(training_config.get('weight_decay', 0.1))
    adam_epsilon = float(training_config.get('adam_epsilon', 1e-8))

    _no_decay_keywords = (
        '.bias', 'LayerNorm.weight', 'LayerNorm.bias',
        'layer_norm.weight', 'layer_norm.bias'
    )

    def _split_by_decay(named_params):
        decay = []
        no_decay = []
        for name, param in named_params:
            if not param.requires_grad:
                continue
            if any(nd in name for nd in _no_decay_keywords):
                no_decay.append(param)
            else:
                decay.append(param)
        return decay, no_decay

    studies_with_separate_lr = (
        1, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        53,
        54,
    )
    if study_number in studies_with_separate_lr:
        audio_named = []
        text_named = []
        fusion_named = []
        classifier_named = []

        use_classifier_lr = training_config.get('learning_rate_classifier') is not None

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'audio_encoder' in name or name == 'wavlm_layer_logits':
                audio_named.append((name, param))
            elif 'text_encoder' in name:
                text_named.append((name, param))
            elif use_classifier_lr and (
                'classifier' in name or 'fusion.classifier' in name or '_aux' in name
            ):
                classifier_named.append((name, param))
            else:
                fusion_named.append((name, param))

        audio_decay, audio_no_decay = _split_by_decay(audio_named)
        text_decay, text_no_decay = _split_by_decay(text_named)
        fusion_decay, fusion_no_decay = _split_by_decay(fusion_named)
        cls_decay, cls_no_decay = _split_by_decay(classifier_named)

        param_groups = []
        if audio_decay:
            param_groups.append({'params': audio_decay, 'lr': lr_audio, 'weight_decay': weight_decay})
        if audio_no_decay:
            param_groups.append({'params': audio_no_decay, 'lr': lr_audio, 'weight_decay': 0.0})
        if text_decay:
            param_groups.append({'params': text_decay, 'lr': lr_text, 'weight_decay': weight_decay})
        if text_no_decay:
            param_groups.append({'params': text_no_decay, 'lr': lr_text, 'weight_decay': 0.0})
        if fusion_decay:
            param_groups.append({'params': fusion_decay, 'lr': lr_fusion, 'weight_decay': weight_decay})
        if fusion_no_decay:
            param_groups.append({'params': fusion_no_decay, 'lr': lr_fusion, 'weight_decay': 0.0})
        if cls_decay:
            param_groups.append({'params': cls_decay, 'lr': lr_classifier, 'weight_decay': weight_decay})
        if cls_no_decay:
            param_groups.append({'params': cls_no_decay, 'lr': lr_classifier, 'weight_decay': 0.0})

        optimizer = optim.AdamW(param_groups, eps=adam_epsilon)
    else:
        # Other studies: Standard grouping (fusion only, encoders frozen)
        all_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        decay_params, no_decay_params = _split_by_decay(all_named)

        param_groups = []
        if decay_params:
            param_groups.append({'params': decay_params, 'lr': lr_fusion, 'weight_decay': weight_decay})
        if no_decay_params:
            param_groups.append({'params': no_decay_params, 'lr': lr_fusion, 'weight_decay': 0.0})

        if not param_groups:
            param_groups.append({'params': [p for p in model.parameters() if p.requires_grad], 'lr': lr_fusion, 'weight_decay': weight_decay})

        optimizer = optim.AdamW(param_groups, eps=adam_epsilon)
    
    return optimizer

