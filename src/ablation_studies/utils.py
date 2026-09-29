"""
Utility functions for ablation studies.
"""

import warnings

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

    # OJO -- ESTA LISTA DECIDE SI LAS TASAS POR GRUPO SE APLICAN O NO.
    #
    # Los estudios que NO estan aqui caen en la rama 'else', cuyo comentario dice
    # "encoders frozen". Esa suposicion es FALSA para los estudios 44, 45 y 66-69,
    # que descongelan 6 bloques de wav2vec2 y 4 de DistilBERT: en ellos TODO lo
    # entrenable -- encoders incluidos -- se entreno con un unico lr = lr_fusion =
    # 5e-6, y learning_rate_audio (3e-5) y learning_rate_text (2e-5) nunca se
    # aplicaron. Medido el 17/09/2026; ver docs/REWRITE_NOTES.md.
    #
    # No se anaden estudios a la lista a posteriori: eso cambiaria en silencio el
    # comportamiento de corridas ya hechas y las volveria irreproducibles. Para
    # pedir las tasas por grupo se usa la bandera explicita de abajo, que queda
    # escrita en el config.json de cada fold.
    studies_with_separate_lr = (
        1, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        53,
        54,
    )
    use_param_groups = bool(training_config.get('use_param_groups', False))
    if study_number in studies_with_separate_lr or use_param_groups:
        audio_params = []
        text_params = []
        fusion_params = []
        classifier_params = []

        use_classifier_lr = training_config.get('learning_rate_classifier') is not None

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # NOTA: existen DOS constructores de optimizador — este (solo para
            # studies_with_separate_lr) y create_optimizer() en src/train.py, que
            # agrupa por subcadena. 'audio_encoder_layer_logits' contiene
            # 'audio_encoder', asi que cae en el grupo de audio en los dos.
            if 'audio_encoder' in name or name == 'wavlm_layer_logits':
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
        # Rama de tasa unica. Correcta solo si los encoders estan congelados y lo
        # unico entrenable es la fusion/cabeza. Si algun parametro de un encoder
        # llega aqui entrenable, el regimen diferencial del config NO se esta
        # aplicando: se avisa en vez de dejarlo pasar en silencio, que es
        # exactamente como los estudios 44/45/66-69 acabaron entrenando sus
        # encoders a 5e-6.
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        encoder_entrenables = [
            n for n, q in model.named_parameters()
            if q.requires_grad and ('audio_encoder' in n or 'text_encoder' in n)
            and not n.endswith('layer_logits')
        ]
        if encoder_entrenables:
            warnings.warn(
                f"Estudio {study_number}: {len(encoder_entrenables)} parametros de "
                f"encoder entrenables, pero el optimizador usa UNA sola tasa "
                f"(lr_fusion={lr_fusion}). learning_rate_audio/text del config NO se "
                f"aplican. Para el regimen diferencial, pon training.use_param_groups: "
                f"true en el config.",
                UserWarning, stacklevel=2,
            )

        optimizer = optim.AdamW(
            trainable_params,
            lr=lr_fusion,
            weight_decay=weight_decay,
            eps=adam_epsilon
        )
    
    return optimizer

