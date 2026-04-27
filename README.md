# CONFORMER-SER: Multimodal Speech Emotion Recognition

CONFORMER-SER es un proyecto de reconocimiento de emociones en habla sobre IEMOCAP. El código combina encoders de audio, texto y visión con múltiples mecanismos de fusión y una línea extensa de estudios de ablación.

El objetivo principal del repositorio es investigar arquitecturas multimodales para 4 clases: `angry`, `happy`, `sad` y `neutral`.

## Documentación

La documentación técnica central está en [docs/README.md](docs/README.md).

Lecturas recomendadas:

- [quickstart.md](quickstart.md): comandos esenciales para instalar, entrenar y evaluar.
- [docs/ARCHITECTURE_AND_WORKFLOWS.md](docs/ARCHITECTURE_AND_WORKFLOWS.md): mapa real del código y diferencias entre baseline, ablaciones, ensemble y test-only.
- [config/ablation/ABLATION_SUMMARY.md](config/ablation/ABLATION_SUMMARY.md): catálogo canónico de estudios de ablación.
- [src/ablation_studies/README.md](src/ablation_studies/README.md): resumen del paquete de modelos de ablación.
- [docs/CONFLICT_DETECTION.md](docs/CONFLICT_DETECTION.md): métodos de detección de conflicto para Conflict-Aware Cross-Attention.

## Características principales

- Arquitecturas **bimodales y trimodales** con audio, texto y visión.
- Encoders de audio basados en Wav2Vec2, WavLM y variantes recientes.
- Texto con modelos tipo BERT/RoBERTa/DistilBERT y SpeechCue en estudios específicos.
- Rama visual: el ViT opera sobre **log-mel calculado en el forward** a partir del **mismo** waveform del batch (torchaudio `MelSpectrogram` + conversión a imagen); **no** hay mel precalculado almacenado en el dataset IEMOCAP.
- Fusiones avanzadas: Enhanced Cross-Attention, Conflict-Aware Cross-Attention, Gated Interactive Attention, Tensor Fusion, GLU, Late Fusion, Hierarchical Fusion y más.
- Pérdidas y entrenamiento robusto: Cross Entropy ponderada, Focal Loss, Dirichlet Evidence Learning, SupCon y pérdidas compuestas.
- Evaluación con WA, UA, F1, métricas por clase y matriz de confusión.

## Estructura del proyecto

```text
CONFORMER-SER/
├── config/
│   ├── config.yaml
│   └── ablation/
│       ├── ABLATION_SUMMARY.md
│       └── ablation_*.yaml
├── docs/
│   ├── README.md
│   ├── ARCHITECTURE_AND_WORKFLOWS.md
│   └── experimentos/
├── scripts/
│   ├── run_ablation.py
│   ├── run_test_only.py
│   ├── retrain_post_fix_gpu.ps1
│   └── cleanup_pre_fix_checkpoints.ps1
├── src/
│   ├── ablation_studies/
│   ├── models/
│   ├── utils/
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_ensemble.py
│   └── inference.py
├── checkpoints/
├── logs/
├── README.md
└── quickstart.md
```

## Instalación

En Windows/PowerShell:

```powershell
cd d:\DOCTORADO\PROYECTOS\CONFORMER-SER
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python setup_check.py
```

También puedes usar:

```powershell
.\setup.bat
```

Configura la ruta del dataset IEMOCAP en `config/config.yaml`:

```yaml
dataset:
  data_path: "D:\\ruta\\a\\IEMOCAP_full_release"
```

El estudio 50 usa Emotion2Vec y requiere instalar `funasr`.

## Uso rápido

### Entrenamiento baseline

```powershell
python src\train.py --config config\config.yaml --experiment_name baseline --device cuda
```

### Entrenamiento de ablaciones

```powershell
python scripts\run_ablation.py --study 34 --device cuda
```

Los estudios definidos actualmente son `1-50, 52, 53, 54`; no hay estudio 51 en `scripts/run_ablation.py`.

### Evaluación baseline

```powershell
python src\evaluate.py --checkpoint checkpoints\best_model.pt --config config\config.yaml --split test --device cuda
```

### Evaluación de una ablación o ensemble

```powershell
python src\evaluate_ensemble.py `
  --checkpoints checkpoints\ablation_34\run\best_model.pt `
  --configs config\ablation\ablation_34_conflict_aware_attention.yaml `
  --study_numbers 34 `
  --weights 1.0 `
  --split test `
  --output_dir evaluation_results\ablation_34 `
  --device cuda
```

### Test-only de un checkpoint ya entrenado

```powershell
python scripts\run_test_only.py --checkpoint_dir checkpoints\phase1_wavlm_gated\ablation_34_20260203_094355 --study 34 --device cuda
```

`run_test_only.py` acepta un directorio de run o una ruta directa a `best_model.pt`. Usa `config.json` del run y escribe `final_results.json`, `confusion_matrix_test.png` y `test_eval.log` en el mismo directorio.

## Resultados y análisis

Los resultados dependen de la configuración exacta, split, checkpoint y versión de la ablación. Para consultar los análisis existentes:

- [docs/ANALISIS_RESULTADOS_EXPERIMENTOS.md](docs/ANALISIS_RESULTADOS_EXPERIMENTOS.md): análisis histórico hasta la serie 40.
- [docs/COMPREHENSIVE_ABLATION_ANALYSIS.md](docs/COMPREHENSIVE_ABLATION_ANALYSIS.md): análisis profundo de decisiones arquitectónicas.
- [docs/COMANDOS_EVALUACION.md](docs/COMANDOS_EVALUACION.md): comandos de evaluación usados en comparaciones.

## Monitoreo

```powershell
tensorboard --logdir=logs
```

O:

```powershell
.\start_tensorboard.bat
```

## Cita

Si usas este repositorio en un trabajo académico, cita el proyecto y documenta el commit/configuración exacta usada para reproducibilidad.
