<div align="center">

# CONFORMER-SER

**Reconocimiento multimodal de emociones en habla sobre IEMOCAP**

Framework **CAMF** (*Conflict-Aware Multimodal Fusion*): audio + texto + visión con
fusión sensible al conflicto entre modalidades.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-FFD21E)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

---

> [!CAUTION]
> **Auditoria post-CIARP-2026: las cifras de resultados anteriores estan invalidadas.**
> Un bug de carga de datos descartaba las 2 943 utterances improvisadas. Esta corregido,
> pero ningun numero previo al arreglo (77.98 % UA, etc.) debe reutilizarse. Antes de
> tocar resultados o el articulo, lee [docs/REWRITE_NOTES.md](docs/REWRITE_NOTES.md) y
> [docs/dataset_integrity_patch.md](docs/dataset_integrity_patch.md), y ejecuta
> `python scriptserify_iemocap.py`. Las cifras publicables requieren **LOSO 5-fold**
> (`--loso`) con media ± sigma.

## Descripción

CONFORMER-SER investiga arquitecturas multimodales para clasificar emociones en habla sobre el
corpus **IEMOCAP**, con 4 clases: `angry`, `happy`, `sad` y `neutral`.

El repositorio contiene dos vías de trabajo independientes:

- una **línea base** entrenable desde `config/config.yaml`, y
- una **batería extensa de estudios de ablación** (`1–50`, `52–65`) que explora encoders de audio,
  mecanismos de fusión, cabezas de clasificación y funciones de pérdida.

| Aspecto | Valor |
|---|---|
| Dataset | IEMOCAP (`IEMOCAP_full_release`) |
| Clases | `angry`, `happy`, `sad`, `neutral` |
| Split | Hablante-independiente: sesiones **1–3** train · **4** val · **5** test |
| Métricas | WA, UA, F1 macro, F1 por clase, matriz de confusión |
| Entorno principal | Windows / PowerShell + CUDA |

---

## Tabla de contenidos

- [Instalación](#instalación)
- [Uso rápido](#uso-rápido)
- [Estudios de ablación](#estudios-de-ablación)
- [Arquitectura](#arquitectura)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Documentación](#documentación)
- [Resultados y análisis](#resultados-y-análisis)
- [Contribuir](#contribuir)
- [Cita](#cita)

---

## Instalación

### 1. Entorno virtual

```powershell
cd d:\DOCTORADO\PROYECTOS\CONFORMER-SER
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Dependencias

Para **GPU/CUDA** conviene instalar primero `torch` y `torchaudio` desde
[pytorch.org](https://pytorch.org/get-started/locally/) y después el resto:

```powershell
pip install -r requirements.txt
```

Atajo equivalente (instala dependencias y ejecuta la verificación):

```powershell
.\scripts\windows\setup.bat
```

### 3. Ruta del dataset

Edita `config\config.yaml`:

```yaml
dataset:
  data_path: "D:\\ruta\\a\\IEMOCAP_full_release"
```

### 4. Verificación

```powershell
python setup_check.py
```

Comprueba versión de Python, paquetes, disponibilidad de CUDA, estructura del proyecto y ruta del
dataset.

> **Dependencia opcional:** los estudios **50** y **57** (encoder Emotion2Vec) requieren `funasr`
> (línea comentada al final de [requirements.txt](requirements.txt)).

---

## Uso rápido

### Entrenamiento

```powershell
# Línea base
python src\train.py --config config\config.yaml --experiment_name baseline --device cuda

# Estudio de ablación
python scripts\run_ablation.py --study 45 --device cuda
```

### Evaluación

> [!IMPORTANT]
> `src\evaluate.py` **solo** reconstruye el modelo base. Para checkpoints de ablación usa
> `src\evaluate_ensemble.py` o `scripts\run_test_only.py`.

```powershell
# Checkpoint de la línea base
python src\evaluate.py `
  --checkpoint checkpoints\best_model.pt `
  --config config\config.yaml `
  --split test --device cuda

# Checkpoint de ablación (o ensemble de varios)
python src\evaluate_ensemble.py `
  --checkpoints checkpoints\ablation_34\run\best_model.pt `
  --configs config\ablation\ablation_34_conflict_aware_attention.yaml `
  --study_numbers 34 --weights 1.0 `
  --split test --output_dir evaluation_results\ablation_34 --device cuda

# Solo test sobre un run ya entrenado (lee su config.json)
python scripts\run_test_only.py `
  --checkpoint_dir checkpoints\ablation_45\ablation_45_20260203_094355 `
  --study 45 --device cuda
```

`run_test_only.py` acepta el directorio de un run o la ruta directa a `best_model.pt`, y escribe
`final_results.json`, `confusion_matrix_test.png` y `test_eval.log` en ese mismo directorio.

### Inferencia sobre un audio

```powershell
python src\inference.py `
  --checkpoint checkpoints\best_model.pt `
  --audio ruta\al\audio.wav `
  --text "Transcripción opcional"
```

### Monitoreo

```powershell
tensorboard --logdir=logs
# o bien
.\scripts\windows\start_tensorboard.bat
```

---

## Estudios de ablación

Estudios válidos: **`1–50`** y **`52–65`**. **No existe el estudio 51.**

El mapeo canónico *número de estudio → clase de modelo → YAML* está en
[`scripts/run_ablation.py`](scripts/run_ablation.py) (función `create_ablation_model`). El catálogo
legible está en [`config/ablation/ABLATION_SUMMARY.md`](config/ablation/ABLATION_SUMMARY.md).

| Serie | Tema |
|-------|------|
| 1–4 | Cabezas de clasificación (MLP, LSTM, Transformer, CNN1D) |
| 5–7 | Estrategias de descongelado de encoders |
| 8–9 | Profundidad de la fusión |
| 10–12 | SpeechCue (prompts prosódicos) y multimodalidad |
| 13–20 | Línea base trimodal + K-Fold |
| 21–28 | Mecanismos de fusión (jerárquica, tensorial, bilineal, GLU, gated, late…) |
| 29–30 | Cross-attention mejorada y fusión Dirichlet |
| 31–35 | Alineamiento temporal, bucketing, SpeechCue robusto, **Conflict-Aware** |
| 36–39 | Conflict-Aware + Dirichlet, OpenSMILE residual |
| 40–43 | WavLM + Conflict-Aware, ViT, mejoras de entrenamiento |
| 44–46 | Cabezas alternativas (Transformer, **MultiScaleTemporal**, CNN1D) |
| 47–50 | SupCon pre-fusión y encoders de audio alternativos |
| 52–54 | WavLM + SpecAugment + hidden states ponderados |
| 55–57 | Receta del estudio 45 con WavLM / HuBERT / Emotion2Vec |
| 58–63 | Combos SOTA: MultiScale + SupCon + SpecAugment, pooling atencional, fine-tuning |

> [!WARNING]
> Existen YAML sin ruta de código (p. ej. `ablation_65_unfrozen_vit.yaml` y
> `ablation_phase2_49_..._FUTURE.yaml`). **Un archivo de configuración no implica un estudio
> ejecutable**: verifica siempre contra `run_ablation.py`.

Los estudios **20, 29 y 41** ejecutan validación cruzada (K-Fold / LOSO) en lugar del split estándar.

Para agregar resultados de todos los runs:

```powershell
python scripts\aggregate_ablation_results.py --write-docs
```

---

## Arquitectura

- **Encoders de audio:** Wav2Vec2, WavLM (base / base-plus, con *weighted hidden states*
  opcional), HuBERT, Emotion2Vec.
- **Encoders de texto:** BERT / RoBERTa / DistilBERT, con inyección **SpeechCue** de prompts
  prosódicos en estudios concretos.
- **Rama visual:** un ViT opera sobre un **log-mel calculado en el `forward`** a partir del
  **mismo** waveform del batch (torchaudio `MelSpectrogram` → imagen).

> [!NOTE]
> IEMOCAP se consume como **audio crudo**. **No** hay espectrogramas mel precalculados en disco:
> se generan dentro del modelo en cada paso.

- **Fusiones** (`src/models/fusion.py`): Enhanced Cross-Attention, **Conflict-Aware
  Cross-Attention**, Gated Interactive Attention, Tensor Fusion, Bilinear, GLU, Late, Hierarchical,
  Adaptive Weighted.
- **Cabezas** (`src/ablation_studies/classification_heads.py`): MLP, LSTM, Transformer, CNN1D,
  MultiScaleTemporal.
- **Pérdidas** (`src/utils/losses.py`): Cross Entropy ponderada, Focal, Dirichlet Evidence
  Learning, SupCon y combinaciones compuestas.
- **Grupos de LR** (`src/ablation_studies/utils.py`): tasas de aprendizaje separadas para
  encoders, fusión y clasificador.

Detalle completo: [docs/ARCHITECTURE_AND_WORKFLOWS.md](docs/ARCHITECTURE_AND_WORKFLOWS.md) y
[docs/CONFLICT_DETECTION.md](docs/CONFLICT_DETECTION.md).

---

## Estructura del proyecto

```text
CONFORMER-SER/
├── config/
│   ├── config.yaml              # configuración de la línea base
│   └── ablation/                # YAML por estudio + ABLATION_SUMMARY.md
├── src/
│   ├── train.py                 # entrenamiento línea base
│   ├── evaluate.py              # evaluación línea base (solo)
│   ├── evaluate_ensemble.py     # evaluación de ablaciones y ensembles
│   ├── inference.py
│   ├── data/                    # dataset, preprocesado, augmentation, prosodia
│   ├── models/                  # modelo base, fusiones, encoders
│   ├── ablation_studies/        # clases de modelo por estudio + cabezas
│   ├── experiments/             # línea de encoders ligeros
│   └── utils/                   # pérdidas, métricas, logging
├── scripts/
│   ├── run_ablation.py          # entrenamiento de ablaciones
│   ├── run_test_only.py         # evaluación en test de un run existente
│   ├── aggregate_ablation_results.py
│   └── windows/                 # atajos .bat / .ps1
├── docs/                        # documentación versionada (índice: docs/README.md)
│   ├── experimentos/            # bitácoras por serie de estudios
│   └── diagrams/                # fuentes Draw.io / PlantUML / Mermaid
├── paper/                       # artefactos de publicación
│   ├── manuscripts/             # borradores .docx
│   ├── figures/                 # figuras SVG + graphical abstract
│   ├── posters/                 # pósters y su fuente HTML
│   └── presentations/           # presentaciones .pptx
├── tests/
├── notebooks/
├── setup_check.py
└── requirements.txt
```

Se generan al ejecutar y están **fuera de git**: `checkpoints/`, `logs/`, `evaluation_results/`,
`data/`, `node_modules/`, `.cache/`, `venv/`, `openspec/`.

---

## Documentación

Índice central: **[docs/README.md](docs/README.md)**.

Orden de lectura sugerido:

1. [quickstart.md](quickstart.md) — instalación y comandos esenciales.
2. [docs/ARCHITECTURE_AND_WORKFLOWS.md](docs/ARCHITECTURE_AND_WORKFLOWS.md) — mapa del código y
   flujos de datos.
3. [config/ablation/ABLATION_SUMMARY.md](config/ablation/ABLATION_SUMMARY.md) — catálogo de
   estudios.
4. [docs/experimentos/](docs/experimentos/) — bitácoras por serie.

Otros documentos útiles:

| Documento | Contenido |
|---|---|
| [AGENTS.md](AGENTS.md) | Manual para asistentes de IA: convenciones y trampas del repo |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Guía de contribución y estilo de código |
| [CHANGELOG.md](CHANGELOG.md) | Historial de versiones |
| [docs/CONFLICT_DETECTION.md](docs/CONFLICT_DETECTION.md) | Modos de detección de conflicto |
| [docs/COMANDOS_EVALUACION.md](docs/COMANDOS_EVALUACION.md) | Recetario de comandos de evaluación |
| [src/ablation_studies/README.md](src/ablation_studies/README.md) | Paquete de modelos de ablación |

---

## Resultados y análisis

Los resultados dependen de la configuración exacta, el split, el checkpoint y la versión de la
ablación. Al reproducir un run, usa siempre el `config.json` guardado **dentro del directorio del
run**, no el YAML de `config/ablation/` (puede haber cambiado desde entonces).

| Documento | Alcance |
|---|---|
| [docs/RANKING_Y_RESULTADOS.md](docs/RANKING_Y_RESULTADOS.md) | Ranking regenerable desde los `final_results.json` |
| [docs/ANALISIS_RESULTADOS_EXPERIMENTOS.md](docs/ANALISIS_RESULTADOS_EXPERIMENTOS.md) | Análisis histórico hasta la serie 40 |
| [docs/COMPREHENSIVE_ABLATION_ANALYSIS.md](docs/COMPREHENSIVE_ABLATION_ANALYSIS.md) | Análisis profundo de decisiones arquitectónicas |
| [docs/COLLAPSE_ANALYSIS.md](docs/COLLAPSE_ANALYSIS.md) | Diagnóstico de *class collapse* |
| [docs/ESTRATEGIAS_MEJORA_ACCURACY.md](docs/ESTRATEGIAS_MEJORA_ACCURACY.md) | Estrategias para superar el estancamiento de métricas |

---

## Contribuir

Consulta [CONTRIBUTING.md](CONTRIBUTING.md) para el flujo de trabajo, el estilo de código y las
convenciones de commits.

---

## Cita

Si usas este repositorio en un trabajo académico, cita el proyecto e indica el **commit exacto** y
la **configuración** empleada, para garantizar la reproducibilidad:

```bibtex
@software{conformer_ser,
  author  = {Jorrin, Jorge Luis},
  title   = {CONFORMER-SER: Conflict-Aware Multimodal Fusion for Speech Emotion Recognition},
  url     = {https://github.com/jorrin2018/CONFORMER-SER},
  note    = {Commit: <hash>, Config: <ruta al config.json del run>}
}
```

## Licencia

Distribuido bajo licencia MIT. Ver [LICENSE](LICENSE).
