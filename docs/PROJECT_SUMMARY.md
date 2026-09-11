# Resumen ejecutivo — CONFORMER-SER

Visión de conjunto del proyecto. Para instalar y ejecutar, ve a
[../quickstart.md](../quickstart.md); para el detalle técnico,
[ARCHITECTURE_AND_WORKFLOWS.md](ARCHITECTURE_AND_WORKFLOWS.md).

---

## Qué es

Sistema **trimodal** (audio + texto + visión) de reconocimiento de emociones en habla sobre
**IEMOCAP**, con 4 clases: `angry`, `happy`, `sad`, `neutral`.

La contribución de investigación es **CAMF** (*Conflict-Aware Multimodal Fusion*): una fusión que
detecta y pondera el desacuerdo entre modalidades, pensada para casos donde el contenido semántico
y la prosodia apuntan a emociones distintas (ironía, sarcasmo).

---

## Arquitectura

| Componente | Implementación |
|---|---|
| **Audio** | Wav2Vec2 / WavLM / HuBERT / Emotion2Vec (últimas capas descongeladas) |
| **Texto** | DistilBERT / BERT / RoBERTa + **SpeechCue** (prompts prosódicos) |
| **Visión** | ViT sobre log-mel **calculado en el `forward`** desde el waveform del batch |
| **Fusión** | Conflict-Aware Cross-Attention (est. 34) · Enhanced Cross-Attention (est. 25) · y 7 más |
| **Cabeza** | MLP · Bi-LSTM · Transformer · CNN1D · MultiScaleTemporal (est. 45) · Dirichlet |
| **Pérdidas** | CE ponderada · Focal · Dirichlet Evidence · SupCon · compuestas |

> **Nota sobre la rama visual:** IEMOCAP se consume como audio crudo. **No hay espectrogramas mel
> precalculados en disco**; se generan dentro del modelo en cada paso a partir del mismo waveform
> que consume la rama de audio.

---

## Estado

Componentes principales implementados y en desarrollo activo.

| Métrica | Valor |
|---|---|
| Estudios de ablación ejecutables | **64** (rangos `1-50` y `52-65`) |
| Configuraciones YAML en `config/ablation/` | 70 |
| Clases de modelo en `src/ablation_studies/` | 34 |
| Documentos en `docs/` | 18 |

Hay más YAML que estudios ejecutables: varias configs son variantes de un mismo estudio y **dos
son huérfanas** (`ablation_65_unfrozen_vit.yaml` y
`ablation_phase2_49_wavlm_multiscale_head_FUTURE.yaml`), sin ruta de código en `run_ablation.py`.
Hay menos clases de modelo que estudios porque muchos estudios **reutilizan la misma clase** con
distinto YAML.

---

## Resultados

> [!CAUTION]
> **No hay cifras de resultados validas en este momento.**
>
> La auditoria posterior al rechazo de CIARP 2026 (septiembre de 2026) encontro un bug de
> carga de datos (`_find_audio_file`) que descartaba las **2 943 utterances improvisadas**.
> El bug esta corregido, pero **todas las cifras anteriores al arreglo quedan invalidadas**,
> incluidas las que circulaban como canonicas (Ablation 34 con 77.98 % UA, Ablation 25 con
> 77.44 %, Ablation 30 con 75.24 %). No las reutilices en texto, tablas ni presentaciones.
>
> El corpus verificado es de **5 531 utterances** en 4 clases
> (train 3 259 / val 1 031 / test 1 241). Verificalo con `python scriptserify_iemocap.py`.
>
> Requisitos para cualquier cifra publicable, segun la auditoria:
> **LOSO 5-fold** (`--loso`) con media ± sigma — la sigma entre folds de este codigo es de
> 4–5 pp, asi que un unico split no sostiene una afirmacion de menos de 1 pp; **un solo
> factor por fila de ablacion**; y reportar N de test y soporte por clase.
>
> Detalle y procedencia de cada numero medido:
> [REWRITE_NOTES.md](REWRITE_NOTES.md) y
> [dataset_integrity_patch.md](dataset_integrity_patch.md).

Cuando existan corridas LOSO posteriores al arreglo, el ranking se regenera con:

```powershell
python scriptsggregate_ablation_results.py --write-docs
```

y queda en [RANKING_Y_RESULTADOS.md](RANKING_Y_RESULTADOS.md).

---

## Series de ablación

| Serie | Tema |
|-------|------|
| 01–04 | Cabezas de clasificación (MLP, LSTM, Transformer, CNN1D) |
| 05–07 | Estrategias de descongelado de encoders |
| 08–09 | Profundidad de la fusión |
| 10–12 | SpeechCue y multimodalidad |
| 13–20 | Línea base trimodal (ViT) y K-Fold |
| 21–28 | Mecanismos de fusión (jerárquica, tensorial, bilineal, GMU, GLU, late) |
| 29–30 | Enhanced Cross-Attention y Dirichlet Evidence Learning |
| 31–35 | Alineamiento temporal, bucketing, SpeechCue robusto, **Conflict-Aware** |
| 36–39 | Conflict-Aware + Dirichlet/Focal, residual OpenSMILE, ACN+MIC+BCAF |
| 40–43 | WavLM + Conflict-Aware, ViT, mejoras de entrenamiento |
| 44–46 | Cabezas alternativas (Transformer, **MultiScaleTemporal**, CNN1D) |
| 47–50 | SupCon pre-fusión y encoders de audio alternativos |
| 52–54 | WavLM + SpecAugment + hidden states ponderados |
| 55–57 | Receta del estudio 45 con WavLM / HuBERT / Emotion2Vec |
| 58–63 | Combos SOTA, pooling atencional y ajuste fino del 45 |

Detalle por estudio: [../config/ablation/ABLATION_SUMMARY.md](../config/ablation/ABLATION_SUMMARY.md)
y [../src/ablation_studies/README.md](../src/ablation_studies/README.md).
Los estudios **20, 29 y 41** usan validación cruzada (K-Fold / LOSO) en lugar del split estándar.

---

## Estructura del proyecto

```text
CONFORMER-SER/
├── config/
│   ├── config.yaml              # configuración de la línea base
│   └── ablation/                # 70 YAML + ABLATION_SUMMARY.md
├── src/
│   ├── data/                    # IEMOCAP, preprocesado, ProsodyCue
│   ├── models/                  # conformer.py, fusion.py, encoders
│   ├── ablation_studies/        # 34 clases de modelo + cabezas
│   ├── experiments/             # línea de encoders ligeros
│   ├── utils/                   # métricas, logging, pérdidas
│   ├── train.py · evaluate.py · inference.py
│   └── evaluate_ensemble.py     # evaluación de ablaciones y ensembles
├── scripts/
│   ├── run_ablation.py          # ejecuta cualquier estudio 1-50, 52-65
│   ├── run_test_only.py         # solo test sobre un run existente
│   └── windows/                 # atajos .bat / .ps1
├── docs/                        # documentación (índice: docs/README.md)
├── paper/                       # manuscritos, figuras, pósters, presentaciones
├── tests/
├── checkpoints/                 # generado al entrenar (fuera de git)
└── logs/                        # TensorBoard (fuera de git)
```

---

## Uso

```powershell
# Entrenamiento de una ablación
python scripts\run_ablation.py --study 34 --device cuda   # Conflict-Aware
python scripts\run_ablation.py --study 45 --device cuda   # MultiScaleTemporal

# Evaluación de un checkpoint de ablación
python src\evaluate_ensemble.py `
  --checkpoints checkpoints\ablation_34\run\best_model.pt `
  --configs config\ablation\ablation_34_conflict_aware_attention.yaml `
  --study_numbers 34 --weights 1.0 --split test --device cuda

# Inferencia (modelo base)
python src\inference.py --checkpoint checkpoints\best_model.pt --audio audio.wav
```

> `src\evaluate.py` es **solo** para la línea base. Con checkpoints de ablación usa
> `evaluate_ensemble.py` o `run_test_only.py`.

---

## Características clave

- **Trimodal**: audio + texto + visión (ViT sobre log-mel generado en el modelo).
- **4 clases**: reducidas desde 7 por fiabilidad de las etiquetas.
- **SpeechCue**: prosodia convertida a prompt textual para enriquecer la rama semántica.
- **Fusión Conflict-Aware**: pondera el desacuerdo entre modalidades (ironía, sarcasmo).
- **Dirichlet Evidence Learning**: clasificación con estimación de incertidumbre.
- **OpenSMILE**: descriptores eGeMAPSv02 (estudio 38, requiere `opensmile`).
- **Hablante-independiente**: sesiones 1–3 train, 4 val, 5 test.

---

## Requisitos técnicos

| | |
|---|---|
| Python | 3.8+ |
| PyTorch | 2.0+ |
| CUDA | 11.7+ recomendado |
| Dataset | IEMOCAP (~5K utterances tras filtrar a 4 clases) |
| RAM | 32 GB recomendado |
| VRAM | 12 GB+ recomendado |
| Duración | ~4–8 h por ablación, según GPU |

Dependencias: [../requirements.txt](../requirements.txt). Los extras opcionales (`funasr`,
`opensmile`, `bitsandbytes`, `xformers`, `triton`) están documentados ahí y `setup_check.py`
informa de cuáles tienes instalados.

---

## Documentación relacionada

| Documento | Contenido |
|---|---|
| [../README.md](../README.md) | Documentación principal |
| [../quickstart.md](../quickstart.md) | Instalación y comandos |
| [../AGENTS.md](../AGENTS.md) | Convenciones del repo para asistentes de IA |
| [README.md](README.md) | Índice de la documentación |
| [ARCHITECTURE_AND_WORKFLOWS.md](ARCHITECTURE_AND_WORKFLOWS.md) | Mapa técnico y flujos |
| [../config/ablation/ABLATION_SUMMARY.md](../config/ablation/ABLATION_SUMMARY.md) | Catálogo de estudios |
| [RANKING_Y_RESULTADOS.md](RANKING_Y_RESULTADOS.md) | Ranking regenerable de resultados |
| [COMANDOS_EVALUACION.md](COMANDOS_EVALUACION.md) | Recetario de evaluación |

---

*Versión 5.1.1 · Actualizado en septiembre de 2026 · Licencia MIT*
