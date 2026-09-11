# Documentación de CONFORMER-SER

Índice central de la documentación del proyecto: reconocimiento multimodal de emociones en habla
sobre IEMOCAP, 4 clases (`angry`, `happy`, `sad`, `neutral`).

- Visión general del proyecto → [../README.md](../README.md)
- Puesta en marcha → [../quickstart.md](../quickstart.md)
- Manual para asistentes de IA → [../AGENTS.md](../AGENTS.md)

---

---

> [!CAUTION]
> **Auditoria post-CIARP-2026: las cifras de resultados anteriores estan invalidadas.**
> Un bug de carga de datos descartaba las 2 943 utterances improvisadas. Esta corregido,
> pero ningun numero previo al arreglo (77.98 % UA, etc.) debe reutilizarse. Antes de
> tocar resultados o el articulo, lee [REWRITE_NOTES.md](REWRITE_NOTES.md) y
> [dataset_integrity_patch.md](dataset_integrity_patch.md), y ejecuta
> `python scriptserify_iemocap.py`. Las cifras publicables requieren **LOSO 5-fold**
> (`--loso`) con media ± sigma.

## Política de versionado

| Ruta | En git | Nota |
|------|:------:|------|
| `docs/` | ✅ | Toda la documentación técnica y las bitácoras de experimentos |
| `paper/` | ✅ | Artefactos de publicación (manuscritos, figuras, pósters, presentaciones) |
| `config/` · `src/` · `scripts/` | ✅ | Código y configuraciones |
| `openspec/` | ❌ | Planes de cambio OpenSpec: quedan **locales** |
| `checkpoints/` · `logs/` · `evaluation_results/` · `data/` | ❌ | Salidas de ejecución |
| `node_modules/` · `.cache/` · `venv/` | ❌ | Entornos y cachés de herramientas |

---

## Orden de lectura canónico

1. [../quickstart.md](../quickstart.md) — instalación y comandos en PowerShell.
2. [ARCHITECTURE_AND_WORKFLOWS.md](ARCHITECTURE_AND_WORKFLOWS.md) — línea base vs. ablación; mapa
   del código y flujo de datos.
3. [../config/ablation/ABLATION_SUMMARY.md](../config/ablation/ABLATION_SUMMARY.md) — catálogo
   estudio ↔ YAML ↔ modelo.
4. [experimentos/](experimentos/) — bitácoras por serie de estudios.
5. Análisis globales y estrategias (tablas más abajo).

---

## Estructura de `docs/`

| Ruta | Contenido |
|------|-----------|
| Raíz de `docs/` | Índice, arquitectura, comandos y análisis globales |
| [experimentos/](experimentos/) | Bitácoras agrupadas por rango de números de estudio |
| [diagrams/](diagrams/) | Fuentes de diagramas: Draw.io, PlantUML, Mermaid |

> Los pósters, figuras del artículo, manuscritos y presentaciones **ya no viven en `docs/`**:
> están en [`../paper/`](../paper/). Ver la sección [Artefactos de publicación](#artefactos-de-publicación).

---

## Guías y referencia técnica

| Documento | Rol |
|-----------|-----|
| [ARCHITECTURE_AND_WORKFLOWS.md](ARCHITECTURE_AND_WORKFLOWS.md) | Mapa técnico: módulos, flujo de datos, entrenamiento, evaluación y checkpoints |
| [CONFLICT_DETECTION.md](CONFLICT_DETECTION.md) | Modos `prediction_divergence`, `feature_distance` y `learned` de la fusión Conflict-Aware |
| [COMANDOS_EVALUACION.md](COMANDOS_EVALUACION.md) | Recetario de comandos de evaluación y comparación |
| [TRAINING_IMPROVEMENTS.md](TRAINING_IMPROVEMENTS.md) | Buenas prácticas para estabilizar el entrenamiento |
| [POST_FIX_RETRAIN.md](POST_FIX_RETRAIN.md) | Reentrenamientos tras correcciones en entrenamiento/fusión |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Resumen ejecutivo: arquitectura, estado, cifras y series de ablación |
| [experiment_lightweight.md](experiment_lightweight.md) | Línea de experimentos con encoders ligeros (`src/experiments/`) |
| [../src/ablation_studies/README.md](../src/ablation_studies/README.md) | Paquete `ablation_studies`: clases de modelo por serie |

Comandos base:

```powershell
python src\train.py --config config\config.yaml --device cuda
python scripts\run_ablation.py --study 45 --device cuda
python src\evaluate_ensemble.py --checkpoints <ckpt> --configs <yaml> --study_numbers 45 --weights 1.0 --split test --device cuda
python scripts\run_test_only.py --checkpoint_dir <run_dir> --study 45 --device cuda
```

---

## Análisis de resultados y estrategias

| Documento | Alcance |
|-----------|---------|
| [RANKING_Y_RESULTADOS.md](RANKING_Y_RESULTADOS.md) | Ranking por WA/UA y gap val-test. **Se regenera** con `python scripts\aggregate_ablation_results.py --write-docs` |
| [COMPREHENSIVE_ABLATION_ANALYSIS.md](COMPREHENSIVE_ABLATION_ANALYSIS.md) | 🕰️ **Histórico**: análisis profundo de decisiones arquitectónicas (estudios 1–39) |
| [COLLAPSE_ANALYSIS.md](COLLAPSE_ANALYSIS.md) | Diagnóstico de *class collapse*, señales de alerta y valores seguros |
| [ESTRATEGIAS_MEJORA_ACCURACY.md](ESTRATEGIAS_MEJORA_ACCURACY.md) | 🕰️ **Histórico**: ideas para superar el estancamiento de accuracy/UA (base: 39 experimentos) |
| [RESUMEN_ESTRATEGIAS.md](RESUMEN_ESTRATEGIAS.md) | 🕰️ **Histórico**: resumen ejecutivo de las estrategias anteriores |
| [ANALISIS_RESULTADOS_EXPERIMENTOS.md](ANALISIS_RESULTADOS_EXPERIMENTOS.md) | 🕰️ **Histórico**: análisis maestro hasta ~la serie 40 |
| [ablation_45_deep_dive.md](ablation_45_deep_dive.md) | Análisis exhaustivo del estudio 45 (MultiScaleTemporal + Conflict-Aware) |
| [PROMPT_ACTUALIZACION_DOCUMENTACION.md](PROMPT_ACTUALIZACION_DOCUMENTACION.md) | 🔧 Plantilla interna para regenerar documentación experimental |

---

## Experimentos por serie

Los documentos de [experimentos/](experimentos/) explican grupos de estudios con diagramas,
diferencias arquitectónicas y resultados cuando están disponibles.

| Serie | Archivo | Estudios |
|-------|---------|----------|
| 01–04 | [01-04_classification_heads.md](experimentos/01-04_classification_heads.md) | 1–4 |
| 05–07 | [05-07_unfreezing.md](experimentos/05-07_unfreezing.md) | 5–7 |
| 08–09 | [08-09_depth_fusion.md](experimentos/08-09_depth_fusion.md) | 8–9 |
| 10–12 | [10-12_speechcue_multimodal.md](experimentos/10-12_speechcue_multimodal.md) | 10–12 |
| 13–20 | [13_baseline_trimodal.md](experimentos/13_baseline_trimodal.md) | 13–20 |
| 21–28 | [21-28_fusion_mechanisms.md](experimentos/21-28_fusion_mechanisms.md) | 21–28 |
| 25/29 | [25_enhanced_cross_attention.md](experimentos/25_enhanced_cross_attention.md) | 25, 29 |
| 30 | [30_dirichlet.md](experimentos/30_dirichlet.md) | 30 |
| 31–35 | [31-35_innovations.md](experimentos/31-35_innovations.md) | 31–35 |
| 34 | [34_conflict_aware.md](experimentos/34_conflict_aware.md) | 34 |
| 36–39 | [36-39_conflict_dirichlet_sota.md](experimentos/36-39_conflict_dirichlet_sota.md) | 36–39 |
| 40 | [40_wavlm_conflict_aware.md](experimentos/40_wavlm_conflict_aware.md) | 40 |
| 41–42 | [41-42_wavlm_vit_conflict_aware.md](experimentos/41-42_wavlm_vit_conflict_aware.md) | 41–42 |
| 43 | [43_training_improvements.md](experimentos/43_training_improvements.md) | 43 |
| 44–46 | [44-46_conflict_aware_heads.md](experimentos/44-46_conflict_aware_heads.md) | 44–46 |
| 45 | [ablation_45_arquitectura.md](experimentos/ablation_45_arquitectura.md) · [post_45_optimization.md](experimentos/post_45_optimization.md) | 45 y optimizaciones posteriores |
| 47–50 | [47-50_supcon_audio_encoders.md](experimentos/47-50_supcon_audio_encoders.md) | 47–50 |
| 52–54 | [52-54_wavlm_specaugment_weighted_layers.md](experimentos/52-54_wavlm_specaugment_weighted_layers.md) | 52–54 |
| 55–57 | *Sin bitácora agrupada*: misma receta que el 45 con distinto encoder de audio (WavLM / HuBERT / Emotion2Vec). Ver [ABLATION_SUMMARY](../config/ablation/ABLATION_SUMMARY.md). | 55–57 |
| 58–63 | *Sin bitácora agrupada*: combos SOTA (MultiScale + SupCon + SpecAugment), pooling atencional y fine-tuning del 45. Ver [ABLATION_SUMMARY](../config/ablation/ABLATION_SUMMARY.md). | 58–63 |

> `scripts/run_ablation.py` acepta estudios **`1-50`** y **`52-65`**. **No existe el estudio 51.**
> Algunos YAML de `config/ablation/` (p. ej. `ablation_65_unfrozen_vit.yaml`) **no tienen ruta de
> código** y no son ejecutables.

---

## Diagramas

Fuentes editables en [diagrams/](diagrams/):

| Archivo | Contenido |
|---------|-----------|
| `CAMF_architecture_LNCS.drawio` | Arquitectura CAMF para el artículo |
| `graphical_abstract_LNCS.drawio` | Graphical abstract |
| `mel_vit_pipeline_LNCS.drawio` · `pipeline_waveform_to_m_ViT.drawio` | Pipeline waveform → log-mel → ViT |
| `speechcue_prosody_LNCS.drawio` | Inyección de prompts prosódicos (SpeechCue) |
| `multimodal_ser_architecture.drawio` | Arquitectura multimodal general |
| `ablation_45_architecture.{mmd,puml,png}` | Arquitectura del estudio 45 (Mermaid / PlantUML / render) |
| `_gen_mel_pipeline_drawio.py` | Generador del diagrama de pipeline mel |

---

## Artefactos de publicación

Viven fuera de `docs/`, en [`../paper/`](../paper/):

| Ruta | Contenido |
|------|-----------|
| [../paper/manuscripts/](../paper/manuscripts/) | Borradores `.docx` (CIARP 2025 y 2026) |
| [../paper/figures/](../paper/figures/) | `fig1`–`fig6` en SVG y `graphical_abstract.pdf` |
| [../paper/posters/](../paper/posters/) | Pósters PDF/PNG, su fuente `poster_conflict_aware_print.html` y `assets/` |
| [../paper/presentations/](../paper/presentations/) | Presentación `.pptx` del deep dive del estudio 45 |

Los pósters son **generados**. Edita el HTML y reexporta:

```powershell
.\scripts\export_poster_18x24.ps1 -Format pdf
.\scripts\export_poster_18x24.ps1 -Format png -Dpi 300
.\scripts\export_poster_a0.ps1 -Format pdf
```

La salida se escribe en `paper\posters\`. No edites a mano los PDF/PNG generados.

---

## Estado de la documentación

- **Canónico para ejecutar ablaciones:** `scripts/run_ablation.py` y
  [ABLATION_SUMMARY.md](../config/ablation/ABLATION_SUMMARY.md).
- **Operativa diaria:** [../quickstart.md](../quickstart.md),
  [COMANDOS_EVALUACION.md](COMANDOS_EVALUACION.md) y
  [ARCHITECTURE_AND_WORKFLOWS.md](ARCHITECTURE_AND_WORKFLOWS.md).
- **Histórico:** los documentos marcados con 🕰️ cubren principalmente los estudios 1–40 y se
  conservan como referencia; no se re-tabulan.
