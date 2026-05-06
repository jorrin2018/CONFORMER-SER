# Documentación CONFORMER-SER

Índice central de la documentación del proyecto **CONFORMER-SER**: reconocimiento de emociones en habla multimodal sobre IEMOCAP con 4 clases principales (`angry`, `happy`, `sad`, `neutral`).

## Política de git

- **`docs/`** se versiona en el repositorio.
- **`openspec/`** (planes de cambio OpenSpec) sigue en `.gitignore` por defecto: no esperes esos directorios en el remoto salvo que el equipo quite la regla.

## Orden de lectura (canónico)

1. [../quickstart.md](../quickstart.md) — instalación y comandos en PowerShell.
2. [ARCHITECTURE_AND_WORKFLOWS.md](ARCHITECTURE_AND_WORKFLOWS.md) — baseline vs ablación; `evaluate.py` solo para modelo base.
3. [../config/ablation/ABLATION_SUMMARY.md](../config/ablation/ABLATION_SUMMARY.md) — catálogo YAML ↔ estudio.
4. [experimentos/](experimentos/) — notas por bloque de estudios (tabla abajo).
5. Análisis global y operativa (tablas *Flujos operativos* y *Análisis y estrategias*).

Tres niveles temáticos:

1. **Entrada rápida**: instalación, comandos y rutas principales.
2. **Arquitectura y flujos**: entrenamiento, ablaciones, evaluación, ensemble y test-only.
3. **Análisis de experimentos**: resultados, estrategias e histórico por serie.

## Estructura de carpetas bajo `docs/`

| Ruta | Uso |
|------|-----|
| [diagrams/](diagrams/) | Diagramas Draw.io (arquitectura, pipelines, resúmenes gráficos) y scripts auxiliares. |
| [experimentos/](experimentos/) | Bitácoras agrupadas por rango de números de estudio. |
| Raíz de `docs/` | Índices, arquitectura, comandos de evaluación y análisis globales (ver siguiente sección). |

## Documentos en la raíz de `docs/` (informes y guías globales)

| Documento | Rol |
|-----------|-----|
| [README.md](README.md) | Este índice. |
| [ARCHITECTURE_AND_WORKFLOWS.md](ARCHITECTURE_AND_WORKFLOWS.md) | Mapa técnico del código y flujos. |
| [COMANDOS_EVALUACION.md](COMANDOS_EVALUACION.md) | Comandos de evaluación y comparación. |
| [CONFLICT_DETECTION.md](CONFLICT_DETECTION.md) | Modos de detección de conflicto (Conflict-Aware). |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Resumen ejecutivo del proyecto. |
| [POST_FIX_RETRAIN.md](POST_FIX_RETRAIN.md) | Reentrenamientos tras fixes. |
| [TRAINING_IMPROVEMENTS.md](TRAINING_IMPROVEMENTS.md) | Estabilidad y buenas prácticas de entrenamiento. |
| [ANALISIS_RESULTADOS_EXPERIMENTOS.md](ANALISIS_RESULTADOS_EXPERIMENTOS.md) | Análisis maestro hasta ~serie 40. |
| [COMPREHENSIVE_ABLATION_ANALYSIS.md](COMPREHENSIVE_ABLATION_ANALYSIS.md) | Análisis profundo de decisiones arquitectónicas. |
| [COLLAPSE_ANALYSIS.md](COLLAPSE_ANALYSIS.md) | Collapse de clases y señales de alerta. |
| [ESTRATEGIAS_MEJORA_ACCURACY.md](ESTRATEGIAS_MEJORA_ACCURACY.md) | Ideas para mejorar métricas. |
| [RESUMEN_ESTRATEGIAS.md](RESUMEN_ESTRATEGIAS.md) | Resumen ejecutivo de estrategias. |
| [RANKING_Y_RESULTADOS.md](RANKING_Y_RESULTADOS.md) | Ranking desde `final_results.json` agregados. |
| [PROMPT_ACTUALIZACION_DOCUMENTACION.md](PROMPT_ACTUALIZACION_DOCUMENTACION.md) | Plantilla para mantener doc experimental. |

**Renombres:** en esta iteración no se han movido rutas públicas dentro de `docs/`; enlaces externos antiguos no requieren tabla de migración.

---

## Lectura recomendada

| Documento | Propósito |
|-----------|-----------|
| [../README.md](../README.md) | Visión general del proyecto, estructura, instalación y comandos principales. |
| [../AGENTS.md](../AGENTS.md) | Convenciones del repo para colaboradores/agents: baseline vs ablación, comandos y trampas. |
| [../quickstart.md](../quickstart.md) | Guía operativa corta para instalar, entrenar, evaluar y ejecutar test-only en Windows/PowerShell. |
| [ARCHITECTURE_AND_WORKFLOWS.md](ARCHITECTURE_AND_WORKFLOWS.md) | Mapa técnico actual: módulos de código, flujo de datos, entrenamiento baseline, ablaciones, evaluación y checkpoints. |
| [../config/ablation/ABLATION_SUMMARY.md](../config/ablation/ABLATION_SUMMARY.md) | Catálogo canónico study -> YAML -> modelo para las ablaciones disponibles. |
| [../src/ablation_studies/README.md](../src/ablation_studies/README.md) | Resumen del paquete `src/ablation_studies` y de las clases de modelo por serie. |
| [RANKING_Y_RESULTADOS.md](RANKING_Y_RESULTADOS.md) | Ranking de estudios a partir de `checkpoints/**/final_results.json`: top por WA, por UA, gap val-test, tabla completa. Se regenera con `python scripts/aggregate_ablation_results.py --write-docs`. |

---

## Flujos operativos

| Documento | Contenido |
|-----------|-----------|
| [COMANDOS_EVALUACION.md](COMANDOS_EVALUACION.md) | Comandos de evaluación, rutas de checkpoints y ejemplos de comparación. |
| [POST_FIX_RETRAIN.md](POST_FIX_RETRAIN.md) | Reentrenamiento tras correcciones en entrenamiento/fusión y listado de estudios a repetir. |
| [TRAINING_IMPROVEMENTS.md](TRAINING_IMPROVEMENTS.md) | Mejoras prácticas para estabilizar entrenamiento y evitar problemas frecuentes. |
| [CONFLICT_DETECTION.md](CONFLICT_DETECTION.md) | Métodos `prediction_divergence`, `feature_distance` y `learned` para fusión Conflict-Aware. |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Resumen ejecutivo del proyecto y de sus componentes principales. |

Comandos base:

```powershell
python src\train.py --config config\config.yaml --device cuda
python scripts\run_ablation.py --study 34 --device cuda
python src\evaluate_ensemble.py --checkpoints <ckpt> --configs <yaml> --study_numbers 34 --weights 1.0 --split test --device cuda
python scripts\run_test_only.py --checkpoint_dir <run_dir_o_best_model.pt> --study 34 --device cuda
```

---

## Análisis y estrategias

| Documento | Alcance |
|-----------|---------|
| [ANALISIS_RESULTADOS_EXPERIMENTOS.md](ANALISIS_RESULTADOS_EXPERIMENTOS.md) | Análisis maestro de resultados hasta la serie 40. Mantener como histórico si no se re-tabula 41-54. |
| [COMPREHENSIVE_ABLATION_ANALYSIS.md](COMPREHENSIVE_ABLATION_ANALYSIS.md) | Análisis profundo de ablaciones y decisiones de arquitectura. |
| [COLLAPSE_ANALYSIS.md](COLLAPSE_ANALYSIS.md) | Diagnóstico de class collapse, señales de alerta y valores seguros de entrenamiento. |
| [ESTRATEGIAS_MEJORA_ACCURACY.md](ESTRATEGIAS_MEJORA_ACCURACY.md) | Ideas para superar el estancamiento de accuracy/UA. |
| [RESUMEN_ESTRATEGIAS.md](RESUMEN_ESTRATEGIAS.md) | Resumen ejecutivo de estrategias de mejora. |
| [PROMPT_ACTUALIZACION_DOCUMENTACION.md](PROMPT_ACTUALIZACION_DOCUMENTACION.md) | Prompt de mantenimiento para regenerar o actualizar documentación experimental. |

---

## Experimentos por serie

Los documentos de `docs/experimentos/` explican grupos de estudios con diagramas, diferencias arquitectónicas y resultados cuando están disponibles.

| Serie | Archivo | Estudios cubiertos |
|-------|---------|-------------------|
| 01-04 | [experimentos/01-04_classification_heads.md](experimentos/01-04_classification_heads.md) | 1-4 |
| 05-07 | [experimentos/05-07_unfreezing.md](experimentos/05-07_unfreezing.md) | 5-7 |
| 08-09 | [experimentos/08-09_depth_fusion.md](experimentos/08-09_depth_fusion.md) | 8-9 |
| 10-12 | [experimentos/10-12_speechcue_multimodal.md](experimentos/10-12_speechcue_multimodal.md) | 10-12 |
| 13 | [experimentos/13_baseline_trimodal.md](experimentos/13_baseline_trimodal.md) | 13-20 |
| 21-28 | [experimentos/21-28_fusion_mechanisms.md](experimentos/21-28_fusion_mechanisms.md) | 21-28 |
| 25/29 | [experimentos/25_enhanced_cross_attention.md](experimentos/25_enhanced_cross_attention.md) | 25, 29 |
| 30 | [experimentos/30_dirichlet.md](experimentos/30_dirichlet.md) | 30 |
| 31-35 | [experimentos/31-35_innovations.md](experimentos/31-35_innovations.md) | 31-35 |
| 34 | [experimentos/34_conflict_aware.md](experimentos/34_conflict_aware.md) | 34 |
| 36-39 | [experimentos/36-39_conflict_dirichlet_sota.md](experimentos/36-39_conflict_dirichlet_sota.md) | 36-39 |
| 40 | [experimentos/40_wavlm_conflict_aware.md](experimentos/40_wavlm_conflict_aware.md) | 40 |
| 41-42 | [experimentos/41-42_wavlm_vit_conflict_aware.md](experimentos/41-42_wavlm_vit_conflict_aware.md) | 41-42 |
| 43 | [experimentos/43_training_improvements.md](experimentos/43_training_improvements.md) | 43 |
| 44-46 | [experimentos/44-46_conflict_aware_heads.md](experimentos/44-46_conflict_aware_heads.md) | 44-46 |
| 47-50 | [experimentos/47-50_supcon_audio_encoders.md](experimentos/47-50_supcon_audio_encoders.md) | 47-50 |
| 52-54 | [experimentos/52-54_wavlm_specaugment_weighted_layers.md](experimentos/52-54_wavlm_specaugment_weighted_layers.md) | 52-54 |
| 55-57 | *Sin nota agrupada aún*: misma “receta” que 45 con distinto encoder de audio (`ablation_55_*`, `ablation_56_*`, `ablation_57_*` en `config/ablation/`). Detalle en [ABLATION_SUMMARY](../config/ablation/ABLATION_SUMMARY.md). | 55-57 |

`scripts/run_ablation.py` acepta estudios `1-50` y `52-57`. **No hay estudio 51.**

---

## Estado de documentación

- **Canónico para ejecutar ablaciones**: [../config/ablation/ABLATION_SUMMARY.md](../config/ablation/ABLATION_SUMMARY.md) y `scripts/run_ablation.py`.
- **Histórico de resultados**: documentos de análisis global como `ANALISIS_RESULTADOS_EXPERIMENTOS.md`; algunos cubren principalmente estudios 1-40.
- **Operativo diario**: [../quickstart.md](../quickstart.md), [COMANDOS_EVALUACION.md](COMANDOS_EVALUACION.md) y [ARCHITECTURE_AND_WORKFLOWS.md](ARCHITECTURE_AND_WORKFLOWS.md).

*Última actualización: mayo 2026*
