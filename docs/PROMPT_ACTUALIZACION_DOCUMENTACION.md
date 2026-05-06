# Prompt optimizado: Actualización y organización de documentación de experimentos

Usa este prompt (o sus secciones) cuando quieras actualizar la documentación de experimentos de forma completa y reproducible.

---

## Prompt principal (copiar y pegar)

```
Objetivo: Actualizar y organizar toda la documentación de experimentos del proyecto CONFORMER-SER para que sirva como mapa claro y detallado de cada experimento y sus resultados.

FUENTES DE DATOS (consultar en este orden):
- Resultados de test: checkpoints, logs, o cualquier salida con métricas test (UA, F1 macro, accuracy por clase).
- Configuraciones: archivos en config/ablation/*.yaml (model, dataset, audio, text, training).
- Código de modelos: src/ablation_studies/*.py y src/models/*.py para arquitecturas y dimensiones.
- Resumen existente: config/ablation/ABLATION_SUMMARY.md para nombres y agrupación.

ENTREGABLES:

1) Actualizar docs/ANALISIS_RESULTADOS_EXPERIMENTOS.md
   - Incluir TODOS los experimentos (01 a 40) con resultados de TEST como métrica principal.
   - Por cada experimento (o grupo si comparten arquitectura), incluir en una tabla o sección:
     - ID y nombre del experimento.
     - Arquitectura resumida: encoders (audio, texto, visión), tipo de fusión, cabeza de clasificación.
     - Modelos usados (ej. wav2vec2-base, distilbert, ViT, WavLM).
     - Hiperparámetros clave: hidden_dim, dropout, num_layers (fusion/LSTM), learning_rate, batch_size, epochs típicos.
     - Técnicas especiales: SpeechCue, Conflict-Aware, Dirichlet, Focal Loss, OpenSMILE, etc.
     - Métricas test: UA, F1 macro, accuracy (y por clase si hay datos).
     - Gap val–test y observaciones breves (generalización, overfitting).
   - Mantener un índice o resumen ejecutivo al inicio y secciones por serie (01–10, 11–13, 21–28, 30–35, 36–40).
   - El documento debe servir como “mapa” para entender de un vistazo qué se probó, con qué configuración y qué resultado dio en test.

2) Crear un .md por experimento o por grupo de experimentos con la misma arquitectura
   - Ubicación: docs/experimentos/ (o docs/ablations/).
   - Agrupar por arquitectura cuando sea la misma base (ej. 25, 31–35 como “Enhanced Cross-Attention + variantes”; 34, 36a–c, 37, 40 como “Conflict-Aware + variantes”).
   - Contenido de cada documento:
     - Título, ID(s) de experimento, objetivo en una frase.
     - Diagrama de bloques (en texto/ASCII o Mermaid): flujo desde entradas (audio, texto, visión) hasta logits/clasificación, con nombres de módulos (encoder audio, encoder texto, fusión, head).
     - Dimensiones: tamaños de entrada, dimensiones ocultas (hidden_dim), dimensiones de salida de cada encoder antes/después de fusión, tamaño de secuencia, num_classes.
     - Explicación paso a paso: cómo fluyen los datos (1. extracción de features, 2. fusión, 3. clasificación), qué hace cada bloque y qué parámetros lo definen.
     - Configuración e hiperparámetros relevantes (extraídos del YAML).
     - Resultados de test (UA, F1, por clase si existen) y referencia al análisis principal.
   - Si un archivo cubre varios experimentos (misma arquitectura), usar subsecciones por experimento con las diferencias concretas (solo loss, solo encoder, etc.).

3) Organizar toda la documentación del proyecto en docs/
   - Estructura objetivo sugerida:
     - docs/README.md: índice de toda la documentación (enlaces a análisis, a cada doc de experimento, a CONFLICT_DETECTION, ABLATION_SUMMARY, etc.).
     - docs/ANALISIS_RESULTADOS_EXPERIMENTOS.md: análisis maestro (actualizado como en 1).
     - docs/experimentos/: un .md por experimento o grupo (como en 2).
     - docs/CONFLICT_DETECTION.md y otros .md existentes: mantener y enlazar desde README.
   - Mover o enlazar desde docs cualquier documento disperso que sea documentación de proyecto (no código), para que docs sea el único punto de entrada.

CRITERIOS DE CALIDAD:
- Claridad: alguien que no haya corrido el experimento debe entender qué se hizo, con qué y qué resultado dio.
- Consistencia: mismas métricas (test como referencia), mismas unidades (%), misma estructura entre documentos de experimentos.
- Trazabilidad: cada resultado debe poder asociarse a un config YAML y a un modelo en código.
```

---

## Prompt resumido (versión corta)

```
Actualiza docs/ANALISIS_RESULTADOS_EXPERIMENTOS.md con todos los experimentos (01–40) usando resultados de TEST. Incluye por experimento: arquitectura, modelos, config, hiperparámetros y técnicas. Debe servir como mapa claro.

Crea en docs/experimentos/ un .md por experimento o por grupo con misma arquitectura, con: diagrama de bloques, dimensiones, explicación paso a paso y resultados test.

Organiza toda la documentación en docs/ con un README índice. Fuentes: config/ablation/*.yaml, src/ablation_studies y src/models, ABLATION_SUMMARY.md y checkpoints/logs de test.
```

---

## Estructura sugerida de docs/

```
docs/
├── README.md                          # Índice de documentación
├── ANALISIS_RESULTADOS_EXPERIMENTOS.md # Análisis maestro (actualizado)
├── CONFLICT_DETECTION.md              # Existente
├── PROMPT_ACTUALIZACION_DOCUMENTACION.md  # Este archivo
└── experimentos/                      # Un .md por experimento o grupo
    ├── 01-04_classification_heads.md
    ├── 05-07_unfreezing.md
    ├── 08-09_depth_fusion.md
    ├── 10-12_speechcue_multimodal.md
    ├── 13_baseline_trimodal.md
    ├── 21-28_fusion_mechanisms.md
    ├── 25_enhanced_cross_attention.md
    ├── 30_dirichlet.md
    ├── 31-35_innovations.md
    ├── 34_conflict_aware.md
    ├── 36-39_conflict_dirichlet_sota.md
    └── 40_wavlm_conflict_aware.md
```

---

## Checklist antes de ejecutar el prompt

- [ ] Tener a mano (o indicar la ruta de) checkpoints/logs con métricas de **test**.
- [ ] Confirmar que `config/ablation/ABLATION_SUMMARY.md` está actualizado con estudios 1–40.
- [ ] Decidir si los diagramas serán Mermaid (recomendado), ASCII o ambos.
- [ ] Crear la carpeta `docs/experimentos/` si no existe.

---

*Última actualización: Marzo 2025*
