# Análisis de Resultados de Experimentos - Conformer Multimodal SER

**Fecha de análisis (tabla 41–54 y rankings agregados)**: Abril 2026  
**Período cubierto por el relato histórico (series 01–35)**: Diciembre 2024 – Marzo 2025  
**Proyecto**: CONFORMER-SER - Reconocimiento de Emociones Multimodal

**Nota sobre métricas**: Además de las ejecuciones detalladas en el cuerpo del documento (p. ej. Ablation **34** con **77.98%** UA en test en una corrida concreta), la tabla maestra incluye filas **41–54** con valores **agregados desde los `final_results.json` presentes bajo `checkpoints/`** (script `scripts/aggregate_ablation_results.py`). Esos archivos corresponden sobre todo a la cadena **post-fix**; los números pueden **no coincidir** con la mejor corrida histórica de cada estudio si ese checkpoint no está en el repositorio.

---

## 📊 Resumen Ejecutivo

Este documento presenta un análisis exhaustivo de los resultados de los estudios de ablación del proyecto CONFORMER-SER (listado operativo en `scripts/run_ablation.py`: **1–50, 52, 53 y 54**; el estudio **51** no está definido). Los experimentos evalúan arquitecturas y técnicas para reconocimiento de emociones multimodal (Audio, Texto, Visual). **Dataset**: IEMOCAP, **4 clases** (angry, happy, sad, neutral). En trimodal, el log-mel del ViT se calcula **en el forward** desde el mismo waveform (no hay mel precalculado en el dataset; ver `config/ablation/ABLATION_SUMMARY.md`).

### Hallazgos Principales

1. **Mejor modelo individual en test (corrida documentada en este informe)**: **Ablation 34 (Conflict-Aware Attention)** con **77.98%** UA en test y **80.66%** UA en validación (gap **2.68%**) en la ejecución **20251210_191151** 🏆  
2. **Mejor modelo en validación (serie 30–35)**: **Ablation 30 (Dirichlet)** con **85.02%** UA en validación y **75.24%** UA en test (gap **9.78%**)  
3. **Mejor modelo base (Enhanced CA)**: **Ablation 25** con **77.44%** UA en test y **82.30%** UA en validación (gap **4.86%**)  
4. **Mejor K-Fold (serie clásica)**: **Ablation 25 K-Fold** con promedio **75.71%** UA en test (std **4.19%**)  
5. **Inventario actual de checkpoints (`final_results.json`, abril 2026)**: el agregador selecciona como mayor **UA test** al estudio **45** (**~79.23%** UA, **~79.61%** WA, gap UA **~2.91%**; un solo run en repo). **48** y **52** también quedan por encima del **34** presente en disco en ese snapshot (**~75.45%** UA al elegir por WA). Reproducir con: `python scripts/aggregate_ablation_results.py`  
6. **Serie 41–54**: extiende la línea **Conflict-Aware** con **WavLM** (K-Fold y split simple), **`Ablation43Model`** (focal + SupCon + augment), **cabezas** Transformer / MultiScaleTemporal / CNN1D, **SupCon** (2 y 3 vistas), **WavLM + SupCon**, **Emotion2Vec**, **SpecAugment en log-mel del ViT**, y **weighted hidden states** (WavLM Base / Base+). Catálogo: `config/ablation/ABLATION_SUMMARY.md`  
7. **Problema identificado**: variabilidad entre ejecuciones del mismo experimento y entre carpetas de checkpoint

---

## Tabla maestra de experimentos (01–50, 52–54)

Mapa de experimentos con arquitectura, modelos e ideas clave. **El estudio 51 no existe en el listado actual de `run_ablation.py`.** Métricas **41–54** (últimas filas): mejor run por estudio (o media K-Fold) según `scripts/aggregate_ablation_results.py` sobre **`checkpoints/**/final_results.json`** a abril 2026; donde no hay archivo en repo → **N/A**. Filas **01–40** conservan cifras del análisis original salvo donde se indica.

| ID | Nombre | Arquitectura (encoders + fusión + head) | Modelos | hidden_dim / dropout / fusion_layers | Técnicas | Test UA | F1 macro | Gap val-test | Observaciones |
|----|--------|------------------------------------------|---------|--------------------------------------|----------|---------|---------|--------------|---------------|
| 01 | Unfreeze Encoders | Bimodal: Wav2Vec2 + DistilBERT, cross_attention, MLP | wav2vec2-base, distilbert | 256 / 0.3 / - | Focal loss | 55.31% | - | - | Baseline 4 clases |
| 02 | LSTM Head | Bimodal, cross_attention, **LSTM** head | wav2vec2-base, distilbert | 256 / 0.3 / - | - | N/A | N/A | - | Ver config ablation_02 |
| 03 | Transformer Head | Bimodal, cross_attention, **Transformer** head | wav2vec2-base, distilbert | 256 / 0.3 / - | - | N/A | N/A | - | Ver config ablation_03 |
| 04 | CNN1D Head | Bimodal, cross_attention, **CNN1D** head | wav2vec2-base, distilbert | 256 / 0.3 / - | - | N/A | N/A | - | Ver config ablation_04 |
| 05 | Unfreeze LSTM | Bimodal, LSTM head, unfreeze estándar | wav2vec2-base, distilbert | 256 / 0.3 / - | - | N/A | N/A | - | ablation_05 |
| 06 | Unfreeze Transformer | Bimodal, Transformer head, unfreeze estándar | wav2vec2-base, distilbert | 256 / 0.3 / - | - | N/A | N/A | - | ablation_06 |
| 07 | Unfreeze Mid-High LSTM | Solo capas medias/altas descongeladas | wav2vec2-base, distilbert | 256 / 0.3 / - | - | N/A | N/A | - | ablation_07 |
| 08 | Deep LSTM | LSTM más profundo | wav2vec2-base, distilbert | 256 / 0.3 / - | - | N/A | N/A | - | ablation_08 |
| 09 | GIA LSTM | **Gated Interaction Attention** + LSTM | wav2vec2-base, distilbert | 256 / 0.3 / - | GIA | N/A | N/A | - | ablation_09 |
| 10 | SpeechCue Internal | Bimodal + **SpeechCue** (prosodia en texto) | wav2vec2-base, distilbert | 256 / 0.3 / - | SpeechCue | N/A | N/A | - | ablation_10 |
| 11 | MelSpec ViT | **Trimodal**: Wav2Vec2 + DistilBERT + **ViT** (log-mel en forward desde waveform) | wav2vec2-base, distilbert, vit-base | 256 / 0.3 / - | Rama visual ViT | N/A | N/A | - | ablation_11 |
| 12 | SpeechCue LOSO | Igual que 10, evaluación **LOSO** | wav2vec2-base, distilbert | 256 / 0.3 / - | SpeechCue, LOSO | 69.60%-78.52% | - | - | Ver sección LOSO |
| 13 | Trimodal ViT + SpeechCue | Trimodal: Wav2Vec2 + DistilBERT + ViT (unfreeze ViT), cross_attention, LSTM | wav2vec2-base, distilbert, vit-base | 256 / 0.3 / 4 heads | SpeechCue, ViT unfreeze | **74.46%** | 74.18% | 7.59% | Base robusta |
| 14 | 13a Regularized | Trimodal 13 + regularización (dropout, weight decay) | wav2vec2-base, distilbert, vit-base | 256 / 0.4 / - | Regularización | N/A | N/A | - | ablation_13a |
| 15 | 13b No SpeechCue | Trimodal sin SpeechCue | wav2vec2-base, distilbert, vit-base | 256 / - / - | - | N/A | N/A | - | ablation_13b |
| 16 | 13c Frozen ViT | Trimodal con **ViT congelado** | wav2vec2-base, distilbert, vit-base | 256 / - / - | ViT frozen | N/A | N/A | - | Base para 21-28 |
| 17 | 13d High Reg | Regularización agresiva | wav2vec2-base, distilbert, vit-base | 256 / - / - | High reg | N/A | N/A | - | ablation_13d |
| 18 | 13e Advanced Reg | Label smoothing, Mixup | wav2vec2-base, distilbert, vit-base | 256 / - / - | Advanced reg | N/A | N/A | - | ablation_13e |
| 19 | 13f Focal Loss | Focal Loss en lugar de CE | wav2vec2-base, distilbert, vit-base | 256 / - / - | Focal Loss | N/A | N/A | - | ablation_13f |
| 20 | 13d K-Fold | Mismo que 13d, **K-Fold** | wav2vec2-base, distilbert, vit-base | 256 / - / - | K-Fold | N/A | N/A | - | ablation_13d_kfold |
| 21 | Hierarchical Fusion | Trimodal: **Fusión jerárquica** (Audio+Texto → Visión), LSTM | wav2vec2-base, distilbert, vit-base | 256 / 0.4 / 4 | Hierarchical | **74.64%** | - | ~6.9% | ablation_21 |
| 22 | Tensor Fusion | **Tensor Fusion Network** (producto exterior) | wav2vec2-base, distilbert, vit-base | 256 / 0.4 / - | Tensor Fusion | N/A | N/A | - | ablation_22 |
| 23 | Bilinear Fusion | **Bilinear Pooling** | wav2vec2-base, distilbert, vit-base | 256 / 0.4 / - | Bilinear | N/A | N/A | - | ablation_23 |
| 24 | Adaptive Weighted Fusion | **Fusión ponderada adaptativa** (pesos aprendidos) | wav2vec2-base, distilbert, vit-base | 256 / 0.4 / - | Adaptive weights | N/A | N/A | - | ablation_24 |
| 25 | Enhanced Cross-Attention | **TriModalEnhancedCrossAttentionFusion** (3 capas), LSTM | wav2vec2-base, distilbert, vit-base | 256 / 0.4 / 3 layers, 4 heads | Enhanced CA | **77.44%** | 77.09% | 4.86% | Base 31-35 |
| 26 | Gated Fusion | **GMU** (Gated Multimodal Unit) | wav2vec2-base, distilbert, vit-base | 256 / 0.4 / - | GMU | N/A | N/A | - | ablation_26 |
| 27 | Late Fusion | **Late Fusion** (logits/prob concatenación) | wav2vec2-base, distilbert, vit-base | 256 / 0.4 / - | Late fusion | N/A | N/A | - | ablation_27 |
| 28 | GLU Fusion | **GLU Fusion** (Gated Linear Units) | wav2vec2-base, distilbert, vit-base | 256 / 0.4 / - | GLU | **74.27%** | - | ~7.4% | ablation_28 |
| 29 | 25 K-Fold | Igual que 25, **K-Fold** 5 folds | wav2vec2-base, distilbert, vit-base | 256 / 0.4 / 3 | Enhanced CA, K-Fold | **75.71%** (prom) | 74.87% | - | Std 4.19% |
| 30 | Dirichlet | Enhanced CA + **Dirichlet Evidence** head | wav2vec2-base, distilbert, vit-base | 512 / 0.4 / 3, 8 heads | Dirichlet | **75.24%** | 75.06% | 9.78% | Mejor val, overfitting |
| 31 | Temporal Alignment | Enhanced CA + **Temporal Alignment (DTW)** | wav2vec2-base, distilbert, vit-base | 256 / 0.4 / 3 | Temporal align | 73.36% | 73.28% | ~8.4% | ablation_31 |
| 32 | Temporal Bucketing | Enhanced CA + **Temporal Bucketing** (~400ms) | wav2vec2-base, distilbert, vit-base | 256 / 0.4 / 3 | Bucketing | 73.26%-73.80% | 78.40% | ~7.9% | ablation_32 |
| 33 | Noise-Aware SpeechCue | Enhanced CA + **Noise-Aware SpeechCue** ([UNCERTAIN]) | wav2vec2-base, distilbert, vit-base | 256 / 0.4 / 3 | Noise-aware | 72.96% | 72.98% | ~6.5% | ablation_33 |
| 34 | Conflict-Aware Cross-Attention | **Conflict-Aware** fusión (detección conflicto + ponderación) + LSTM | wav2vec2-base*, distilbert, vit-base | 256 / 0.4 / 3, 4 heads | Conflict-Aware | **77.98%** | 77.59% | **2.68%** | Mejor test, mejor ejecución |
| 35 | Modal-Stratified AL | Enhanced CA + **Modal-Stratified Active Learning** | wav2vec2-base, distilbert, vit-base | 256 / 0.4 / 3 | Active Learning | **74.05%** | 73.74% | 6.32% | ablation_35 |
| 36 | 36A Dirichlet+Conflict | **Conflict-Aware** + **Dirichlet** head | wav2vec2-base, distilbert, vit-base | 256 / 0.5 / 3 | Conflict-Aware, Dirichlet | N/A | N/A | - | ablation_36a |
| 37 | Dirichlet+Conflict-Aware | Dirichlet + Conflict-Aware (config 37) | wav2vec2-base, distilbert, vit-base | 256 / - / 3 | Dirichlet, Conflict-Aware | N/A | N/A | - | ablation_37 |
| 38 | OpenSMILE + Gates + Residual | **OpenSMILE** (eGeMAPSv02) + attention gates + residual, Conflict-Aware | wav2vec2-base, distilbert, vit-base | 384 / 0.3 / 3 | OpenSMILE, gates, residual | N/A | N/A | - | ablation_38 |
| 39 | 36C Hybrid / SOTA | Conflict-Aware + Dirichlet hybrid o **ACN+MIC+BCAF** (según config) | wav2vec2-base, distilbert, vit-base | 256 / - / 3 | Dirichlet hybrid / SOTA | N/A | N/A | - | ablation_36c / 39 |
| 40 | WavLM + Conflict-Aware | **WavLM-base** + Conflict-Aware (trimodal, misma línea que 34) | **wavlm-base**, distilbert, vit-base | 512 / 0.5 / 3, 8 heads | WavLM, Conflict-Aware | **70.32%** | 70.12% | 0.18% | post-fix en repo |
| 41 | WavLM + ViT + CA **K-Fold** | Igual que 40, validación **5 folds** (leave-one-session-out) | wavlm-base, distilbert, vit-base | según YAML | K-Fold | **60.16%** (prom folds) | 58.95% | 1.70% | media_kfold; checkpoint actual |
| 42 | WavLM + ViT + CA split | Igual que 40, **split único** (train 1–3 / val 4 / test 5) | wavlm-base, distilbert, vit-base | según YAML | WavLM trimodal | **68.33%** | 67.88% | -1.91% | gap negativo en snapshot |
| 43 | Training improvements | **Ablation43Model**: Conflict-Aware + focal + SupCon + aug | wav2vec2*, distilbert, vit | según YAML | Focal, SupCon, aug | **56.83%** | 51.75% | -1.27% | run en repo débil; reentrenar |
| 44 | + Transformer head | Como 34 + **TransformerClassificationHead** | wav2vec2-base, distilbert, vit | según YAML | Conflict-Aware | **47.73%** | 41.24% | 0.59% | posible run incompleto |
| 45 | + MultiScale head | Como 34 + **MultiScaleTemporal** head | wav2vec2-base, distilbert, vit | según YAML | MultiScale head | **79.23%** | 78.68% | **2.91%** | mejor UA en agregador abril 2026 |
| 46 | + CNN1D head | Como 34 + **CNN1D** head | wav2vec2-base, distilbert, vit | según YAML | CNN1D head | N/A | N/A | - | sin `final_results` en repo |
| 47 | SupCon 2 vistas | Conflict-Aware + SupCon **audio+texto** pre-fusión | wav2vec2-base, distilbert, vit | según YAML | SupCon, **Ablation47Model** | **63.00%** | 60.29% | 4.40% | ablation_47 |
| 48 | SupCon 3 vistas | Conflict-Aware + SupCon **audio+texto+visión** | wav2vec2-base, distilbert, vit | según YAML | SupCon 3v, **Ablation48Model** | **77.28%** | 76.96% | 4.97% | ablation_48 |
| 49 | SupCon 3v + WavLM | Como 48 con encoder **WavLM-base** | wavlm-base, distilbert, vit | según YAML | SupCon, WavLM | **74.78%** | 74.75% | 6.67% | ablation_49 |
| 50 | SupCon 3v + Emotion2Vec | Como 48 con **Emotion2Vec** (FunASR) | emotion2vec, distilbert, vit | según YAML | SupCon, FunASR | N/A | N/A | - | sin resultado en repo; requiere funasr |
| 52 | SpecAug 49 | Como **49** sin waveform aug + **SpecAug** en log-mel→ViT | wavlm-base, distilbert, vit | según YAML | WavLM, SupCon, SpecAug mel | **76.82%** | 76.53% | 3.02% | ablation_52 |
| 53 | Weighted hidden WavLM | Como 52 + **weighted hidden_states** WavLM | wavlm-base, distilbert, vit | según YAML | WavLM layers | N/A | N/A | - | sin `final_results` en repo |
| 54 | WavLM-base+ weighted | Como **53** con **wavlm-base-plus** | wavlm-base-plus, distilbert, vit | según YAML | Base+ weighted | **75.74%** | 75.62% | 8.64% | ablation_54 |

*En el análisis reportado, la mejor ejecución de Ablation 34 usa la arquitectura trimodal con Wav2Vec2; el YAML en repo puede variar (ej. phase1 WavLM-large bimodal). **Métricas 41–54** en la tabla: snapshot de checkpoints en disco (abril 2026), no necesariamente el mejor entrenamiento posible por estudio.

**Leyenda**: Test UA = Unweighted Accuracy test; Gap = Val UA − Test UA. N/A = sin resultados de test en este documento (ejecutar experimento o ver checkpoints).

---

## 📈 Análisis por Serie de Experimentos

### Serie 01-10: Arquitecturas Base Bimodales (4 clases)

#### Ablation 01: Baseline Unfreeze
- **Validación UA**: 54.97%
- **Test UA**: 55.31%
- **Observaciones**: Modelo base con rendimiento bajo, probablemente debido a las 7 clases y arquitectura bimodal simple.

**Conclusión**: Necesidad de mejoras arquitectónicas significativas.

---

### Serie 11-13: Transición a Trimodal (4 clases)

#### Ablation 13: Modelo Base Trimodal (ViT + SpeechCue + Unfreeze)
- **Validación UA**: 82.05%
- **Test UA**: 74.46%
- **Gap Val-Test**: 7.59%
- **Mejora vs Baseline**: +19.15% en test

**Componentes Clave**:
- ✅ Trimodal: Audio + Texto + Visual (ViT)
- ✅ SpeechCue: Prosodia inyectada en texto
- ✅ Unfreeze parcial de encoders

**Análisis**: Reducción a 4 clases y adición de modalidad visual mejoraron significativamente el rendimiento.

---

### Serie 21-28: Mecanismos de Fusión Avanzados

#### Ablation 21: Hierarchical Fusion
- **Validación UA**: 81.57%
- **Test UA**: 74.64%
- **Gap Val-Test**: ~6.9%

#### Ablation 25: Enhanced Cross-Attention ⭐ (Base para 31-35)
- **Validación UA**: 82.30%
- **Test UA**: 77.44%
- **Gap Val-Test**: 4.86%
- **F1 Macro**: 77.09%
- **Epochs**: 51
- **Mejora vs Ablation 13**: +2.98% en test

**Características**:
- Múltiples capas de cross-attention (3 capas)
- Mejor modelado de interacciones multimodales
- **Seleccionado como base para experimentos avanzados**

#### Ablation 28: GLU Fusion
- **Validación UA**: 81.68%
- **Test UA**: 74.27%
- **Gap Val-Test**: ~7.4%

**Ranking Serie 21-28** (por Test UA):
1. **Ablation 25**: 77.44% ⭐
2. Ablation 21: 74.64%
3. Ablation 28: 74.27%

---

### Serie 30-35: Innovaciones Multimodales Avanzadas

Todos basados en **Ablation 25** (Enhanced Cross-Attention).

#### Ablation 30: Dirichlet Fusion
- **Validación UA**: **85.02%** 🏆 (Mejor en validación)
- **Test UA**: **75.24%**
- **Gap Val-Test**: 9.78%
- **F1 Macro**: 75.06%
- **Epochs**: 85

**Análisis**:
- ✅ **Mejor rendimiento en validación** de todos los experimentos
- ⚠️ Gap significativo val-test sugiere overfitting
- ✅ Buen rendimiento en Angry (87.05% accuracy, 86.74% F1)
- ⚠️ Neutral con F1 bajo (56.56%)

#### Ablation 31: Temporal Alignment Learning
- **Validación UA**: 81.74%
- **Test UA**: 73.36%
- **Gap Val-Test**: ~8.4%
- **F1 Macro**: 73.28%
- **Epochs**: 45

**Análisis**:
- ❌ No mejoró sobre Ablation 25
- ⚠️ Alineación temporal puede no ser crítica para este dataset
- Hipótesis: IEMOCAP ya está bien segmentado

#### Ablation 32: Temporal Bucketing
- **Validación UA**: 81.14%
- **Test UA**: 73.26%
- **Gap Val-Test**: ~7.9%
- **F1 Macro**: 78.40%
- **Epochs**: 32

**Análisis**:
- ❌ Rendimiento similar a Ablation 31
- ✅ Mejor F1 macro que otros
- ⚠️ División en buckets puede perder contexto global

#### Ablation 33: Noise-Aware SpeechCue
- **Validación UA**: 79.48%
- **Test UA**: 72.96%
- **Gap Val-Test**: ~6.5%
- **F1 Macro**: 72.98%
- **Epochs**: 45

**Análisis**:
- ❌ Peor rendimiento que Ablation 25
- ⚠️ Detección de ruido puede estar introduciendo ruido adicional
- Hipótesis: Dataset limpio no requiere detección de ruido

#### Ablation 34: Conflict-Aware Cross-Attention 🏆 (Mejor en Test)
- **Validación UA**: **80.66%** (Mejor ejecución: 20251210_191151)
- **Test UA**: **77.98%** 🏆
- **Gap Val-Test**: **2.68%** (Excelente generalización)
- **F1 Macro**: 77.59%
- **Epochs**: 23

**Análisis**:
- ✅ **Mejor rendimiento en test** de todos los experimentos
- ✅ **Mejor generalización** (gap más pequeño: 2.68%)
- ✅ Manejo explícito de contradicciones multimodales
- ✅ Buen balance de clases (Neutral: 72.16% accuracy, 62.50% F1)
- ✅ Excelente en Angry (85.61% accuracy, 85.00% F1)
- ⚠️ Nota: Múltiples ejecuciones muestran variabilidad (test UA: 71.85% - 77.98%)

**Ranking Serie 30-35** (por Test UA - Mejor Ejecución):
1. **Ablation 34**: 77.98% 🏆 (Gap: 2.68%)
2. **Ablation 30**: 75.24% (Gap: 9.78%)
3. Ablation 35: 74.05% (Gap: 6.32%)
4. Ablation 32: 73.80% (Gap: 8.91%)
5. Ablation 31: 72.96% (Gap: 8.78%)
6. Ablation 33: 73.36% (Gap: 7.78%)

#### Ablation 35: Modal-Stratified Active Learning
- **Validación UA**: 80.37%
- **Test UA**: 74.05%
- **Gap Val-Test**: ~6.3%
- **F1 Macro**: 73.74%
- **Epochs**: 33

**Análisis**:
- ⚠️ Rendimiento intermedio
- ✅ Gap val-test más pequeño (buena generalización)
- ⚠️ Active Learning requiere modificación del training loop (no solo arquitectura)

---

### Serie 36-39: Conflict-Aware + Dirichlet y SOTA

Estudios posteriores basados en **Ablation 34** (Conflict-Aware Cross-Attention):

- **Ablation 36A**: Conflict-Aware + Dirichlet (Pure) - Combina fusión conflict-aware con cabeza Dirichlet
- **Ablation 36B**: Conflict-Aware + Focal Loss - Focal Loss con pesos para neutral
- **Ablation 36C**: Conflict-Aware + Dirichlet Hybrid - Dirichlet con regularización para neutral
- **Ablation 37**: Dirichlet + Conflict-Aware - Configuración específica combinando ambos
- **Ablation 38**: OpenSMILE + Attention Gates + Residual - Rama eGeMAPSv02, gates, conexiones residuales
- **Ablation 39**: ACN + MIC + BCAF - Audio Context Network, Modal Invariance Constraint, Bidirectional Connection Attention Fusion

### Serie 41-54: WavLM, cabezas alternativas, SupCon y mejoras de audio

Continuación de la línea **Conflict-Aware** (y familia SupCon) tras el estudio 40. Resumen de objetivos; **métricas test** según agregación abril 2026 sobre `checkpoints/` (mejor run por estudio salvo K-Fold):

| Study | Idea | UA test (snapshot repo) | Notas |
|-------|------|-------------------------|--------|
| **41** | WavLM-base + ViT + Conflict-Aware, **K-Fold** 5 folds | **60.16%** (promedio folds) | Generalización por sesión; media por debajo del split único |
| **42** | Mismo modelo que 40, **split estándar** IEMOCAP | **68.33%** | Comparar con 34/40 bajo mismo protocolo |
| **43** | **`Ablation43Model`**: focal + SupCon + augment sobre base tipo 34 | **56.83%** | Checkpoint en repo con rendimiento bajo; la receta merece reentrenar |
| **44** | Cabeza **Transformer** en lugar de LSTM | **47.73%** | Run en repo muy débil; verificar entrenamiento |
| **45** | Cabeza **MultiScaleTemporal** | **79.23%** | **Mejor UA** del inventario actual de `final_results.json` |
| **46** | Cabeza **CNN1D** | N/A | Sin resultado publicado en repo |
| **47** | **SupCon** 2 vistas (audio+texto) pre-fusión | **63.00%** | `Ablation47Model` |
| **48** | **SupCon** 3 vistas (audio+texto+visión) | **77.28%** | `Ablation48Model` |
| **49** | SupCon 3v + encoder **WavLM-base** | **74.78%** | |
| **50** | SupCon 3v + **Emotion2Vec** | N/A | Requiere `funasr`; sin `final_results` en repo |
| **52** | Como 49 sin augment de waveform + **SpecAugment** en log-mel→ViT | **76.82%** | Regularización en rama visual |
| **53** | Como 52 + **weighted hidden states** de WavLM | N/A | Sin resultado en repo |
| **54** | Como 53 con **microsoft/wavlm-base-plus** | **75.74%** | Comparación Base vs Base+ |

El estudio **51** no está en `scripts/run_ablation.py`.

*Ver `config/ablation/ABLATION_SUMMARY.md` para YAML y mapeo study→modelo (`Ablation34Model`, `Ablation43Model`, `Ablation47Model`, `Ablation48Model`, etc.). Ranking regenerable: `python scripts/aggregate_ablation_results.py --write-docs` → `docs/RANKING_Y_RESULTADOS.md`.*

---

## 🔄 Análisis de Validación Cruzada

### Ablation 25 K-Fold (5 folds)

| Fold | Test UA | Test Accuracy | Test F1 Macro | Val UA |
|------|---------|---------------|---------------|--------|
| S1   | 74.05%  | 73.76%        | 73.17%        | 81.99% |
| S2   | 80.04%  | 79.31%        | 80.32%        | 74.84% |
| S3   | 68.75%  | 69.08%        | 68.72%        | 74.81% |
| S4   | 79.53%  | 78.87%        | 76.50%        | 70.82% |
| S5   | 76.17%  | 76.27%        | 75.62%        | 74.86% |
| **Promedio** | **75.71%** | **75.46%** | **74.87%** | **75.46%** |
| **Std Dev** | **4.19%** | **4.01%** | **4.13%** | **3.95%** |

**Análisis**:
- ✅ Promedio consistente con resultado individual (77.44%)
- ⚠️ Desviación estándar moderada (4.19%)
- ⚠️ Fold S3 muestra rendimiento significativamente menor (68.75%)
- ✅ Validación robusta del modelo

### Ablation 12 LOSO (Leave-One-Speaker-Out)

| Fold | Test UA | Test Accuracy | Test F1 Macro |
|------|---------|---------------|---------------|
| S1   | 69.60%  | 69.15%        | 66.91%        |
| S2   | -       | -             | -             |
| S3   | -       | -             | -             |
| S4   | 78.52%  | 77.68%        | 78.42%        |

**Análisis**:
- ⚠️ Rendimiento más bajo que k-fold (esperado en LOSO)
- ✅ Validación de generalización entre hablantes
- ⚠️ Variabilidad entre folds sugiere dependencia de hablante

---

## 📊 Comparación de Métricas Clave

### Top 5 modelos por UA (test) — ejecuciones históricas citadas en este documento

| Rank | Modelo | Test UA | Val UA | Gap | F1 Macro | Epochs | Ejecución |
|------|--------|---------|--------|-----|----------|--------|----------|
| 1 🥇 | **Ablation 34** | **77.98%** | 80.66% | **2.68%** | 77.59% | 23 | 20251210_191151 |
| 2 🥈 | **Ablation 25** | **77.44%** | 82.30% | 4.86% | 77.09% | 51 | 20251125_092101 |
| 3 🥉 | **Ablation 30** | **75.24%** | 85.02% | 9.78% | 75.06% | 85 | 20251126_175535 |
| 4 | **Ablation 35** | **74.05%** | 80.37% | 6.32% | 73.74% | 33 | 20251204_112541 |
| 5 | **Ablation 13** | **74.46%** | 82.05% | 7.59% | 74.18% | 34 | 20251120_061810 |

**Nota**: Ablation 34 (mejor ejecución documentada aquí) muestra el mejor balance entre rendimiento y generalización *entre estas corridas concretas*.

### Top según inventario actual de `checkpoints/` (abril 2026)

Al recorrer todos los `final_results.json` y elegir **mejor WA por estudio** (o media K-Fold), los **UA test** más altos incluyen **45** (~79.2%), **48** (~77.3%) y **52** (~76.8%), mientras que el **34** presente en disco queda ~75.4% UA en ese criterio. No contradicen el ranking histórico anterior: reflejan **otros runs** y carpetas de checkpoint. Comando: `python scripts/aggregate_ablation_results.py`.

### Análisis por Clase de Emoción

#### Ablation 34 (Conflict-Aware) - Mejor Ejecución (20251210_191151)

| Emoción | Accuracy | F1 Score |
|---------|----------|----------|
| Angry   | 85.61%   | 85.00%   |
| Happy   | 74.69%   | 81.21%   |
| Sad     | 79.46%   | 81.65%   |
| Neutral | 72.16%   | 62.50%   |

**Observaciones**:
- ✅ Buen rendimiento en Angry (clase más fácil)
- ⚠️ Neutral sigue siendo la clase más difícil
- ✅ Balance razonable entre clases

#### Ablation 30 (Dirichlet) - Mejor Validación

| Emoción | Accuracy | F1 Score |
|---------|----------|----------|
| Angry   | 87.05%   | 86.74%   |
| Happy   | 72.22%   | 80.97%   |
| Sad     | 70.54%   | 75.96%   |
| Neutral | 71.13%   | 56.56%   |

**Observaciones**:
- ✅ Excelente en Angry
- ⚠️ Neutral con F1 bajo (56.56%)
- ⚠️ Mayor gap val-test sugiere overfitting

---

## 📊 Variabilidad entre Ejecuciones

### Ablation 34: Múltiples Ejecuciones

El experimento Ablation 34 se ejecutó múltiples veces mostrando variabilidad significativa:

| Ejecución | Test UA | Val UA | Gap | Epochs |
|-----------|---------|--------|-----|--------|
| 20251210_191151 | **77.98%** 🏆 | 80.66% | **2.68%** | 23 |
| 20251209_132958 | 77.41% | 82.70% | 5.30% | - |
| 20251205_183511 | 76.95% | 83.21% | 6.26% | - |
| 20251205_155722 | 75.82% | 83.99% | 8.17% | - |
| 20251211_103903 | 71.85% | 82.27% | 10.42% | 54 |

**Análisis**:
- ⚠️ Variabilidad significativa entre ejecuciones (71.85% - 77.98%)
- ✅ La mejor ejecución muestra excelente generalización (gap 2.68%)
- ⚠️ Algunas ejecuciones muestran mayor gap val-test (posible overfitting)
- 💡 **Recomendación**: Ejecutar múltiples veces y reportar estadísticas (promedio, std, mejor)

**Implicaciones**:
- La variabilidad sugiere sensibilidad a inicialización aleatoria o condiciones de entrenamiento
- Necesidad de múltiples ejecuciones para validación robusta
- El mejor resultado (77.98%) es significativamente mejor que el promedio

---

## 🔍 Análisis de Overfitting

### Gap Validación-Test por Modelo (Mejores Ejecuciones)

| Modelo | Gap Val-Test | Interpretación |
|--------|--------------|----------------|
| **Ablation 34** | **2.68%** | ✅ **Excelente generalización** |
| Ablation 25 | 4.86% | ✅ Excelente generalización |
| Ablation 35 | 6.32% | ✅ Buena generalización |
| Ablation 13 | 7.59% | ⚠️ Generalización moderada |
| Ablation 30 | 9.78% | ⚠️ **Posible overfitting** |

**Recomendaciones**:
1. **Ablation 34 (mejor ejecución)** muestra el mejor balance val-test
2. **Ablation 25** también muestra excelente generalización
3. **Ablation 30** requiere regularización adicional o early stopping más agresivo
4. Variabilidad entre ejecuciones sugiere necesidad de múltiples runs para validación robusta

---

## 📉 Evolución del Rendimiento

### Progresión Histórica

```
Baseline (Ablation 01):     55.31% UA (Test)
↓ +19.15%
Ablation 13 (Trimodal):      74.46% UA (Test)
↓ +2.98%
Ablation 25 (Enhanced CA):   77.44% UA (Test)
↓ +0.54%
Ablation 34 (Conflict-Aware): 77.98% UA (Test) — corrida 20251210_191151  
↓ (serie 40–54: WavLM, cabezas, SupCon; ver tabla maestra y `RANKING_Y_RESULTADOS.md`)  
Checkpoints post-fix abril 2026: mayor UA en repo ~79.2% (estudio 45, un run)
```

**Observaciones**:
- ✅ Mejora clara desde baseline hasta la corrida 34 citada (orden de magnitud +22% UA)
- ✅ Ablation 13 marcó un salto significativo (trimodal)
- ✅ Ablation 25 mejoró con fusión avanzada
- ✅ Ablation 34 (mejor ejecución documentada) combina muy buen UA test con gap val-test bajo
- ℹ️ Serie **41–54**: revisar tabla maestra y `docs/RANKING_Y_RESULTADOS.md`; en abril 2026 el mayor UA entre `final_results.json` del repo fue **45** (~79.23%)

---

## 🎯 Análisis de Clases

### Clase "Neutral" - El Desafío Principal

Todos los modelos muestran dificultades con la clase Neutral:

| Modelo | Neutral F1 | Neutral Accuracy |
|--------|------------|------------------|
| Ablation 34 (mejor) | 62.50% | 72.16% |
| Ablation 34 (promedio) | 62.98% | 76.29% |
| Ablation 30 | 56.56% | 71.13% |
| Ablation 25 | 60.99% | 70.10% |
| Ablation 13 | 56.03% | 67.01% |

**Análisis**:
- ⚠️ Neutral es la clase más difícil (F1 < 65% en todos)
- ✅ Ablation 34 tiene mejor F1 para Neutral (62.98%)
- 💡 **Recomendación**: Considerar técnicas específicas para neutral:
  - Data augmentation específica
  - Loss weighting para neutral
  - Análisis de confusión neutral vs otras clases

### Clase "Angry" - La Más Fácil

| Modelo | Angry F1 | Angry Accuracy |
|--------|----------|---------------|
| Ablation 34 (mejor) | 85.00% | 85.61% |
| Ablation 30 | 86.74% | 87.05% |
| Ablation 25 | 83.39% | 84.89% |
| Ablation 34 (promedio) | 82.52% | 84.89% |

**Análisis**:
- ✅ Angry es consistentemente la clase mejor clasificada
- ✅ Características prosódicas y textuales claras para angry

---

## 💡 Conclusiones y Recomendaciones

### Conclusiones Principales

1. **Mejor Modelo General**: **Ablation 34 (Conflict-Aware Cross-Attention)** - Mejor Ejecución
   - Mejor balance entre rendimiento y generalización
   - Test UA: 77.98% 🏆
   - Gap val-test excelente: 2.68%
   - Ejecución: 20251210_191151

2. **Mejor Rendimiento en Validación**: **Ablation 30 (Dirichlet)**
   - Val UA: 85.02%
   - ⚠️ Requiere atención al overfitting

3. **Modelo Base Robusto**: **Ablation 25 (Enhanced Cross-Attention)**
   - Excelente generalización (gap 4.9%)
   - Base sólida para futuros experimentos

4. **Checkpoints post-fix (2026)** en `checkpoints/`: la **mayor UA test** agregada corresponde al estudio **45** (~79.2%); conviene versionar siempre `final_results.json` y regenerar `docs/RANKING_Y_RESULTADOS.md`.

5. **Problema identificado**: overfitting en algunos modelos (gaps 6–10%) y **alta variabilidad** entre ejecuciones y carpetas de checkpoint.

### Recomendaciones Técnicas

#### 1. Regularización y Prevención de Overfitting
- ✅ Implementar dropout más agresivo
- ✅ Aumentar weight decay
- ✅ Early stopping más temprano
- ✅ Data augmentation más robusta

#### 2. Mejora de Clase Neutral
- ✅ Focal loss con mayor peso para neutral
- ✅ Análisis de confusión neutral vs otras clases
- ✅ Data augmentation específica para neutral
- ✅ Considerar sub-clases de neutral

#### 3. Arquitectura
- ✅ **Ablation 34 (mejor ejecución)** como modelo de producción
- ✅ Considerar ensemble de Ablation 25 + Ablation 34
- ⚠️ Evitar técnicas complejas que no mejoran (Ablation 31-33)
- ⚠️ Variabilidad entre ejecuciones requiere múltiples runs para validación robusta

#### 4. Validación
- ✅ Continuar con k-fold para validación robusta
- ✅ Implementar validación en dataset externo
- ✅ Análisis de errores por clase y hablante

#### 5. Próximos experimentos
- 🔬 Completar runs con **resultados faltantes** en repo (**46, 50, 53**) y reentrenar **43–44** si los checkpoints son atípicos
- 🔬 Ensemble entre top **45**, **48**, **52** y **34** bajo el mismo split
- 🔬 Fine-tuning específico para neutral
- 🔬 Análisis de atención para interpretabilidad

---

## 📋 Métricas Detalladas por Modelo

### Ablation 34: Conflict-Aware Cross-Attention (Mejor Ejecución: 20251210_191151)

**Validación**:
- Unweighted Accuracy: 80.66%
- Weighted Accuracy: 83.10%
- F1 Macro: 80.83%
- F1 Weighted: 83.20%

**Test**:
- Unweighted Accuracy: **77.98%** 🏆
- Weighted Accuracy: 78.24%
- F1 Macro: 77.59%
- F1 Weighted: 78.78%

**Por Clase (Test)**:
- Angry: Acc 85.61%, F1 85.00%
- Happy: Acc 74.69%, F1 81.21%
- Sad: Acc 79.46%, F1 81.65%
- Neutral: Acc 72.16%, F1 62.50%

**Nota**: Este experimento se ejecutó múltiples veces. La mejor ejecución (20251210_191151) muestra excelente generalización con gap de solo 2.68%.

### Ablation 30: Dirichlet Fusion

**Validación**:
- Unweighted Accuracy: **85.02%**
- Weighted Accuracy: 84.31%
- F1 Macro: 82.74%
- F1 Weighted: 84.90%

**Test**:
- Unweighted Accuracy: 75.24%
- Weighted Accuracy: 75.69%
- F1 Macro: 75.06%
- F1 Weighted: 76.80%

**Por Clase (Test)**:
- Angry: Acc 87.05%, F1 86.74%
- Happy: Acc 72.22%, F1 80.97%
- Sad: Acc 70.54%, F1 75.96%
- Neutral: Acc 71.13%, F1 56.56%

### Ablation 25: Enhanced Cross-Attention

**Validación**:
- Unweighted Accuracy: 82.30%
- Weighted Accuracy: 81.69%
- F1 Macro: 80.27%
- F1 Weighted: 82.35%

**Test**:
- Unweighted Accuracy: 77.44%
- Weighted Accuracy: 77.84%
- F1 Macro: 77.09%
- F1 Weighted: 78.45%

**Por Clase (Test)**:
- Angry: Acc 84.89%, F1 83.39%
- Happy: Acc 75.31%, F1 82.71%
- Sad: Acc 79.46%, F1 81.28%
- Neutral: Acc 70.10%, F1 60.99%

---

## 📊 Estadísticas de Entrenamiento

### Epochs hasta Convergencia

| Modelo | Epochs | Interpretación |
|--------|--------|----------------|
| Ablation 34 (mejor) | 23 | ✅ Convergencia muy rápida |
| Ablation 32 | 32 | ✅ Convergencia rápida |
| Ablation 35 | 33 | ✅ Convergencia rápida |
| Ablation 13 | 34 | ✅ Convergencia rápida |
| Ablation 31 | 45 | ⚠️ Convergencia lenta |
| Ablation 25 | 51 | ⚠️ Convergencia lenta |
| Ablation 30 | 85 | ⚠️ **Muy lenta, posible overfitting** |

**Observaciones**:
- Modelos con mejor generalización convergen más rápido
- Ablation 34 (mejor ejecución) converge en solo 23 epochs
- Ablation 30 requiere 85 epochs (sospecha de overfitting)

---

## 🎓 Lecciones Aprendidas

### ✅ Técnicas Exitosas

1. **Enhanced Cross-Attention (Ablation 25)**
   - Múltiples capas de cross-attention mejoran interacciones
   - Base sólida para experimentos avanzados

2. **Conflict-Aware Attention (Ablation 34)**
   - Manejo explícito de contradicciones mejora robustez
   - Mejor balance val-test

3. **Trimodal Architecture (Ablation 13)**
   - Adición de modalidad visual (ViT) mejora significativamente
   - SpeechCue aporta información prosódica valiosa

### ❌ Técnicas que No Mejoraron

1. **Temporal Alignment (Ablation 31)**
   - No mejora sobre baseline
   - Dataset ya está bien segmentado

2. **Temporal Bucketing (Ablation 32)**
   - Rendimiento similar, no justifica complejidad adicional

3. **Noise-Aware SpeechCue (Ablation 33)**
   - Peor rendimiento que baseline
   - Dataset limpio no requiere detección de ruido

### 💡 Insights Clave

1. **Simplicidad vs Complejidad**: Técnicas más complejas no siempre mejoran
2. **Overfitting**: Gap val-test es crítico para selección de modelo
3. **Clase Neutral**: Requiere atención especial en futuros experimentos
4. **Generalización**: Modelos con mejor gap val-test son más confiables

---

## 📝 Notas Finales

Este análisis se basa en los resultados disponibles en los checkpoints. Para un análisis más profundo, se recomienda:

1. Análisis de matrices de confusión detalladas
2. Visualización de curvas de entrenamiento
3. Análisis de atención para interpretabilidad
4. Validación en dataset externo
5. Análisis de errores por hablante y sesión

---

**Fin del análisis**

*Tabla maestra ampliada a estudios **1–50, 52–54** (sin 51). Métricas 41–54 y ranking: datos agregados de `checkpoints/**/final_results.json` (abril 2026) vía `scripts/aggregate_ablation_results.py`; ver también `docs/RANKING_Y_RESULTADOS.md`. Las secciones detalladas de series 01–35 conservan cifras de corridas históricas donde aplica.*  
*Catálogo YAML: `config/ablation/ABLATION_SUMMARY.md`. Modelos: `src/ablation_studies/`.*

---

*Última actualización del documento: **abril 2026***.
