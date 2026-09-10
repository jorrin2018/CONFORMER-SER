# Estudios de Ablación - Conformer Multimodal SER

Este módulo contiene la implementación de los modelos usados por los estudios de ablación disponibles en `scripts/run_ablation.py`: estudios **1-50, 52-63**. El estudio 51 no está definido en el listado de ejecución actual.

## 📋 Descripción General

Los estudios de ablación analizan sistemáticamente la contribución de diferentes componentes del modelo.

**Rama ViT y datos**: IEMOCAP se carga como **waveform + texto** (y SpeechCue cuando aplica). El **log-mel que usa el ViT no es una feature precalculada del dataset**: se obtiene en el **forward** del modelo a partir del **mismo** audio del batch (ver `ablation_11_model.py`, `_waveform_to_vit_image`). Cualquier mención a “mel para ViT” en configs o docs se refiere a esa vista derivada, no a un archivo mel en el repositorio de datos.

**Mejores resultados**: Ablation 34 (77.98% UA, gap 2.68%) | Ablation 25 (77.44% UA, base fusión)

**Evolución del Proyecto**:
- **Estudios 01-10**: Enfocados en 7 clases y arquitectura Bimodal (Audio+Texto).
- **Estudios 11-13**: Enfocados en **4 clases** (Angry, Happy, Sad, Neutral) y arquitectura **Trimodal** (Audio+Texto+Visual).
- **Estudios 20-28**: Mecanismos de fusión avanzados (basados en Ablation 13c).
- **Estudios 30**: Dirichlet Evidence Learning (modelado de incertidumbre).
- **Estudios 31-35**: Innovaciones multimodales avanzadas (**basados en Ablation 25** - mejor resultado).
- **Estudios 36-39**: Conflict-Aware + Dirichlet, OpenSMILE, SOTA (**basados en Ablation 34**).
- **Estudios 40-42**: Variantes WavLM + Conflict-Aware y K-Fold.
- **Estudio 43**: `Ablation43Model` con mejoras de entrenamiento.
- **Estudios 44-46**: Ablation 34 con cabezas Transformer, MultiScaleTemporal y CNN1D.
- **Estudios 47-50, 52-57**: SupCon, WavLM, Emotion2Vec, SpecAugment, weighted hidden states y **55-57** (receta Ablation 45 + encoder audio alternativo).
- **Estudios 58-63**: combos SOTA (MultiScaleTemporal + SupCon + SpecAugment), pooling atencional de audio y ajuste fino del estudio 45. Usan `Ablation48Model` (58, 60, 61) o `Ablation34Model` (59, 62, 63).

## 🗂️ Estructura de Archivos

```
src/ablation_studies/
├── ablation_01_model.py ... ablation_11_model.py   # Bimodal/Trimodal base
├── ablation_13_model.py                            # Modelo Base Trimodal (ViT + SpeechCue + Unfreeze)
├── ablation_21_hierarchical_fusion.py ... ablation_28_glu_fusion.py  # Fusión avanzada
├── ablation_25_enhanced_cross_attention.py         # Enhanced Cross-Attention (mejor resultado)
├── ablation_30_dirichlet.py                        # Dirichlet Evidence Learning
├── ablation_31_model.py                            # Temporal Alignment Learning
├── ablation_32_model.py                            # Temporal Bucketing
├── ablation_33_model.py                            # Noise-Aware SpeechCue
├── ablation_34_model.py                            # Conflict-Aware Cross-Attention
├── ablation_35_model.py                            # Modal-Stratified Active Learning
├── ablation_36a_model.py                           # Conflict-Aware + Dirichlet (Pure)
├── ablation_36b_model.py                           # Conflict-Aware + Focal Loss
├── ablation_36c_model.py                           # Conflict-Aware + Dirichlet Hybrid
├── ablation_38_model.py                            # OpenSMILE + Attention Gates + Residual
├── ablation_39_model.py                            # ACN + MIC + BCAF (SOTA improvements)
├── ablation_43_model.py                            # Conflict-Aware + training improvements
├── ablation_47_supcon_model.py, ablation_48_supcon_model.py  # SupCon variants
├── ablation_07_architecture.mmd                     # Diagrama Mermaid de arquitectura
├── base_config.py, classification_heads.py, utils.py
└── __init__.py
```

*Study 40-42, 44-46, 55-57*: reutilizan principalmente `Ablation34Model` con cambios declarados por YAML. *Study 43* usa `Ablation43Model`. *Studies 47-50 y 52-54* usan la familia SupCon (`Ablation47Model` / `Ablation48Model`) según el mapeo de `scripts/run_ablation.py`.

*Mapeo study→config: ver `config/ablation/ABLATION_SUMMARY.md`*

## 🔬 Estudios de Ablación (Nuevos)

### Estudios Previos (01-10)
*Consultar versiones anteriores del README para detalles de los estudios 01 al 10 (Unfreeze, LSTM Head, GIA Fusion, etc.).*

---

### Ablation 11: MelSpec ViT (Trimodal Baseline)

**Objetivo**: Evaluar el impacto de añadir una tercera rama visual que procese una vista **time-frequency derivada del waveform** (no mel almacenado en el dataset).

**Modificaciones**:
- Añade rama visual en el forward: `MelSpectrogram(audio_batch)` → log-mel opcionalmente con SpecAugment → resize a imagen → `Vision Transformer (ViT)`.
- Utiliza `TriModalCrossAttentionFusion` para integrar Audio, Texto y Visual.
- Reduce el problema a **4 clases**.

**Hipótesis**: Los espectrogramas tratados como imágenes contienen patrones globales que ViT puede capturar mejor que Conformer.

**Configuración**:
- `vision_encoder: "google/vit-base-patch16-224"`
- `fusion_type: "trimodal_cross_attention"`

---

### Ablation 12: SpeechCue LOSO

**Objetivo**: Validar la robustez del modelo mediante validación cruzada Leave-One-Speaker-Out.

**Modificaciones**:
- Utiliza la arquitectura de SpeechCue.
- Itera sobre las sesiones de IEMOCAP para asegurar independencia del hablante.

---

### Ablation 13: MelSpec ViT + SpeechCue + Unfreeze (Modelo Base)

**Objetivo**: Combinar todas las mejoras más potentes en un solo modelo.

**Modificaciones**:
- **Trimodal**: Audio + Texto + ViT.
- **SpeechCue**: Inyecta prompts de prosodia en el encoder de texto (`text.use_prosody_cues: true`).
- **Unfreeze Completo**: Descongela capas superiores de Audio (6), Texto (4) y Visión (2).
- **Cabeza LSTM**: Mantiene la clasificación secuencial.

**Configuración**:
- Combina configuraciones de Ablation 10 (SpeechCue) y Ablation 11 (ViT).
- `vision_unfreeze_last_n_layers: 2`

---

### Ablation 25: Enhanced Cross-Attention (⭐ Mejor Fusión Base)

**Objetivo**: Mejorar la fusión multimodal con múltiples capas de cross-attention. **Resultado**: 77.44% UA, gap 4.86%.

**Modificaciones**:
- **Base**: Ablation 13 (Trimodal + SpeechCue + ViT congelado).
- **Fusión**: `TriModalEnhancedCrossAttentionFusion` con múltiples capas de atención cruzada.
- **Capas de atención**: 3 capas de cross-attention para modelado más profundo de interacciones.

**Características**:
- Múltiples capas de cross-attention entre las 3 modalidades.
- Mejor captura de interacciones complejas entre modalidades.
- **Este es el modelo base para los estudios 31-35** por haber obtenido los mejores resultados.

**Configuración**:
- `fusion_type: "enhanced_cross_attention"`
- `fusion_num_layers: 3`
- `fusion_num_heads: 4`

---

## 🔬 Serie 20-28: Mecanismos de Fusión Avanzados

Estudios enfocados en diferentes estrategias de fusión multimodal, manteniendo la arquitectura base de Ablation 13c.

**Estudios principales**:
- **Ablation 21**: Fusión Jerárquica (Audio+Texto → Visión)
- **Ablation 22**: Tensor Fusion Network
- **Ablation 23**: Bilinear Pooling
- **Ablation 24**: Fusión Ponderada Adaptativa
- **Ablation 25**: Enhanced Cross-Attention ⭐ (mejor resultado)
- **Ablation 26**: Gated Multimodal Unit (GMU)
- **Ablation 27**: Late Fusion
- **Ablation 28**: GLU Fusion

*Ver `config/ablation/ABLATION_SUMMARY.md` para detalles completos.*

---

### Ablation 30: Dirichlet Evidence Learning

**Objetivo**: Modelar incertidumbre en la predicción mediante Dirichlet Evidence Loss.

**Modificaciones**:
- **Dirichlet Classification Head**: Sustituye softmax por distribución Dirichlet.
- **Evidencia aprendida**: El modelo produce evidencia por clase en lugar de logits directos.
- **Base**: Ablation 25 (Enhanced Cross-Attention).

**Características**:
- Mejor calibración de confianza.
- Captura explícita de incertidumbre epistémica.
- Útil para detección de muestras ambiguas.

---

## 🚀 Serie 31-35: Innovaciones Multimodales Avanzadas

Nuevas técnicas avanzadas para mejorar el procesamiento multimodal, basadas en **Ablation 25** (Enhanced Cross-Attention).

### Ablation 31: Temporal Alignment Learning (Adaptive DTW)

**Objetivo**: Resolver la asincronía real entre modalidades aprendiendo a alinear secuencias temporalmente.

**Modificaciones**:
- **Módulo de Alineación Temporal**: Aprende warping paths adaptativos usando DTW.
- **Aplicación**: Alinea audio y texto evento-a-evento antes de la fusión.
- **Base**: Ablation 25 (Enhanced Cross-Attention).

**Características**:
- Alineación aprendible y diferenciable.
- Regularización de suavidad para warping paths monótonos.
- Resuelve desincronías reales entre modalidades.

---

### Ablation 32: Temporal-Aligned Multimodal Bucketing

**Objetivo**: Mejorar la precisión local procesando el audio en ventanas temporales.

**Modificaciones**:
- **División en Buckets**: Divide audio en ventanas de ~400ms con solapamiento.
- **Fusión por Bucket**: Cada bucket se fusiona independientemente.
- **Conexión Temporal**: LSTM conecta la secuencia de buckets.
- **Base**: Ablation 25 (Enhanced Cross-Attention).

**Características**:
- Evita que ruido de segmentos largos contamine decisiones locales.
- Modelado temporal explícito con LSTM.
- Configurable: duración y solapamiento de buckets.

---

### Ablation 33: Noise-Aware SpeechCue

**Objetivo**: Aumentar robustez en condiciones reales detectando ruido de fondo.

**Modificaciones**:
- **Detección de Ruido**: Estima SNR del audio.
- **Token [UNCERTAIN]**: Inyecta token cuando detecta ruido.
- **Base**: Ablation 25 (Enhanced Cross-Attention).

**Características**:
- Detección automática de ruido de fondo.
- Avisa al modelo de texto sobre incertidumbre en prosodia.
- Robustez en condiciones reales.

---

### Ablation 34: Conflict-Aware Cross-Attention (🏆 Mejor Resultado)

**Objetivo**: Manejar contradicciones entre modalidades (ironía, sarcasmo). **Resultado**: 77.98% UA, gap 2.68% (mejor generalización).

**Modificaciones**:
- **Detección de Conflictos**: Identifica cuando modalidades se contradicen.
- **Ponderación por Confianza**: Asigna pesos basados en confianza cuando hay conflictos.
- **Base**: Ablation 25 (Enhanced Cross-Attention).

**Características**:
- Detecta explícitamente contradicciones (ej. texto feliz + audio enojado).
- Usa confianza aprendida para resolver conflictos.
- Maneja ironía y sarcasmo.

**Método de detección de conflicto**: En la config, `model.conflict_aware.conflict_detection_method` puede ser `prediction_divergence` (por defecto), `feature_distance` o `learned`. Ver [docs/CONFLICT_DETECTION.md](../../docs/CONFLICT_DETECTION.md) para cambiar entre ellos.

---

### Ablation 35: Modal-Stratified Active Learning

**Objetivo**: Eficiencia de datos identificando qué modalidad necesita más etiquetas.

**Modificaciones**:
- **Estrategia de Entrenamiento**: Identifica muestras con alta incertidumbre por modalidad.
- **Solicitud Selectiva**: Pide etiquetas solo para la modalidad confusa.
- **Base**: Ablation 25 (Enhanced Cross-Attention).

**Características**:
- Reduce costos de anotación.
- Eficiencia de datos.
- **Nota**: Requiere modificación del training loop (no solo arquitectura).

---

## 🚀 Serie 36-39: Conflict-Aware + Dirichlet y SOTA

Estudios que combinan Conflict-Aware Cross-Attention (Ablation 34) con Dirichlet, Focal Loss y mejoras state-of-the-art.

### Ablation 36A: Conflict-Aware + Dirichlet (Pure)

**Objetivo**: Combinar Conflict-Aware Fusion con Dirichlet Classification Head.

**Modificaciones**:
- **Base**: Ablation 34 + Ablation 30.
- **Regularización**: Dropout 0.5, Weight decay 0.1, Label smoothing 0.1.

---

### Ablation 36B: Conflict-Aware + Focal Loss

**Objetivo**: Mejorar clasificación de neutral con Focal Loss y pesos específicos.

**Modificaciones**:
- **Base**: Ablation 34 (Conflict-Aware).
- **Focal Loss**: gamma=2.5, peso 2.0x para neutral.

---

### Ablation 36C: Conflict-Aware + Dirichlet Hybrid

**Objetivo**: Dirichlet con regularización adicional para neutral.

**Modificaciones**:
- **Base**: Ablation 34 + Ablation 30.
- **DirichletNeutralRegularizedLoss**: Regularización específica para clase neutral.

---

### Ablation 37: Dirichlet + Conflict-Aware

**Objetivo**: Combinar Dirichlet Evidence Loss con Conflict-Aware Cross-Attention.

**Modificaciones**:
- Usa arquitectura Ablation 36A con configuración específica.
- Aprovecha modelado de incertidumbre junto con detección de conflictos.

---

### Ablation 38: OpenSMILE + Attention Gates + Residual

**Objetivo**: Añadir características de bajo nivel (eGeMAPSv02) con gates y conexiones residuales.

**Modificaciones**:
- **Rama OpenSMILE**: 88 features eGeMAPSv02.
- **Attention Gates**: Filtran características importantes.
- **Cross-attention**: Fusión Wav2Vec2 ↔ OpenSMILE antes de fusión modal.
- **Conexiones residuales**: Conservan contexto en puntos clave.

---

### Ablation 39: ACN + MIC + BCAF (SOTA Improvements)

**Objetivo**: Mejoras state-of-the-art sobre Conflict-Aware.

**Modificaciones**:
- **Audio Context Network (ACN)**: Antes del pooling.
- **Modal Invariance Constraint (MIC)**: Con SKL.
- **Bidirectional Connection Attention Fusion (BCAF)**: Con autoencoder.

---

### Ablation 40: WavLM-base + Conflict-Aware Cross-Attention

**Objetivo**: Evaluar si WavLM mejora sobre Wav2Vec2 manteniendo la misma arquitectura trimodal Conflict-Aware.

**Modificaciones**:
- **Encoder de audio**: `microsoft/wavlm-base` en lugar de `facebook/wav2vec2-base`.
- Resto idéntico a Ablation 34: Conflict-Aware Cross-Attention, ViT, DistilBERT, LSTM head.

**Config**: `config/ablation/ablation_40_wavlm_conflict_aware.yaml`. Modelo: `Ablation34Model`.

**Método de detección de conflicto**: Igual que Ablation 34; se puede elegir `prediction_divergence`, `feature_distance` o `learned` en `model.conflict_aware.conflict_detection_method`. Ver [docs/CONFLICT_DETECTION.md](../../docs/CONFLICT_DETECTION.md).

---

## 🚀 Serie 41-57: WavLM, SupCon y mejoras recientes

Estos estudios extienden la línea Conflict-Aware con validación K-Fold, nuevas cabezas de clasificación, pérdidas contrastivas y encoders de audio más robustos.

**Resumen**:
- **41**: WavLM-base + ViT + Conflict-Aware con K-Fold.
- **42**: WavLM-base + ViT + Conflict-Aware en split simple.
- **43**: Conflict-Aware + Focal Loss + Supervised Contrastive Loss + augmentación (`Ablation43Model`).
- **44-46**: Conflict-Aware con cabezas Transformer, MultiScaleTemporal y CNN1D.
- **47-48**: Conflict-Aware + SupCon pre-fusión con 2 o 3 vistas.
- **49**: SupCon 3 vistas con WavLM-base.
- **50**: SupCon 3 vistas con Emotion2Vec; requiere `funasr`.
- **52**: Variante de 49 sin waveform augmentation y con SpecAugment sobre el **log-mel del ViT** (derivado del mismo waveform en el forward).
- **53**: WavLM con suma ponderada aprendible de hidden states.
- **54**: Misma receta que 53 con `microsoft/wavlm-base-plus`.
- **55-57**: Misma arquitectura y entrenamiento que **Ablation 45** (MultiScaleTemporal + Conflict-Aware); solo cambia el encoder de audio: **55** WavLM-base, **56** HuBERT-base-ls960, **57** Emotion2Vec (`funasr`).

El catálogo completo y el mapeo study -> YAML -> modelo se mantiene en `config/ablation/ABLATION_SUMMARY.md`.

---

## 🚀 Uso

### Ejecutar Estudios de Ablación

```powershell
# Estudios 1-13: Base, fusión inicial, trimodal
python scripts/run_ablation.py --study 1   # ... hasta 13

# Estudio 25: Enhanced Cross-Attention (mejor fusión base)
python scripts/run_ablation.py --study 25

# Estudio 30: Dirichlet Evidence Learning
python scripts/run_ablation.py --study 30

# Estudios 31-35: Innovaciones Avanzadas (base Ablation 25)
python scripts/run_ablation.py --study 31  # Temporal Alignment
python scripts/run_ablation.py --study 32  # Temporal Bucketing
python scripts/run_ablation.py --study 33  # Noise-Aware SpeechCue
python scripts/run_ablation.py --study 34  # Conflict-Aware (mejor resultado)
python scripts/run_ablation.py --study 35  # Active Learning

# Estudios 36-40: Conflict-Aware + Dirichlet y SOTA (base Ablation 34)
python scripts/run_ablation.py --study 36  # Dirichlet + Conflict (36A)
python scripts/run_ablation.py --study 37  # Dirichlet + Conflict-Aware
python scripts/run_ablation.py --study 38  # OpenSMILE + Residual
python scripts/run_ablation.py --study 39  # 36C / ACN+MIC+BCAF
python scripts/run_ablation.py --study 40  # WavLM-base + Conflict-Aware

# Estudios recientes
python scripts/run_ablation.py --study 43  # Training improvements
python scripts/run_ablation.py --study 48  # SupCon 3 vistas
python scripts/run_ablation.py --study 54  # WavLM-base-plus + weighted hidden states
python scripts/run_ablation.py --study 55  # Ablation 45 + WavLM-base
python scripts/run_ablation.py --study 56  # Ablation 45 + HuBERT
python scripts/run_ablation.py --study 57  # Ablation 45 + Emotion2Vec (funasr)
```

### Configuración

Todos los estudios usan archivos YAML en `config/ablation/`:
- `ablation_XX_nombre.yaml` - Configuración del estudio
- Ver `config/ablation/ABLATION_SUMMARY.md` para resumen completo

---

## 📊 Arquitecturas Base

**Estudios 31-35** (base: Ablation 25):
- **Audio**: Wav2Vec2 (fine-tuning parcial - últimas 6 capas)
- **Texto**: DistilBERT con SpeechCue (fine-tuning parcial - últimas 4 capas)
- **Visión**: ViT congelado (log-mel **calculado desde el waveform** y pasado como imagen)
- **Fusión**: Enhanced Cross-Attention (3 capas de cross-attention)
- **Clasificación**: LSTM Head (2 capas bidireccionales)

**Estudios 34, 36-57** (Conflict-Aware y variantes):
- Misma base que Ablation 25, pero con **Conflict-Aware Cross-Attention**.
- Detecta contradicciones entre modalidades y pondera por confianza.
- Ablation 38 añade rama OpenSMILE; Ablation 39 añade ACN, MIC y BCAF.
- Ablation 40-42 evalúan WavLM y K-Fold.
- Ablation 43-54 exploran mejoras de entrenamiento, SupCon, Emotion2Vec, SpecAugment y weighted hidden states.
- **Ablation 55-57**: ablación sobre el **mejor blueprint del estudio 45** (MultiScale head) con **WavLM, HuBERT y Emotion2Vec** como única variable de encoder de audio.

---

*Última actualización: Abril 2026*
