# Estudios de Ablación - Conformer Multimodal SER

Este módulo contiene la implementación de múltiples estudios de ablación para evaluar el impacto de diferentes modificaciones arquitectónicas (y de interfaz con el lenguaje) sobre el modelo base de reconocimiento de emociones multimodal.

## 📋 Descripción General

Los estudios de ablación están diseñados para analizar sistemáticamente la contribución de diferentes componentes del modelo.

**Evolución del Proyecto**:
- **Estudios 01-10**: Enfocados en 7 clases y arquitectura Bimodal (Audio+Texto).
- **Estudios 11-13**: Enfocados en **4 clases** (Angry, Happy, Sad, Neutral) y arquitectura **Trimodal** (Audio+Texto+Visual).
- **Estudios 20-28**: Mecanismos de fusión avanzados (basados en Ablation 13c).
- **Estudios 31-35**: Innovaciones multimodales avanzadas (**basados en Ablation 25** - mejor resultado).

## 🗂️ Estructura de Archivos

```
src/ablation_studies/
├── ablation_11_model.py           # Modelo Trimodal (ViT)
├── ablation_12_model.py           # (Reservado para LOSO)
├── ablation_13_model.py           # Modelo Base Trimodal (ViT + SpeechCue + Unfreeze)
├── ablation_25_enhanced_cross_attention.py  # Enhanced Cross-Attention (mejor resultado)
├── ablation_31_model.py           # Temporal Alignment Learning
├── ablation_32_model.py           # Temporal Bucketing
├── ablation_33_model.py           # Noise-Aware SpeechCue
├── ablation_34_model.py           # Conflict-Aware Attention
├── ablation_35_model.py           # Modal-Stratified Active Learning
└── ... (archivos anteriores 01-10, 21-28)
```

## 🔬 Estudios de Ablación (Nuevos)

### Estudios Previos (01-10)
*Consultar versiones anteriores del README para detalles de los estudios 01 al 10 (Unfreeze, LSTM Head, GIA Fusion, etc.).*

---

### Ablation 11: MelSpec ViT (Trimodal Baseline)

**Objetivo**: Evaluar el impacto de añadir una tercera rama visual que procese espectrogramas.

**Modificaciones**:
- Añade rama visual: `MelSpectrogram` -> `Vision Transformer (ViT)`.
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

### Ablation 25: Enhanced Cross-Attention (⭐ Mejor Resultado)

**Objetivo**: Mejorar la fusión multimodal con múltiples capas de cross-attention.

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

### Ablation 34: Conflict-Aware Cross-Attention

**Objetivo**: Manejar contradicciones entre modalidades (ironía, sarcasmo).

**Modificaciones**:
- **Detección de Conflictos**: Identifica cuando modalidades se contradicen.
- **Ponderación por Confianza**: Asigna pesos basados en confianza cuando hay conflictos.
- **Base**: Ablation 25 (Enhanced Cross-Attention).

**Características**:
- Detecta explícitamente contradicciones (ej. texto feliz + audio enojado).
- Usa confianza aprendida para resolver conflictos.
- Maneja ironía y sarcasmo.

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

## 🚀 Uso

### Ejecutar Estudios de Ablación

```bash
# Estudio 11: Trimodal ViT Base
python scripts/run_ablation.py --study 11

# Estudio 13: Modelo Base Trimodal
python scripts/run_ablation.py --study 13

# Estudio 25: Enhanced Cross-Attention (mejor resultado)
python scripts/run_ablation.py --study 25

# Estudios 31-35: Innovaciones Avanzadas (basados en Ablation 25)
python scripts/run_ablation.py --study 31  # Temporal Alignment
python scripts/run_ablation.py --study 32  # Temporal Bucketing
python scripts/run_ablation.py --study 33  # Noise-Aware SpeechCue
python scripts/run_ablation.py --study 34  # Conflict-Aware Attention
python scripts/run_ablation.py --study 35  # Active Learning
```

### Configuración

Todos los estudios usan archivos YAML en `config/ablation/`:
- `ablation_XX_nombre.yaml` - Configuración del estudio
- Ver `config/ablation/ABLATION_SUMMARY.md` para resumen completo

---

## 📊 Arquitectura Base (Ablation 25)

Los estudios 31-35 comparten la siguiente arquitectura base:

- **Audio**: Wav2Vec2 (fine-tuning parcial - últimas 6 capas)
- **Texto**: DistilBERT con SpeechCue (fine-tuning parcial - últimas 4 capas)
- **Visión**: ViT congelado (Mel-spectrograms como imágenes)
- **Fusión**: Enhanced Cross-Attention (3 capas de cross-attention)
- **Clasificación**: LSTM Head (2 capas bidireccionales)
