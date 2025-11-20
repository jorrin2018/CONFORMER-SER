# Estudios de Ablación - Conformer Multimodal SER

Este módulo contiene la implementación de 13 estudios de ablación para evaluar el impacto de diferentes modificaciones arquitectónicas (y de interfaz con el lenguaje) sobre el modelo base de reconocimiento de emociones multimodal.

## 📋 Descripción General

Los estudios de ablación están diseñados para analizar sistemáticamente la contribución de diferentes componentes del modelo.

**Evolución del Proyecto**:
- **Estudios 01-10**: Enfocados en 7 clases y arquitectura Bimodal (Audio+Texto).
- **Estudios 11-13**: Enfocados en **4 clases** (Angry, Happy, Sad, Neutral) y arquitectura **Trimodal** (Audio+Texto+Visual).

## 🗂️ Estructura de Archivos

```
src/ablation_studies/
├── ablation_11_model.py           # Modelo Trimodal (ViT)
├── ablation_12_model.py           # (Reservado para LOSO)
├── ablation_13_model.py           # Modelo Final (ViT + SpeechCue + Unfreeze)
└── ... (archivos anteriores 01-10)
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

### Ablation 13: MelSpec ViT + SpeechCue + Unfreeze (Modelo Final)

**Objetivo**: Combinar todas las mejoras más potentes en un solo modelo.

**Modificaciones**:
- **Trimodal**: Audio + Texto + ViT.
- **SpeechCue**: Inyecta prompts de prosodia en el encoder de texto (`text.use_prosody_cues: true`).
- **Unfreeze Completo**: Descongela capas superiores de Audio (6), Texto (4) y Visión (2).
- **Cabeza LSTM**: Mantiene la clasificación secuencial.

**Resultados (Test Session 5)**:
- **Weighted Accuracy**: ~80.3%
- **Sad Accuracy**: ~93.5%
- **Happy Accuracy**: ~84.3%

**Configuración**:
- Combina configuraciones de Ablation 10 (SpeechCue) y Ablation 11 (ViT).
- `vision_unfreeze_last_n_layers: 2`

---

## 🚀 Uso

### Ejecutar Estudios Nuevos

```bash
# Estudio 11: Trimodal ViT Base
python scripts/run_ablation.py --study 11

# Estudio 13: Modelo Final Optimizado
python scripts/run_ablation.py --study 13
```
