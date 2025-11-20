# Estudios de Ablación - Conformer Multimodal SER

Este módulo contiene la implementación de 10 estudios de ablación para evaluar el impacto de diferentes modificaciones arquitectónicas (y de interfaz con el lenguaje) sobre el modelo base de reconocimiento de emociones multimodal.

## 📋 Descripción General

Los estudios de ablación están diseñados para analizar sistemáticamente la contribución de diferentes componentes del modelo. Cada estudio modifica un aspecto específico mientras mantiene el resto de la arquitectura constante, permitiendo una comparación justa.

**Modelo Base**: Checkpoint `20251029_190659` (mejor rendimiento obtenido)

**Dataset**: IEMOCAP con 7 clases de emociones (angry, happy, sad, neutral, frustrated, excited, disgusted)

## 🗂️ Estructura de Archivos

```
src/ablation_studies/
├── __init__.py                    # Exportaciones del módulo
├── base_config.py                 # Carga y ajusta configuración base
├── classification_heads.py         # Nuevas cabezas de clasificación
├── ablation_01_model.py           # Modelo con encoders descongelados
├── ablation_02_model.py           # Modelo con cabeza LSTM
├── ablation_03_model.py           # Modelo con cabeza Transformer
├── ablation_04_model.py           # Modelo con cabeza CNN 1D
├── ablation_05_model.py           # Modelo con encoders descongelados + LSTM head
├── ablation_06_model.py           # Modelo con encoders descongelados + Transformer head
├── ablation_07_model.py           # Modelo con capas medias/altas descongeladas + LSTM head
├── ablation_08_model.py           # Modelo con encoders descongelados + Deep LSTM head
├── ablation_09_model.py           # Modelo con encoders descongelados + GIA fusion + LSTM head
├── ablation_10_model.py           # Igual que Ablation 07 pero con prompts internos de prosodia (SpeechCue)
└── utils.py                       # Utilidades (optimizadores)

config/ablation/
├── ablation_01_unfreeze_encoders.yaml
├── ablation_02_lstm_head.yaml
├── ablation_03_transformer_head.yaml
├── ablation_04_cnn1d_head.yaml
├── ablation_05_unfreeze_lstm.yaml
├── ablation_06_unfreeze_transformer.yaml
├── ablation_07_unfreeze_mid_high_lstm.yaml
├── ablation_08_deep_lstm.yaml
├── ablation_09_gia_lstm.yaml
└── ablation_10_speechcue_internal.yaml
```

## 🔬 Estudios de Ablación

### Ablation 01: Unfreeze Encoders

**Objetivo**: Evaluar el impacto de descongelar capas superiores de los encoders pre-entrenados.

**Modificaciones**:
- Descongela las últimas 4 capas del encoder de audio (Wav2Vec2)
- Descongela las últimas 2 capas del encoder de texto (DistilBERT)
- Ajusta learning rates conservadores para fine-tuning

**Hipótesis**: El fine-tuning de capas superiores puede capturar características más específicas del dominio de emociones.

**Configuración**:
- `freeze_audio_encoder: false`
- `audio_unfreeze_last_n_layers: 4`
- `freeze_text_encoder: false`
- `text_unfreeze_last_n_layers: 2`
- `learning_rate_audio: 5.0e-7`
- `learning_rate_text: 1.0e-6`

---

### Ablation 02: LSTM Classification Head

**Objetivo**: Evaluar si una cabeza de clasificación basada en LSTM mejora el rendimiento sobre la MLP estándar.

**Modificaciones**:
- Reemplaza la cabeza de clasificación MLP con una red LSTM bidireccional
- Mantiene encoders congelados (como en el modelo base)

**Hipótesis**: Las LSTM pueden capturar dependencias temporales en las características fusionadas que una MLP no puede modelar.

**Arquitectura**:
- LSTM bidireccional (2 capas)
- Proyección lineal + LayerNorm + GELU
- Clasificador final

**Configuración**:
- `classification_head_type: "lstm"`
- `lstm_num_layers: 2`
- `lstm_bidirectional: true`

---

### Ablation 03: Transformer Classification Head

**Objetivo**: Evaluar si una cabeza de clasificación basada en Transformer mejora el rendimiento.

**Modificaciones**:
- Reemplaza la cabeza de clasificación MLP con un pequeño Transformer encoder
- Usa atención multi-cabeza y pooling por atención
- Mantiene encoders congelados

**Hipótesis**: La atención puede ayudar a identificar características relevantes para la clasificación de emociones.

**Arquitectura**:
- TransformerEncoderLayer (1 capa, 8 heads)
- MultiheadAttention para pooling
- MLP con conexión residual
- Clasificador final

**Configuración**:
- `classification_head_type: "transformer"`
- `transformer_num_layers: 1`
- `transformer_num_heads: 8`

---

### Ablation 04: CNN 1D Classification Head

**Objetivo**: Evaluar si una cabeza de clasificación basada en CNN 1D mejora el rendimiento.

**Modificaciones**:
- Reemplaza la cabeza de clasificación MLP con capas convolucionales 1D
- Trata las características fusionadas como secuencia temporal
- Mantiene encoders congelados

**Hipótesis**: Las convoluciones pueden extraer patrones locales en las características fusionadas.

**Arquitectura**:
- Capas Conv1d (2 capas)
- BatchNorm1d después de cada convolución
- AdaptiveAvgPool1d
- Clasificador final (2 capas lineales)

**Configuración**:
- `classification_head_type: "cnn1d"`
- `cnn_num_layers: 2`

---

### Ablation 05: Unfreeze Encoders + LSTM Head

**Objetivo**: Evaluar el impacto combinado de descongelar encoders y usar una cabeza LSTM.

**Modificaciones**:
- Descongela las últimas 4 capas del encoder de audio (Wav2Vec2)
- Descongela las últimas 2 capas del encoder de texto (DistilBERT)
- Reemplaza la cabeza de clasificación MLP con una red LSTM bidireccional
- Combina las modificaciones de Ablation 01 y Ablation 02

**Hipótesis**: La combinación de fine-tuning de encoders y procesamiento secuencial con LSTM puede capturar mejor las características específicas del dominio de emociones.

**Arquitectura**:
- Encoders con capas superiores descongeladas (como Ablation 01)
- LSTM bidireccional (2 capas) para clasificación (como Ablation 02)
- Proyección lineal + LayerNorm + GELU
- Clasificador final

**Configuración**:
- `freeze_audio_encoder: false`
- `audio_unfreeze_last_n_layers: 4`
- `freeze_text_encoder: false`
- `text_unfreeze_last_n_layers: 2`
- `classification_head_type: "lstm"`
- `lstm_num_layers: 2`
- `lstm_bidirectional: true`
- `learning_rate_audio: 5.0e-7`
- `learning_rate_text: 1.0e-6`

---

### Ablation 06: Unfreeze Encoders + Transformer Head

**Objetivo**: Evaluar el impacto combinado de descongelar encoders y usar una cabeza Transformer.

**Modificaciones**:
- Descongela las últimas 4 capas del encoder de audio (Wav2Vec2)
- Descongela las últimas 2 capas del encoder de texto (DistilBERT)
- Reemplaza la cabeza de clasificación MLP con un pequeño Transformer encoder
- Combina las modificaciones de Ablation 01 y Ablation 03

**Hipótesis**: La combinación de fine-tuning de encoders y atención multi-cabeza puede identificar mejor las características relevantes para la clasificación de emociones.

**Arquitectura**:
- Encoders con capas superiores descongeladas (como Ablation 01)
- Transformer encoder (1 capa, 8 heads) para clasificación (como Ablation 03)
- MultiheadAttention para pooling
- MLP con conexión residual
- Clasificador final

**Configuración**:
- `freeze_audio_encoder: false`
- `audio_unfreeze_last_n_layers: 4`
- `freeze_text_encoder: false`
- `text_unfreeze_last_n_layers: 2`
- `classification_head_type: "transformer"`
- `transformer_num_layers: 1`
- `transformer_num_heads: 8`
- `learning_rate_audio: 5.0e-7`
- `learning_rate_text: 1.0e-6`

---

### Ablation 07: Unfreeze Middle and High Encoder Layers + LSTM Head

**Objetivo**: Evaluar el impacto de descongelar más capas (medias y altas) de los encoders en combinación con la cabeza LSTM.

**Modificaciones**:
- Descongela las últimas 6 capas del encoder de audio (Wav2Vec2) - capas medias y altas
- Descongela las últimas 4 capas del encoder de texto (DistilBERT) - capas medias y altas
- Reemplaza la cabeza de clasificación MLP con una red LSTM bidireccional
- Similar a Ablation 05, pero con más capas descongeladas

**Hipótesis**: Descongelar más capas (incluyendo medias) puede capturar características más específicas del dominio de emociones, pero requiere learning rates más cuidadosos para evitar overfitting.

**Arquitectura**:
- Encoders con capas medias y altas descongeladas (más agresivo que Ablation 05)
- LSTM bidireccional (2 capas) para clasificación (igual que Ablation 05)
- Proyección lineal + LayerNorm + GELU
- Clasificador final

**Configuración**:
- `freeze_audio_encoder: false`
- `audio_unfreeze_last_n_layers: 6` (vs 4 en Ablation 05)
- `freeze_text_encoder: false`
- `text_unfreeze_last_n_layers: 4` (vs 2 en Ablation 05)
- `classification_head_type: "lstm"`
- `lstm_num_layers: 2`
- `lstm_bidirectional: true`
- `learning_rate_audio: 3.0e-7` (ligeramente más alto que Ablation 05)
- `learning_rate_text: 1.5e-6` (ligeramente más alto que Ablation 05)

---

### Ablation 08: Unfreeze Encoders + Deep LSTM Head

**Objetivo**: Evaluar el impacto de aumentar la profundidad de la cabeza LSTM en combinación con encoders descongelados.

**Modificaciones**:
- Descongela las últimas 4 capas del encoder de audio (Wav2Vec2)
- Descongela las últimas 2 capas del encoder de texto (DistilBERT)
- Reemplaza la cabeza de clasificación MLP con una red LSTM bidireccional profunda (4 capas)
- Similar a Ablation 05, pero con LSTM más profunda

**Hipótesis**: Una LSTM más profunda puede capturar dependencias más complejas y jerárquicas en las características fusionadas, mejorando la capacidad de modelado del modelo.

**Arquitectura**:
- Encoders con capas superiores descongeladas (igual que Ablation 05)
- LSTM bidireccional profunda (4 capas) para clasificación (vs 2 capas en Ablation 05)
- Proyección lineal + LayerNorm + GELU
- Clasificador final

**Configuración**:
- `freeze_audio_encoder: false`
- `audio_unfreeze_last_n_layers: 4`
- `freeze_text_encoder: false`
- `text_unfreeze_last_n_layers: 2`
- `classification_head_type: "lstm"`
- `lstm_num_layers: 4` (vs 2 en Ablation 05)
- `lstm_bidirectional: true`
- `learning_rate_audio: 5.0e-7`
- `learning_rate_text: 1.0e-6`
- `learning_rate_fusion: 5.0e-5`

---

### Ablation 09: Unfreeze Encoders + GIA Fusion + LSTM Head

**Objetivo**: Evaluar el impacto de incorporar el bloque GIA (Gated Interaction Attention) en la capa de fusión, combinándolo con encoders parcialmente descongelados y una cabeza LSTM.

**Modificaciones**:
- Descongela las últimas 4 capas del encoder de audio (Wav2Vec2) y las últimas 2 capas del encoder de texto (DistilBERT), igual que Ablation 05.
- Sustituye la fusión por Cross-Attention por una fusión **GIA**:
  - Cross-attention bidireccional entre audio y texto.
  - Puertas por modalidad que deciden cuánto mezclar la información propia con la del otro modo.
- Mantiene la cabeza de clasificación LSTM bidireccional (2 capas).

**Hipótesis**: La combinación de fine-tuning de encoders, procesamiento secuencial con LSTM y una fusión más expresiva (GIA) permite modelar mejor las interacciones entre modalidades, capturando dependencias multimodales más ricas que la simple cross-attention.

**Arquitectura**:
- Encoders con capas superiores descongeladas (como Ablation 05).
- Bloque de fusión GIA (cross-attention bidireccional + gates).
- LSTM bidireccional (2 capas) + proyección lineal + LayerNorm + GELU.
- Clasificador final.

**Configuración**:
- `freeze_audio_encoder: false`
- `audio_unfreeze_last_n_layers: 4`
- `freeze_text_encoder: false`
- `text_unfreeze_last_n_layers: 2`
- `classification_head_type: "lstm"`
- `lstm_num_layers: 2`
- `lstm_bidirectional: true`
- `fusion_type: "gia"`
- `learning_rate_audio: 5.0e-7`
- `learning_rate_text: 1.0e-6`
- `learning_rate_fusion: 5.0e-5`

---

### Ablation 10: Unfreeze Middle/High Layers + LSTM Head + Internal SpeechCue Prompts

**Objetivo**: Evaluar si inyectar descripciones prosódicas en lenguaje natural (tipo SpeechCueLLM) mejora el rendimiento sobre Ablation 07.

**Modificaciones**:
- Mantiene la arquitectura de Ablation 07:
  - Encoders parcialmente descongelados (últimas 6 capas de audio, últimas 4 de texto).
  - Cabeza de clasificación LSTM bidireccional (2 capas).
- Añade un módulo de prosodia interna que, a partir del waveform, genera descripciones textuales como:
  - `"overall energy is high, variability is medium, speaking rate is fast"`.
- El texto de entrada al encoder de texto se convierte en:
  - `[PROSODY] <descripción prosódica>. [SPEECH] <transcript original>`.

**Hipótesis**: Traducir características de audio (energía, variabilidad, ritmo) a “pistas” lingüísticas ayuda al encoder de texto (DistilBERT) a razonar mejor sobre el estado emocional, actuando como un adaptador tipo SpeechCueLLM pero interno al modelo.

**Arquitectura**:
- Igual que Ablation 07 en términos de encoders, fusión y cabeza LSTM.
- El cambio está en el pipeline de datos:
  - `IEMOCAPDataset` usa `ProsodyCueExtractor` cuando `use_prosody_cues: true` en la sección `text` de la configuración.

**Configuración**:
- Igual que Ablation 07, más:
  - `text.use_prosody_cues: true`

---

## 🚀 Uso

### Ejecutar Todos los Estudios

```bash
python scripts/run_ablation.py
```

### Ejecutar un Estudio Específico

```bash
# Estudio 01: Unfreeze encoders
python scripts/run_ablation.py --study 1

# Estudio 02: LSTM head
python scripts/run_ablation.py --study 2

# Estudio 03: Transformer head
python scripts/run_ablation.py --study 3

# Estudio 04: CNN 1D head
python scripts/run_ablation.py --study 4

# Estudio 05: Unfreeze encoders + LSTM head
python scripts/run_ablation.py --study 5

# Estudio 06: Unfreeze encoders + Transformer head
python scripts/run_ablation.py --study 6

# Estudio 07: Unfreeze middle/high encoder layers + LSTM head
python scripts/run_ablation.py --study 7

# Estudio 08: Unfreeze encoders + Deep LSTM head
python scripts/run_ablation.py --study 8

# Estudio 09: Unfreeze encoders + GIA fusion + LSTM head
python scripts/run_ablation.py --study 9

# Estudio 10: Unfreeze middle/high layers + LSTM head + Internal SpeechCue prompts
python scripts/run_ablation.py --study 10
```

### Opciones Adicionales

```bash
# Especificar dispositivo
python scripts/run_ablation.py --study 1 --device cuda

# Especificar nombre de experimento
python scripts/run_ablation.py --study 1 --experiment_name "my_experiment"
```

## ⚙️ Configuración

### Configuración Base

La configuración base se carga automáticamente desde `checkpoints/20251029_190659/config.json` y se ajusta para:
- Reducir `num_classes` de 10 a 7 (elimina fearful, surprised, other)
- Actualizar lista de emociones

### Configuraciones de Ablación

Cada estudio tiene su propio archivo YAML en `config/ablation/` que:
- Hereda la configuración base
- Sobrescribe parámetros específicos del estudio
- Define directorios de logs y checkpoints únicos

**Ejemplo de estructura de configuración**:

```yaml
# config/ablation/ablation_02_lstm_head.yaml
model:
  classification_head_type: "lstm"
  lstm_num_layers: 2
  lstm_bidirectional: true
  # ... otros parámetros

training:
  # Parámetros de entrenamiento
  batch_size: 2
  learning_rate_fusion: 5.0e-5
  # ...

logging:
  log_dir: "./logs/ablation_02"
  checkpoint_dir: "./checkpoints/ablation_02"
```

## 📊 Resultados

### Estructura de Directorios de Resultados

Cada estudio genera resultados en su directorio correspondiente:

```
checkpoints/ablation_0X/
├── ablation_0X_TIMESTAMP/
│   ├── best_model.pt              # Mejor modelo según val UA
│   ├── training_curves.png        # Gráficas de entrenamiento
│   ├── confusion_matrix_test.png  # Matriz de confusión
│   └── final_results.json         # Métricas finales
└── ...

logs/ablation_0X/
├── ablation_0X_TIMESTAMP.log              # Log de entrenamiento
└── ...
```

### Métricas Guardadas

Cada estudio guarda:
- **Métricas de entrenamiento**: Loss, Unweighted Accuracy, Weighted Accuracy
- **Métricas de validación**: Loss, Unweighted Accuracy, Weighted Accuracy
- **Métricas de test**: Todas las métricas de clasificación
- **Mejor modelo**: Checkpoint del modelo con mejor UA en validación
- **Gráficas**: Curvas de entrenamiento y matriz de confusión

### Comparación de Resultados

Para comparar los resultados de todos los estudios:

1. Revisar `final_results.json` en cada directorio de checkpoint
2. Comparar `best_val_ua` y `test_metrics` entre estudios
3. Analizar las curvas de entrenamiento en `training_curves.png`

## 🔧 Componentes Técnicos

### Clasificación Heads

#### LSTMClassificationHead
```python
from src.ablation_studies import LSTMClassificationHead

head = LSTMClassificationHead(
    hidden_dim=128,      # Dimensión de entrada (fusion output)
    num_classes=7,
    dropout=0.3,
    num_layers=2,
    bidirectional=True
)
```

#### TransformerClassificationHead
```python
from src.ablation_studies import TransformerClassificationHead

head = TransformerClassificationHead(
    hidden_dim=128,
    num_classes=7,
    dropout=0.3,
    num_layers=1,
    num_heads=8
)
```

#### CNN1DClassificationHead
```python
from src.ablation_studies import CNN1DClassificationHead

head = CNN1DClassificationHead(
    hidden_dim=128,
    num_classes=7,
    dropout=0.3,
    num_layers=2
)
```

### Modelos de Ablación

```python
from src.ablation_studies import (
    Ablation01Model,
    Ablation02Model,
    Ablation03Model,
    Ablation04Model,
    Ablation05Model,
    Ablation06Model,
    Ablation07Model,
    Ablation08Model
)

# Cargar configuración
from src.ablation_studies import load_base_config
import yaml

base_config = load_base_config()
with open('config/ablation/ablation_01_unfreeze_encoders.yaml') as f:
    ablation_config = yaml.safe_load(f)
config = {**base_config, **ablation_config}

# Crear modelo
model = Ablation01Model(config)
```

### Optimizador Específico para Ablación

```python
from src.ablation_studies import create_ablation_optimizer

# Para Ablation 01, 05, 06, 07, 08: grupos separados (audio, text, fusion) - encoders descongelados
optimizer = create_ablation_optimizer(model, config, study_number=1)
optimizer = create_ablation_optimizer(model, config, study_number=5)
optimizer = create_ablation_optimizer(model, config, study_number=6)
optimizer = create_ablation_optimizer(model, config, study_number=7)
optimizer = create_ablation_optimizer(model, config, study_number=8)

# Para otros estudios (02, 03, 04): agrupación estándar - encoders congelados
optimizer = create_ablation_optimizer(model, config, study_number=2)
```

## 📝 Notas Importantes

1. **Checkpoint Base Requerido**: El checkpoint `20251029_190659` debe existir en `checkpoints/` para cargar la configuración base.

2. **Dataset**: Los estudios usan 7 clases de emociones (se eliminaron fearful, surprised, other por bajo número de muestras).

3. **Encoders**: Por defecto, los encoders están congelados excepto en Ablation 01, 05, y 06.

4. **Learning Rates**: Se usan learning rates conservadores, especialmente para encoders descongelados.

5. **Mixed Precision**: Habilitado por defecto en las configuraciones (puede deshabilitarse si hay problemas de NaN).

6. **Early Stopping**: Patience de 15 épocas por defecto.

## 🐛 Troubleshooting

### Error: "Base config not found"
- Asegúrate de que existe `checkpoints/20251029_190659/config.json`

### Error: "CUDA out of memory"
- Reduce `batch_size` en la configuración YAML
- Aumenta `gradient_accumulation_steps` para mantener el batch efectivo

### NaN/Inf Gradients
- Deshabilita `mixed_precision: false` en la configuración
- Reduce los learning rates
- Verifica que los datos de entrada no contengan NaN

### Class Collapse
- Reduce `dropout` (intenta 0.1-0.2)
- Reduce `learning_rate_audio` y `learning_rate_fusion`
- Deshabilita `label_smoothing` (ponlo en 0.0)

## 📚 Referencias

- **Modelo Base**: ConformerMultimodalModel con Cross-Attention Fusion
- **Dataset**: IEMOCAP (5 sesiones)
- **Encoders**: Wav2Vec2 (audio), DistilBERT (texto)

## 👥 Contribución

Para agregar nuevos estudios de ablación:

1. Crear nuevo modelo en `ablation_XX_model.py`
2. Crear configuración YAML en `config/ablation/`
3. Agregar entrada en `scripts/run_ablation.py`
4. Actualizar este README

---

**Última actualización**: 2025-11-17

**Estudios disponibles**: 10 (01-10)

