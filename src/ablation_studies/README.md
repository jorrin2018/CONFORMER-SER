# Estudios de Ablación - Conformer Multimodal SER

Este módulo contiene la implementación de 4 estudios de ablación para evaluar el impacto de diferentes modificaciones arquitectónicas sobre el modelo base de reconocimiento de emociones multimodal.

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
└── utils.py                       # Utilidades (optimizadores)

config/ablation/
├── ablation_01_unfreeze_encoders.yaml
├── ablation_02_lstm_head.yaml
├── ablation_03_transformer_head.yaml
└── ablation_04_cnn1d_head.yaml
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
    Ablation04Model
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

# Para Ablation 01: grupos separados (audio, text, fusion)
optimizer = create_ablation_optimizer(model, config, study_number=1)

# Para otros estudios: agrupación estándar
optimizer = create_ablation_optimizer(model, config, study_number=2)
```

## 📝 Notas Importantes

1. **Checkpoint Base Requerido**: El checkpoint `20251029_190659` debe existir en `checkpoints/` para cargar la configuración base.

2. **Dataset**: Los estudios usan 7 clases de emociones (se eliminaron fearful, surprised, other por bajo número de muestras).

3. **Encoders**: Por defecto, los encoders están congelados excepto en Ablation 01.

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

**Última actualización**: 2025-01-29

