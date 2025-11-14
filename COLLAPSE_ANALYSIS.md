# Análisis del Colapso del Modelo (Class Collapse)

## Problema Identificado

El modelo estaba colapsando a una sola clase ("other"), resultando en VAL UA inamovible alrededor de 0.1 (1/10 clases = 10%). Esto ocurre cuando el modelo predice la misma clase para todas las muestras, lo que hace que la Unweighted Accuracy (UA) se quede estancada.

### Síntomas Observados

- **VAL UA**: 0.0999 (≈0.1) - sin cambios entre épocas
- **Distribución de predicciones**: 99.9% de las muestras predichas como "other"
- **Todas las demás clases**: 0% de accuracy
- **VAL UA inamovible**: No mejora independientemente de las épocas

### Checkpoint Problemático

- **Checkpoint**: `20251104_135102`
- **Configuración problemática**:
  - `dropout: 0.4` ❌ (MUY ALTO)
  - `label_smoothing: 0.1` ❌ (con dropout alto causa colapso)
  - `learning_rate_audio: 3e-05` ❌ (demasiado alto)
  - `weight_decay: 0.05` ❌ (demasiada regularización)
  - `scheduler_type: "plateau"` ❌ (sin warmup)
  - `max_grad_norm: 1.0` ❌ (demasiado restrictivo)

### Checkpoint Funcional (Referencia)

- **Checkpoint**: `20251103_164731`
- **Configuración funcional**:
  - `dropout: 0.1` ✅
  - `label_smoothing: 0.0` ✅
  - `learning_rate_audio: 1e-06` ✅ (conservador)
  - `weight_decay: 0.01` ✅ (ligero)
  - `scheduler_type: "cosine"` ✅ (con warmup)
  - `warmup_ratio: 0.1` ✅
  - `max_grad_norm: 0.5` ✅
  - `audio_unfreeze_last_n_layers: 8` ✅ (más capas)

## Causas del Colapso

### 1. **Dropout Demasiado Alto (0.4)**
   - **Problema**: Elimina demasiada información durante el entrenamiento
   - **Efecto**: El modelo no puede aprender patrones distintivos entre clases
   - **Solución**: Reducir a 0.1 o menos

### 2. **Learning Rate Demasiado Alto para el Encoder (3e-05 vs 1e-06)**
   - **Problema**: Actualiza el encoder de forma muy agresiva
   - **Efecto**: El modelo se desestabiliza y colapsa a una clase dominante
   - **Solución**: Usar learning rate conservador (1e-06)

### 3. **Combinación de Label Smoothing + Dropout Alto**
   - **Problema**: Label smoothing (0.1) + dropout alto (0.4) causa confusión
   - **Efecto**: El modelo no puede distinguir entre clases
   - **Solución**: Desactivar label smoothing o usar valores muy bajos

### 4. **Weight Decay Alto (0.05)**
   - **Problema**: Demasiada regularización
   - **Efecto**: El modelo no puede aprender características distintivas
   - **Solución**: Reducir a 0.01 o menos

### 5. **Falta de Warmup**
   - **Problema**: Scheduler "plateau" sin warmup puede causar colapso temprano
   - **Efecto**: El modelo colapsa en las primeras épocas
   - **Solución**: Usar scheduler "cosine" con warmup_ratio: 0.1

### 6. **Gradient Clipping Muy Restrictivo (1.0)**
   - **Problema**: Limita demasiado los gradientes
   - **Efecto**: El modelo no puede aprender adecuadamente
   - **Solución**: Reducir a 0.5

## Solución Implementada

### Cambios en Configuración (`config/config_audio_only.yaml`)

```yaml
model:
  dropout: 0.1  # Reducido de 0.4 (previene colapso)
  audio_unfreeze_last_n_layers: 8  # Aumentado de 6 (mejor fine-tuning)

training:
  learning_rate_audio: 1e-06  # Reducido de 3e-05 (conservador)
  learning_rate_classifier: 0.001  # Mantenido
  weight_decay: 0.01  # Reducido de 0.05 (menos regularización)
  max_grad_norm: 0.5  # Reducido de 1.0 (menos restrictivo)
  
  loss:
    type: "cross_entropy"  # Cambiado de "focal" (más estable)
    label_smoothing: 0.0  # Desactivado (previene colapso)
  
  scheduler_type: "cosine"  # Cambiado de "plateau" (con warmup)
  warmup_ratio: 0.1  # Agregado (previene colapso temprano)

audio:
  max_length: 8  # Reducido de 10 (más estable)
  apply_waveform_augment: false  # Desactivado inicialmente (puede habilitarse después)
```

### Validaciones Agregadas (`src/train.py`)

1. **Función `detect_class_collapse()`**:
   - Detecta si el modelo predice una sola clase para >80% de las muestras
   - Muestra distribución de clases y warnings claros

2. **Validación en `validate()`**:
   - Verifica colapso después de cada validación
   - Muestra mensajes de error claros con recomendaciones
   - Reporta diversidad de clases predichas

3. **Mejores logs en entrenamiento**:
   - Detecta colapso temprano en el primer batch
   - Muestra warnings claros con recomendaciones

## Estrategia de Prevención

### 1. **Valores Seguros Recomendados**

```yaml
# VALORES SEGUROS (previenen colapso)
dropout: 0.1  # Máximo 0.2
learning_rate_audio: 1e-06  # Máximo 5e-06
learning_rate_classifier: 0.001  # Estándar
weight_decay: 0.01  # Máximo 0.02
label_smoothing: 0.0  # Máximo 0.05
max_grad_norm: 0.5  # Máximo 1.0
scheduler_type: "cosine"  # Con warmup
warmup_ratio: 0.1  # 10% de pasos
```

### 2. **Monitoreo Durante el Entrenamiento**

- ✅ Revisar logs de validación para mensajes de "CLASS COLLAPSE DETECTED"
- ✅ Verificar que VAL UA no esté estancado en ~0.1 (1/num_classes)
- ✅ Verificar diversidad de clases predichas (debe ser >3 clases)
- ✅ Revisar distribución de clases en cada validación

### 3. **Señales de Alerta**

- ⚠️ VAL UA estancado en ~0.1 (para 10 clases)
- ⚠️ Una sola clase predicha en >80% de las muestras
- ⚠️ Todas las demás clases con 0% accuracy
- ⚠️ Logits con muy baja variabilidad (std < 0.1)

### 4. **Si Ocurre Colapso**

1. **Detener el entrenamiento inmediatamente**
2. **Revisar configuración**:
   - Reducir dropout a ≤0.1
   - Reducir learning_rate_audio a ≤1e-06
   - Desactivar label_smoothing (0.0)
   - Reducir weight_decay a ≤0.01
   - Usar scheduler "cosine" con warmup
3. **Reiniciar desde el inicio** (no desde checkpoint problemático)
4. **Verificar logs** para confirmar que el problema se resolvió

## Comparación de Métricas

### Checkpoint Problemático (20251104_135102)
- VAL UA: 0.0999 (≈0.1) ❌
- Clases predichas: 1/10 (only "other")
- Distribución: 99.9% "other", 0% demás clases

### Checkpoint Funcional (20251103_164731)
- VAL UA: 0.2778 (27.78%) ✅
- Clases predichas: Múltiples clases
- Distribución: Balanceada entre clases

## Conclusión

El colapso del modelo se debe principalmente a:
1. **Dropout demasiado alto** (factor más crítico)
2. **Learning rate demasiado alto** para el encoder
3. **Combinación de regularizaciones agresivas** (dropout + weight_decay + label_smoothing)

La solución implementada:
- ✅ Corrige valores problemáticos
- ✅ Agrega detección automática de colapso
- ✅ Proporciona mensajes claros de diagnóstico
- ✅ Establece valores seguros recomendados

**Última actualización**: 4 de Noviembre, 2025

