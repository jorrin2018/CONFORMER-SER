# Estrategias para Superar el Estancamiento en 76-77% Accuracy

> [!NOTE]
> **Documento historico.** Escrito a partir de los primeros **39 experimentos**. Las estrategias
> siguen siendo validas como analisis, pero las cifras y el diagnostico corresponden a ese momento.
> Estado actual: [RANKING_Y_RESULTADOS.md](RANKING_Y_RESULTADOS.md).

**Fecha**: Diciembre 2024 (actualizado Marzo 2025)  
**Problema Identificado**: Todos los experimentos se estancan alrededor del 76-78% UA en test, con gaps significativos entre validación y test (overfitting).  
**Nota**: Ablation 34 (mejor ejecución) alcanza 77.98% UA con gap de solo 2.68%.

---

## 📊 Análisis del Problema

### Estado Actual de los Mejores Modelos

| Modelo | Test UA | Val UA | Gap | F1 Macro | Problema Principal |
|--------|---------|--------|-----|----------|-------------------|
| **Ablation 25** | 77.44% | 82.30% | 4.9% | 77.09% | Mejor generalización, pero aún con gap |
| **Ablation 34** | 76.33-77.98% | 80-83% | 2.7-7.1% | 75-78% | Mejor ejecución: 77.98% UA, gap 2.68% |
| **Ablation 30** | 75.24% | 85.02% | 9.8% | 75.06% | **Overfitting severo** |

### Problemas Identificados

1. **Overfitting Crítico**
   - Gaps val-test de 4.9% a 9.8%
   - Validación alcanza 82-85% pero test solo 75-77%
   - Pérdida de validación no correlaciona bien con test

2. **Clase Neutral es el Cuello de Botella**
   - F1 Score para Neutral: 56-63% (muy bajo)
   - Neutral es la clase más difícil y limita el accuracy global
   - Confusión frecuente entre Neutral y otras emociones

3. **Configuraciones Subóptimas**
   - Batch size muy pequeño (2) → gradientes ruidosos
   - Learning rates muy conservadoras
   - Data augmentation limitada en algunos experimentos
   - Dropout 0.4 pero aún hay overfitting

4. **Limitaciones del Dataset**
   - IEMOCAP tiene ~5,000 muestras (relativamente pequeño)
   - División speaker-independent es estricta
   - Desbalance de clases (especialmente Neutral)

---

## 🎯 Estrategias Prioritarias para Mejorar Accuracy

### **Estrategia 1: Ensemble de Modelos Top** ⭐⭐⭐ (Alta Prioridad)

**Objetivo**: Combinar las fortalezas de Ablation 25, 34 y 30 para superar el techo individual.

**Implementación**:
```python
# Ensemble ponderado por confianza
predictions = (
    0.4 * ablation_25_pred +  # Mejor generalización
    0.35 * ablation_34_pred +  # Mejor balance
    0.25 * ablation_30_pred   # Mejor en validación (con regularización)
)
```

**Ventajas**:
- Reduce variabilidad entre modelos
- Compensa debilidades individuales
- **Ganancia esperada: +2-4% accuracy**

**Pasos**:
1. Cargar checkpoints de los 3 mejores modelos
2. Implementar ensemble por votación ponderada
3. Ajustar pesos basados en confianza por clase
4. Evaluar en test set

---

### **Estrategia 2: Regularización Avanzada** ⭐⭐⭐ (Alta Prioridad)

**Problema**: Overfitting es el principal limitante. Necesitamos regularización más efectiva.

#### 2.1 Dropout Adaptativo por Capa
```yaml
model:
  dropout: 0.5  # Aumentar de 0.4
  dropout_audio: 0.6  # Más agresivo en audio (más parámetros)
  dropout_text: 0.4   # Menos en texto (menos parámetros)
  dropout_fusion: 0.5  # Intermedio en fusión
```

#### 2.2 Weight Decay Diferenciado
```yaml
training:
  weight_decay_audio: 0.1   # Más regularización para audio
  weight_decay_text: 0.05   # Menos para texto (frozen)
  weight_decay_fusion: 0.08 # Intermedio para fusión
```

#### 2.3 Early Stopping Más Agresivo
```yaml
training:
  early_stopping_patience: 8  # Reducir de 15 a 8
  early_stopping_min_delta: 0.001  # Requerir mejora mínima
  monitor: "val_loss"  # Monitorear pérdida, no accuracy
```

#### 2.4 Label Smoothing Aumentado
```yaml
training:
  loss:
    label_smoothing: 0.1  # Aumentar de 0.05 a 0.1
```

**Ganancia esperada: +1-2% accuracy** (reduciendo overfitting)

---

### **Estrategia 3: Data Augmentation Mejorada** ⭐⭐ (Media Prioridad)

**Problema**: Dataset pequeño (~5K muestras) requiere augmentation más agresiva.

#### 3.1 Audio Augmentation Avanzada
```yaml
audio:
  augmentation:
    apply_time_masking: true
    apply_noise: true
    apply_speed_perturb: true
    apply_pitch_shift: true  # HABILITAR
    apply_time_stretch: true  # NUEVO
    apply_reverb: true  # NUEVO
    
    # Parámetros más agresivos
    time_mask_prob: 0.6  # Aumentar de 0.5
    num_time_masks: 2     # Aumentar de 1
    noise_prob: 0.4       # Aumentar de 0.3
    speed_prob: 0.4       # Aumentar de 0.3
    speed_range: [0.85, 1.15]  # Más amplio
    pitch_prob: 0.3       # HABILITAR
    pitch_range: [-3, 3]  # Más amplio
```

#### 3.2 Text Augmentation Habilitada
```yaml
text:
  apply_text_augment: true  # HABILITAR (estaba false en algunos)
  augmentation:
    apply_random_mask: true
    apply_random_insert: true
    apply_synonym_replacement: true  # NUEVO
    apply_back_translation: false     # Opcional (lento)
    mask_prob: 0.2
    insert_prob: 0.15
```

#### 3.3 Visual Augmentation (Mel-Spectrograms)
```yaml
vision:
  augmentation:
    apply_spec_augment: true  # NUEVO
    freq_mask_prob: 0.5
    time_mask_prob: 0.5
    num_freq_masks: 2
    num_time_masks: 2
```

**Ganancia esperada: +1-2% accuracy**

---

### **Estrategia 4: Fine-tuning Específico para Neutral** ⭐⭐ (Media Prioridad)

**Problema**: Neutral tiene F1 < 65%, limitando el accuracy global.

#### 4.1 Loss Ponderado por Clase
```yaml
training:
  loss:
    type: "focal"  # Cambiar de cross_entropy
    class_weights: [1.0, 1.2, 1.0, 1.5]  # [angry, happy, sad, neutral]
    # Neutral tiene peso 1.5x para enfocarse más
    gamma: 2.5  # Aumentar de 2.0 para enfocarse en hard examples
```

#### 4.2 Oversampling de Neutral
```python
# En dataset.py, implementar oversampling
# Aumentar muestras de neutral en training set
```

#### 4.3 Análisis de Confusión Neutral
- Identificar qué clases se confunden más con Neutral
- Ajustar threshold de decisión para Neutral
- Entrenar clasificador binario Neutral vs Resto

**Ganancia esperada: +2-3% accuracy** (mejorando Neutral de 63% a ~75% F1)

---

### **Estrategia 5: Mejoras Arquitectónicas Selectivas** ⭐ (Baja Prioridad)

#### 5.1 Batch Normalization en Fusion Layers
```python
# Agregar BatchNorm después de cada capa de atención
# Ayuda con estabilidad y generalización
```

#### 5.2 Layer Normalization Adicional
```python
# Agregar LayerNorm antes y después de fusion
# Mejor estabilidad del entrenamiento
```

#### 5.3 Attention Dropout
```python
# Agregar dropout específico en attention weights
attention_dropout: 0.2
```

**Ganancia esperada: +0.5-1% accuracy**

---

### **Estrategia 6: Optimización de Hiperparámetros** ⭐⭐ (Media Prioridad)

#### 6.1 Learning Rate Schedule Mejorado
```yaml
training:
  scheduler_type: "cosine_with_restarts"  # En lugar de plateau
  # O usar OneCycleLR para mejor convergencia
  learning_rate_audio: 5.0e-7   # Aumentar ligeramente
  learning_rate_text: 2.0e-6    # Aumentar ligeramente
  learning_rate_fusion: 8.0e-6  # Aumentar ligeramente
```

#### 6.2 Batch Size Aumentado (si GPU lo permite)
```yaml
training:
  batch_size: 4  # Aumentar de 2 a 4 si hay memoria
  gradient_accumulation_steps: 4  # Reducir de 8 a 4
  # Batch efectivo: 16 (igual que antes)
```

#### 6.3 Warmup Más Largo
```yaml
training:
  warmup_ratio: 0.15  # Aumentar de 0.1
  # Más tiempo para estabilizar antes de entrenamiento agresivo
```

**Ganancia esperada: +0.5-1% accuracy**

---

### **Estrategia 7: Pseudo-Labeling / Semi-Supervised Learning** ⭐ (Baja Prioridad)

**Idea**: Usar predicciones de alta confianza del modelo como labels adicionales.

**Implementación**:
1. Entrenar modelo inicial
2. Predecir en validation set
3. Agregar muestras con confianza > 0.95 al training set
4. Re-entrenar con dataset expandido

**Ganancia esperada: +1-2% accuracy** (pero requiere validación cuidadosa)

---

### **Estrategia 8: Análisis de Errores y Hard Negative Mining** ⭐⭐ (Media Prioridad)

**Implementación**:
1. Identificar muestras que el modelo clasifica incorrectamente
2. Analizar patrones comunes en errores
3. Aplicar hard negative mining: re-entrenar enfocándose en errores
4. Ajustar thresholds por clase basados en análisis

**Ganancia esperada: +1-2% accuracy**

---

## 🎯 Plan de Acción Recomendado (Priorizado)

### Fase 1: Quick Wins (1-2 días)
1. ✅ **Ensemble de Modelos** (Estrategia 1)
   - Implementar ensemble de Ablation 25, 34, 30
   - **Ganancia esperada: +2-4%**

2. ✅ **Regularización Mejorada** (Estrategia 2)
   - Aumentar dropout a 0.5
   - Aumentar label smoothing a 0.1
   - Early stopping más agresivo
   - **Ganancia esperada: +1-2%**

### Fase 2: Mejoras Sustanciales (3-5 días)
3. ✅ **Data Augmentation Avanzada** (Estrategia 3)
   - Habilitar todas las augmentations
   - Parámetros más agresivos
   - **Ganancia esperada: +1-2%**

4. ✅ **Fine-tuning para Neutral** (Estrategia 4)
   - Focal loss con pesos para neutral
   - Oversampling de neutral
   - **Ganancia esperada: +2-3%**

### Fase 3: Optimizaciones Finales (2-3 días)
5. ✅ **Optimización de Hiperparámetros** (Estrategia 6)
   - Ajustar learning rates
   - Mejorar scheduler
   - **Ganancia esperada: +0.5-1%**

6. ✅ **Análisis de Errores** (Estrategia 8)
   - Hard negative mining
   - Ajuste de thresholds
   - **Ganancia esperada: +1-2%**

---

## 📈 Proyección de Resultados

### Escenario Conservador
- Ensemble: +2%
- Regularización: +1%
- Augmentation: +1%
- **Total: +4% → 80-81% UA**

### Escenario Optimista
- Ensemble: +3%
- Regularización: +2%
- Augmentation: +2%
- Neutral fine-tuning: +2%
- **Total: +9% → 85-86% UA**

### Escenario Realista (Más Probable)
- Ensemble: +2.5%
- Regularización: +1.5%
- Augmentation: +1.5%
- Neutral fine-tuning: +2%
- **Total: +7.5% → 83-84% UA**

---

## ⚠️ Consideraciones Importantes

1. **No todas las estrategias son compatibles**
   - Ensemble es independiente y se puede aplicar siempre
   - Regularización y augmentation pueden competir (elegir balance)

2. **Validación Rigurosa**
   - Usar k-fold o validación cruzada para evitar overfitting en test
   - No ajustar hiperparámetros basándose en test set

3. **Límites del Dataset**
   - IEMOCAP tiene ~5K muestras
   - El techo teórico puede estar alrededor del 85-90% UA
   - Mejoras adicionales pueden requerir más datos

4. **Trade-offs**
   - Más regularización → menos overfitting pero puede sub-entrenar
   - Más augmentation → mejor generalización pero entrenamiento más lento
   - Ensemble → mejor accuracy pero inferencia más lenta

---

## 🔬 Experimentos Sugeridos

### Experimento 1: Ensemble Básico
```bash
python src/evaluate_ensemble.py \
  --checkpoints checkpoints/ablation_25/best_model.pt \
                checkpoints/ablation_34/best_model.pt \
                checkpoints/ablation_30/best_model.pt \
  --weights 0.4 0.35 0.25
```

### Experimento 2: Regularización Mejorada (Ablation 36A/37)
```powershell
# Ablation 36A: Conflict-Aware + Dirichlet con regularización mejorada
python scripts/run_ablation.py --study 36

# Ablation 37: Dirichlet + Conflict-Aware (config específica)
python scripts/run_ablation.py --study 37
```

### Experimento 3: Neutral-Focused (Ablation 36B)
```powershell
# Ablation 36B: Conflict-Aware + Focal Loss con peso 2x para neutral
# Config: config/ablation/ablation_36b_focal_conflict.yaml
# Si no está mapeado en run_ablation.py, añadir entrada para study 36B
# o ejecutar con: python scripts/run_ablation.py --study 36 --config config/ablation/ablation_36b_focal_conflict.yaml
```

---

## 📝 Notas Finales

El estancamiento en 76-77% es común en SER con IEMOCAP debido a:
- Tamaño limitado del dataset
- División speaker-independent estricta
- Ambigüedad inherente en emociones (especialmente Neutral)

Las estrategias propuestas deberían llevar el modelo a **83-85% UA**, que está en el rango state-of-the-art para IEMOCAP 4-class speaker-independent.

**Próximo paso recomendado**: Implementar Ensemble (Estrategia 1) como quick win, luego proceder con regularización mejorada.

---

*Documento generado basado en análisis de 39 experimentos de ablación*  
*Última actualización: Marzo 2025*  
*Nota: Ablations 36-39 implementan Conflict-Aware+Dirichlet, OpenSMILE, SOTA*






