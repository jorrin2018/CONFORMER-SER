# Comandos de Evaluación - Guía Rápida

## 🚀 Inicio Rápido

### Opción Más Fácil: Script Interactivo
```powershell
.\scripts\windows\EVALUAR_RAPIDO.bat
```
Este script te permite elegir qué modelo evaluar desde un menú.

---

## 📋 Evaluar Modelo Individual de Ablación

**⚠️ IMPORTANTE**: `evaluate.py` solo funciona con modelos estándar. Para modelos de ablación, usa `evaluate_ensemble.py` con un solo modelo:

### Ablation 25 (Enhanced Cross-Attention) - Mejor Generalización
```powershell
python src\evaluate_ensemble.py `
    --checkpoints checkpoints\ablation_25\ablation_25_20251125_092101\best_model.pt `
    --configs config\ablation\ablation_25_enhanced_cross_attention.yaml `
    --study_numbers 25 `
    --weights 1.0 `
    --split test `
    --output_dir evaluation_results\ablation_25 `
    --device cuda
```

### Ablation 34 (Conflict-Aware Attention) - Mejor Balance
```powershell
# Mejor ejecución: ablation_34_20251210_191151 (77.98% UA, gap 2.68%)
python src\evaluate_ensemble.py `
    --checkpoints checkpoints\ablation_34\ablation_34_20251210_191151\best_model.pt `
    --configs config\ablation\ablation_34_conflict_aware_attention.yaml `
    --study_numbers 34 `
    --weights 1.0 `
    --split test `
    --output_dir evaluation_results\ablation_34 `
    --device cuda
```

### Ablation 30 (Dirichlet Fusion) - Mejor Validación (pero overfitting)
```powershell
python src\evaluate_ensemble.py `
    --checkpoints checkpoints\ablation_30\ablation_30_20251126_175535\best_model.pt `
    --configs config\ablation\ablation_30_dirichlet.yaml `
    --study_numbers 30 `
    --weights 1.0 `
    --split test `
    --output_dir evaluation_results\ablation_30 `
    --device cuda
```

---

## 🎯 Evaluar Ensemble (Recomendado)

### Ensemble de los 3 Mejores Modelos
```powershell
# Usar mejor ejecución de Ablation 34: 20251210_191151 (77.98% UA)
python src\evaluate_ensemble.py `
    --checkpoints `
        checkpoints\ablation_25\ablation_25_20251125_092101\best_model.pt `
        checkpoints\ablation_34\ablation_34_20251210_191151\best_model.pt `
        checkpoints\ablation_30\ablation_30_20251126_175535\best_model.pt `
    --configs `
        config\ablation\ablation_25_enhanced_cross_attention.yaml `
        config\ablation\ablation_34_conflict_aware_attention.yaml `
        config\ablation\ablation_30_dirichlet.yaml `
    --study_numbers 25 34 30 `
    --weights 0.4 0.35 0.25 `
    --split test `
    --method weighted_average `
    --output_dir evaluation_results\ensemble `
    --device cuda
```

**Nota**: Usar archivos YAML de configuración. Ajusta las rutas de checkpoints según tus ejecuciones (ver `checkpoints/ablation_XX/`).

---

## 🚀 Usar Script Batch (Más Fácil)

### Evaluar Ensemble
```powershell
.\scripts\windows\evaluate_ensemble.bat
```

**Nota**: Necesitas editar `scripts\windows\evaluate_ensemble.bat` y actualizar las rutas de los checkpoints según tus archivos.

---

## 🧪 Test-only sin reentrenar

Si el run ya contiene `best_model.pt` y `config.json`, puedes ejecutar directamente el split de test y regenerar `final_results.json` en el mismo directorio:

```powershell
python scripts\run_test_only.py --checkpoint_dir checkpoints\phase1_wavlm_gated\ablation_34_20260203_094355 --study 34 --device cuda
```

También puedes pasar una ruta directa al checkpoint:

```powershell
python scripts\run_test_only.py --checkpoint_dir checkpoints\ablation_34\run\best_model.pt --study 34 --device cuda
```

Usa este flujo cuando quieras evaluar un modelo ya entrenado sin iniciar otra corrida de entrenamiento.

---

## 📊 Comparar Resultados

Después de ejecutar las evaluaciones, compara los resultados:

1. **Ablation 25 individual**: `evaluation_results\ablation_25\metrics_test.txt`
2. **Ablation 34 individual**: `evaluation_results\ablation_34\metrics_test.txt`
3. **Ablation 30 individual**: `evaluation_results\ablation_30\metrics_test.txt`
4. **Ensemble**: `evaluation_results\ensemble\ensemble_metrics_test.txt`

El ensemble puede mejorar o estabilizar las métricas frente a un modelo individual, pero conviene validarlo con el mismo split y revisar WA, UA, F1 macro y matriz de confusión.

---

## ⚠️ Notas Importantes

1. **Config vs config.json**: 
   - `evaluate_ensemble.py` necesita los archivos YAML de `config/ablation/`
   - O `config.json` del checkpoint (según implementación)

2. **Modelos de Ablación (1-50, 52-65)**: 
   - Usar `evaluate_ensemble.py` con `--study_numbers` para cargar arquitectura correcta
   - Usar `run_test_only.py` si tienes el directorio del run con `config.json`
   - El estudio 51 no está definido en el listado actual de `scripts/run_ablation.py`

3. **GPU Memory**: 
   - Si tienes problemas de memoria, reduce `batch_size` en el config
   - O usa `--device cpu` (será más lento)

---

## 🔧 Solución de Problemas

### Error: "Model architecture mismatch"
- Asegúrate de usar el `config.json` correcto del checkpoint
- O usa los archivos YAML originales con `evaluate_ensemble.py`

### Error: "CUDA out of memory"
- Reduce `batch_size` en la configuración
- O usa `--device cpu`

### Error: "Checkpoint not found"
- Verifica que las rutas sean correctas
- Usa rutas relativas desde el directorio raíz del proyecto

