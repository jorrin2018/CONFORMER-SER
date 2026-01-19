# Comandos de Evaluación - Guía Rápida

## 🚀 Inicio Rápido

### Opción Más Fácil: Script Interactivo
```powershell
.\EVALUAR_RAPIDO.bat
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
python src\evaluate_ensemble.py `
    --checkpoints checkpoints\ablation_34\ablation_34_20251205_155722\best_model.pt `
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
python src\evaluate_ensemble.py `
    --checkpoints `
        checkpoints\ablation_25\ablation_25_20251125_092101\best_model.pt `
        checkpoints\ablation_34\ablation_34_20251205_155722\best_model.pt `
        checkpoints\ablation_30\ablation_30_20251126_175535\best_model.pt `
    --configs `
        checkpoints\ablation_25\ablation_25_20251125_092101\config.json `
        checkpoints\ablation_34\ablation_34_20251205_155722\config.json `
        checkpoints\ablation_30\ablation_30_20251126_175535\config.json `
    --study_numbers 25 34 30 `
    --weights 0.4 0.35 0.25 `
    --split test `
    --method weighted_average `
    --output_dir evaluation_results\ensemble `
    --device cuda
```

**Nota**: El script de ensemble requiere los archivos YAML de configuración originales, no los config.json de los checkpoints. Si tienes problemas, usa:

```powershell
python src\evaluate_ensemble.py `
    --checkpoints `
        checkpoints\ablation_25\ablation_25_20251125_092101\best_model.pt `
        checkpoints\ablation_34\ablation_34_20251205_155722\best_model.pt `
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

---

## 🚀 Usar Script Batch (Más Fácil)

### Evaluar Ensemble
```powershell
.\evaluate_ensemble.bat
```

**Nota**: Necesitas editar `evaluate_ensemble.bat` y actualizar las rutas de los checkpoints según tus archivos.

---

## 📊 Comparar Resultados

Después de ejecutar las evaluaciones, compara los resultados:

1. **Ablation 25 individual**: `evaluation_results\ablation_25\metrics_test.txt`
2. **Ablation 34 individual**: `evaluation_results\ablation_34\metrics_test.txt`
3. **Ablation 30 individual**: `evaluation_results\ablation_30\metrics_test.txt`
4. **Ensemble**: `evaluation_results\ensemble\ensemble_metrics_test.txt`

El ensemble debería mostrar una mejora de **+2-4%** sobre el mejor modelo individual.

---

## ⚠️ Notas Importantes

1. **Config vs config.json**: 
   - `evaluate.py` puede usar `config.json` del checkpoint
   - `evaluate_ensemble.py` necesita los archivos YAML originales

2. **Modelos de Ablación**: 
   - `evaluate.py` está diseñado para modelos estándar
   - Para modelos de ablación, usa `evaluate_ensemble.py` o modifica `evaluate.py`

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

