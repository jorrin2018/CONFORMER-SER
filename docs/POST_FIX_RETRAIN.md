# Reentrenamiento tras correcciones (training + fusion)

Se corrigieron en código:

1. **Acumulación de gradientes** (`gradient_accumulation_steps`) en `src/train.py`.
2. **Scheduler**: pasos totales alineados con updates del optimizador; `cosine_with_warmup` tratado como cosine+warmup; aviso si `warmup_ratio` con `plateau`.
3. **`run_ablation.py`**: ya no se llama `scheduler.step()` dos veces en schedulers por pasos (solo `ReduceLROnPlateau` tras validación).
4. **`ConflictAwareCrossAttentionFusion`**: los pesos conflict-aware ponderan los tokens **antes** del MLP de fusión.
5. **`TriModalGLUFusion`**: GLU registrado con `nn.ModuleDict` (parámetros entrenables y en checkpoint).

Los reentrenamientos los ejecutas tú en la máquina con GPU. Usa un prefijo de experimento si quieres separar carpetas, por ejemplo `--experiment_name post_fix_YYYYMMDD`.

## Fase A (prioridad): conflict-aware y cadena reciente

Estudios: **34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57**

```powershell
cd "D:\DOCTORADO\PROYECTOS\CONFORMER-SER"
foreach ($s in 34,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,52,53,54,55,56,57) {
  python scripts/run_ablation.py --study $s --experiment_name post_fix_conflict_chain
}
```

(12 y 20/29/41 tienen rutas especiales LOSO/K-fold; el 41 ya está en el bucle.)

## Fase B: GLU (28) y baselines de referencia

```powershell
python scripts/run_ablation.py --study 28 --experiment_name post_fix_glu
python scripts/run_ablation.py --study 25 --experiment_name post_fix_baseline
python scripts/run_ablation.py --study 30 --experiment_name post_fix_baseline
```

## Notas

- Checkpoints anteriores **no** son comparables métrica a métrica con los nuevos si el estudio usaba `gradient_accumulation_steps > 1` o `conflict_aware_cross_attention`.
- Modelos guardados con la fusión conflict-aware antigua **no** cargan con interpretación idéntica tras el cambio de `forward` (arquitectura efectiva distinta); conviene reentrenar desde cero.
