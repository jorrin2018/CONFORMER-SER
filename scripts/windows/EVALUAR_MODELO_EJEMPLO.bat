@echo off
REM Ejecutable desde cualquier ubicacion: se situa en la raiz del repositorio.
pushd "%~dp0..\.."
REM Script de ejemplo para evaluar un modelo de ablación individual
REM Modifica las variables según el modelo que quieras evaluar

echo ========================================
echo Evaluar Modelo de Ablación Individual
echo ========================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM ========================================
REM CONFIGURACIÓN - Modifica estas variables
REM ========================================

REM Opción 1: Ablation 25 (Enhanced Cross-Attention)
set CHECKPOINT=checkpoints\ablation_25\ablation_25_20251125_092101\best_model.pt
set CONFIG=config\ablation\ablation_25_enhanced_cross_attention.yaml
set STUDY_NUMBER=25
set OUTPUT_DIR=evaluation_results\ablation_25

REM Opción 2: Ablation 34 (Conflict-Aware Attention)
REM set CHECKPOINT=checkpoints\ablation_34\ablation_34_20251205_155722\best_model.pt
REM set CONFIG=config\ablation\ablation_34_conflict_aware_attention.yaml
REM set STUDY_NUMBER=34
REM set OUTPUT_DIR=evaluation_results\ablation_34

REM Opción 3: Ablation 30 (Dirichlet Fusion)
REM set CHECKPOINT=checkpoints\ablation_30\ablation_30_20251126_175535\best_model.pt
REM set CONFIG=config\ablation\ablation_30_dirichlet.yaml
REM set STUDY_NUMBER=30
REM set OUTPUT_DIR=evaluation_results\ablation_30

REM ========================================
REM Ejecutar evaluación
REM ========================================

echo Evaluando modelo...
echo   Checkpoint: %CHECKPOINT%
echo   Config: %CONFIG%
echo   Study: %STUDY_NUMBER%
echo   Output: %OUTPUT_DIR%
echo.

python src\evaluate_ensemble.py ^
    --checkpoints "%CHECKPOINT%" ^
    --configs "%CONFIG%" ^
    --study_numbers %STUDY_NUMBER% ^
    --weights 1.0 ^
    --split test ^
    --output_dir "%OUTPUT_DIR%" ^
    --device cuda

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Evaluación completada exitosamente!
    echo ========================================
    echo Resultados guardados en: %OUTPUT_DIR%
) else (
    echo.
    echo ========================================
    echo ERROR: La evaluación falló!
    echo ========================================
    echo Verifica:
    echo   1. Las rutas del checkpoint y config son correctas
    echo   2. El archivo YAML existe
    echo   3. El modelo es compatible
)

pause






