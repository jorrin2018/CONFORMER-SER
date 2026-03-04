@echo off
REM Script rápido para evaluar modelos
REM Muestra opciones y ejecuta el comando correcto

echo ========================================
echo Evaluación Rápida de Modelos
echo ========================================
echo.
echo Opciones disponibles:
echo.
echo 1. Evaluar Ensemble (3 mejores modelos) - RECOMENDADO
echo 2. Evaluar Ablation 25 individual
echo 3. Evaluar Ablation 34 individual
echo 4. Evaluar Ablation 30 individual
echo 5. Salir
echo.

set /p opcion="Selecciona una opción (1-5): "

if "%opcion%"=="1" goto ensemble
if "%opcion%"=="2" goto ablation25
if "%opcion%"=="3" goto ablation34
if "%opcion%"=="4" goto ablation30
if "%opcion%"=="5" goto end
goto invalid

:ensemble
echo.
echo Ejecutando Ensemble de los 3 mejores modelos...
echo.
call evaluate_ensemble.bat
goto end

:ablation25
echo.
echo Evaluando Ablation 25 (Enhanced Cross-Attention)...
echo.
python src\evaluate_ensemble.py ^
    --checkpoints checkpoints\ablation_25\ablation_25_20251125_092101\best_model.pt ^
    --configs config\ablation\ablation_25_enhanced_cross_attention.yaml ^
    --study_numbers 25 ^
    --weights 1.0 ^
    --split test ^
    --output_dir evaluation_results\ablation_25 ^
    --device cuda
goto end

:ablation34
echo.
echo Evaluando Ablation 34 (Conflict-Aware Attention)...
echo.
python src\evaluate_ensemble.py ^
    --checkpoints checkpoints\ablation_34\ablation_34_20251205_155722\best_model.pt ^
    --configs config\ablation\ablation_34_conflict_aware_attention.yaml ^
    --study_numbers 34 ^
    --weights 1.0 ^
    --split test ^
    --output_dir evaluation_results\ablation_34 ^
    --device cuda
goto end

:ablation30
echo.
echo Evaluando Ablation 30 (Dirichlet Fusion)...
echo.
python src\evaluate_ensemble.py ^
    --checkpoints checkpoints\ablation_30\ablation_30_20251126_175535\best_model.pt ^
    --configs config\ablation\ablation_30_dirichlet.yaml ^
    --study_numbers 30 ^
    --weights 1.0 ^
    --split test ^
    --output_dir evaluation_results\ablation_30 ^
    --device cuda
goto end

:invalid
echo.
echo Opción inválida. Por favor selecciona 1-5.
echo.
pause
goto end

:end
echo.
pause






