@echo off
REM Script to evaluate ensemble of top ablation models
REM Usage: evaluate_ensemble.bat

echo ========================================
echo Ensemble Evaluation - Top Models
echo ========================================
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Find the best checkpoints for each ablation study
REM Ablation 25: Enhanced Cross-Attention (best test UA: 77.44%%)
REM Ablation 34: Conflict-Aware Attention (best test UA: 76.33%%)
REM Ablation 30: Dirichlet Fusion (best val UA: 85.02%%, but overfitting)

REM Checkpoint paths (best models based on analysis)
set CHECKPOINT_25=checkpoints\ablation_25\ablation_25_20251125_092101\best_model.pt
set CHECKPOINT_34=checkpoints\ablation_34\ablation_34_20251205_155722\best_model.pt
set CHECKPOINT_30=checkpoints\ablation_30\ablation_30_20251126_175535\best_model.pt

REM Config files (YAML originals)
set CONFIG_25=config\ablation\ablation_25_enhanced_cross_attention.yaml
set CONFIG_34=config\ablation\ablation_34_conflict_aware_attention.yaml
set CONFIG_30=config\ablation\ablation_30_dirichlet.yaml

echo Running ensemble evaluation...
echo.
echo Models:
echo   - Ablation 25 (Enhanced Cross-Attention)
echo   - Ablation 34 (Conflict-Aware Attention)
echo   - Ablation 30 (Dirichlet Fusion)
echo.

python src\evaluate_ensemble.py ^
    --checkpoints "%CHECKPOINT_25%" "%CHECKPOINT_34%" "%CHECKPOINT_30%" ^
    --configs "%CONFIG_25%" "%CONFIG_34%" "%CONFIG_30%" ^
    --study_numbers 25 34 30 ^
    --weights 0.4 0.35 0.25 ^
    --split test ^
    --method weighted_average ^
    --output_dir evaluation_results\ensemble ^
    --device cuda

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Ensemble evaluation completed successfully!
    echo ========================================
    echo Results saved to: evaluation_results\ensemble
) else (
    echo.
    echo ========================================
    echo ERROR: Ensemble evaluation failed!
    echo ========================================
    echo Please check:
    echo   1. Checkpoint paths are correct
    echo   2. Config files exist
    echo   3. Models are compatible
)

pause

