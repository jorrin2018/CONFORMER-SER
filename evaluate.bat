@echo off
REM Evaluation script for Windows

echo ================================================
echo Conformer IEMOCAP SER - Evaluation
echo ================================================

if "%1"=="" (
    echo Usage: evaluate.bat checkpoint_path [split]
    echo Example: evaluate.bat checkpoints\best_model.pt test
    pause
    exit /b 1
)

set CHECKPOINT=%1
set SPLIT=%2

if "%SPLIT%"=="" (
    set SPLIT=test
)

echo Evaluating checkpoint: %CHECKPOINT%
echo Split: %SPLIT%

python src\evaluate.py --checkpoint %CHECKPOINT% --split %SPLIT% --save_predictions

if errorlevel 1 (
    echo Evaluation failed!
    pause
    exit /b 1
) else (
    echo Evaluation completed successfully!
    pause
)

