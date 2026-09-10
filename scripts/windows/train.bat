@echo off
REM Ejecutable desde cualquier ubicacion: se situa en la raiz del repositorio.
pushd "%~dp0..\.."
REM Training script for Windows

echo ================================================
echo Conformer IEMOCAP SER - Training
echo ================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found in PATH
    pause
    exit /b 1
)

REM Run training
echo Starting training...
python src\train.py %*

if errorlevel 1 (
    echo Training failed!
    pause
    exit /b 1
) else (
    echo Training completed successfully!
    pause
)

