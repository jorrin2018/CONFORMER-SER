@echo off
REM Setup script for Windows
REM Prerequisites: Python 3.8+, optional venv (recommended).
REM CUDA: prefer installing matching torch/torchaudio from pytorch.org before pip install -r requirements.txt

echo ================================================
echo Conformer IEMOCAP SER - Setup
echo ================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo Installation failed!
    pause
    exit /b 1
)

echo.
echo Running setup verification...
python setup_check.py

echo.
echo ================================================
echo Setup completed!
echo ================================================
echo.
echo Next steps:
echo 1. Update dataset path in config\config.yaml
echo 2. Run: train.bat
echo 3. Monitor with: tensorboard --logdir=logs
echo.
pause

