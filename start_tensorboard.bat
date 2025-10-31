@echo off
REM Start TensorBoard for the most recent experiment

echo ================================================
echo Starting TensorBoard
echo ================================================
echo.

REM Activate virtual environment
call venv\Scripts\activate

echo Viewing logs in: logs
echo Open http://localhost:6006 in your browser
echo Press CTRL+C to stop
echo.

REM Start TensorBoard
tensorboard --logdir=logs --reload_interval=5

pause

