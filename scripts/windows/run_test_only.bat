@echo off

REM Ejecutable desde cualquier ubicacion: se situa en la raiz del repositorio.

pushd "%~dp0..\.."

REM Ejecuta solo evaluación en test sobre un checkpoint ya entrenado.

REM Genera final_results.json y confusion_matrix_test.png en el mismo directorio del checkpoint.

REM

REM Uso: scripts\windows\run_test_only.bat [ruta_al_directorio_del_run]

REM Ejemplo: scripts\windows\run_test_only.bat checkpoints\phase1_wavlm_gated\ablation_34_20260203_094355



set CHECKPOINT_DIR=%~1

if "%CHECKPOINT_DIR%"=="" (

  set CHECKPOINT_DIR=checkpoints\phase1_wavlm_gated\ablation_34_20260203_094355

  echo Usando directorio por defecto: %CHECKPOINT_DIR%

)



python scripts\run_test_only.py --checkpoint_dir "%CHECKPOINT_DIR%" --study 34 --device cuda

pause

