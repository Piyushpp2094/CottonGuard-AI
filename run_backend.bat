@echo off
setlocal enabledelayedexpansion
title CottonGuard AI - Backend Server (FastAPI + CUDA)

cd /d "%~dp0"
echo =====================================================================
echo CottonGuard AI - FastAPI Backend Server
echo =====================================================================

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe!
    echo Please ensure the Python 3.11 virtual environment is installed.
    pause
    exit /b 1
)

echo [1/3] Verifying Core Model Checkpoints...
set CHECKPOINT_DIR=artifacts\checkpoints
if not exist "%CHECKPOINT_DIR%\classification\best_model.pt" (
    echo [WARNING] Classification checkpoint not found at %CHECKPOINT_DIR%\classification\best_model.pt
) else (
    echo   [OK] Classification: %CHECKPOINT_DIR%\classification\best_model.pt
)

if not exist "%CHECKPOINT_DIR%\segmentation\best_unet.pt" (
    echo [WARNING] Segmentation checkpoint not found at %CHECKPOINT_DIR%\segmentation\best_unet.pt
) else (
    echo   [OK] Segmentation:   %CHECKPOINT_DIR%\segmentation\best_unet.pt
)

if not exist "%CHECKPOINT_DIR%\detection\best_faster_rcnn.pt" (
    echo [WARNING] Detection checkpoint not found at %CHECKPOINT_DIR%\detection\best_faster_rcnn.pt
) else (
    echo   [OK] Detection:      %CHECKPOINT_DIR%\detection\best_faster_rcnn.pt
)

echo.
echo [2/3] Checking Database...
if exist "cotton_leaf_ai.db" (
    echo   [OK] Database: cotton_leaf_ai.db verified.
) else (
    echo   [WARNING] cotton_leaf_ai.db not found in current directory.
)

echo.
echo [3/3] Launching Uvicorn Server at http://127.0.0.1:8000 ...
echo   API Docs: http://127.0.0.1:8000/docs
echo   Health:   http://127.0.0.1:8000/api/health
echo Press Ctrl+C to stop the server.
echo =====================================================================

".venv\Scripts\python.exe" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 1

pause
