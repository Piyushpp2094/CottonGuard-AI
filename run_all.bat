@echo off
setlocal enabledelayedexpansion
title CottonGuard AI - Platform Launcher

cd /d "%~dp0"
echo =====================================================================
echo           COTTONGUARD AI - FULL-STACK PLATFORM LAUNCHER
echo =====================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe!
    pause
    exit /b 1
)

echo [1/4] Checking Core Checkpoints...
if exist "artifacts\checkpoints\classification\best_model.pt" (
    echo   [OK] Classification Checkpoint Present
) else (
    echo   [WARN] Classification Checkpoint Missing
)

if exist "artifacts\checkpoints\segmentation\best_unet.pt" (
    echo   [OK] Segmentation Checkpoint Present
) else (
    echo   [WARN] Segmentation Checkpoint Missing
)

if exist "artifacts\checkpoints\detection\best_faster_rcnn.pt" (
    echo   [OK] Detection Checkpoint Present
) else (
    echo   [WARN] Detection Checkpoint Missing
)

echo.
echo [2/4] Starting Backend Server (Port 8000)...
start "CottonGuard AI - Backend" cmd /c "%~dp0run_backend.bat"

echo.
echo [3/4] Starting Frontend Server (Port 5173)...
start "CottonGuard AI - Frontend" cmd /c "%~dp0run_frontend.bat"

echo.
echo [4/4] Services dispatched.
echo   - Backend URL:  http://127.0.0.1:8000
echo   - Swagger Docs: http://127.0.0.1:8000/docs
echo   - Frontend UI:  http://localhost:5173
echo.
echo Waiting 5 seconds before opening browser...
timeout /t 5 /nobreak >nul
start http://localhost:5173

echo.
echo =====================================================================
echo Both servers are running in separate terminal windows.
echo To shut down all services cleanly, run stop_all.bat or close windows.
echo =====================================================================
pause
