@echo off
title CottonGuard AI - Demonstration & Serving Server
color 0A

echo ===============================================================================
echo                COTTONGUARD AI - DEMONSTRATION & SERVING SERVER
echo       Intelligent Cotton Leaf Disease Detection, Segmentation & Severity
echo ===============================================================================
echo.

set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

:: 1. Check Python virtual environment
if not exist ".venv\Scripts\python.exe" (
    color 0C
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo Please install dependencies first.
    pause
    exit /b 1
)

:: 2. Run Environment Verification
echo [1/3] Running environment and checkpoint verification...
".venv\Scripts\python.exe" "scripts\verify_environment.py"
if %ERRORLEVEL% neq 0 (
    color 0C
    echo [ERROR] Environment verification failed. Check the error messages above.
    pause
    exit /b 1
)

echo.
echo [2/3] Launching web browser to http://localhost:8000/app ...
start http://localhost:8000/app

echo.
echo [3/3] Starting FastAPI Uvicorn Server on port 8000...
echo -------------------------------------------------------------------------------
echo Access Points:
echo   - Interactive Web Application: http://localhost:8000/app
echo   - Swagger API Documentation:   http://localhost:8000/docs
echo   - Health Telemetry Endpoint:   http://localhost:8000/api/health
echo -------------------------------------------------------------------------------
echo Press Ctrl+C in this console window to gracefully stop the server.
echo.

".venv\Scripts\python.exe" -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --log-level info

pause
