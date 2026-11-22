@echo off
setlocal enabledelayedexpansion
title CottonGuard AI - Frontend (React + Vite)

cd /d "%~dp0"
echo =====================================================================
echo CottonGuard AI - React + Vite Frontend Server
echo =====================================================================

where npm >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [INFO] Detected native Windows Node.js/npm.
    cd frontend
    echo Launching Vite Dev Server at http://localhost:5173 ...
    npm run dev -- --host 127.0.0.1 --port 5173
) else (
    echo [INFO] Native Windows Node.js not detected in PATH.
    echo [INFO] Falling back to Ubuntu WSL 2 Node.js runtime...
    where wsl >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        echo Launching Vite Dev Server via WSL 2 at http://localhost:5173 ...
        wsl -d Ubuntu -- bash -c "export PATH=$HOME/.local/node/bin:$PATH && cd /mnt/d/Main_Project/cotton_leaf_ai/frontend && npm run dev -- --host 0.0.0.0 --port 5173"
    ) else (
        echo [ERROR] Neither native npm nor WSL 2 was found.
        echo Please ensure Node.js is installed on Windows or in WSL 2.
        pause
        exit /b 1
    )
)

pause
