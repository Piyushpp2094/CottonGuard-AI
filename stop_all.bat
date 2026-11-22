@echo off
title CottonGuard AI - Stop Services

cd /d "%~dp0"
echo =====================================================================
echo Stopping CottonGuard AI Services...
echo =====================================================================

powershell -ExecutionPolicy Bypass -File "%~dp0stop_project.ps1"

taskkill /FI "WINDOWTITLE eq CottonGuard AI - Backend*" /T /F >nul 2>nul
taskkill /FI "WINDOWTITLE eq CottonGuard AI - Frontend*" /T /F >nul 2>nul

echo Done.
timeout /t 2 /nobreak >nul
