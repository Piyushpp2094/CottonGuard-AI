<#
.SYNOPSIS
    Starts the Cotton Leaf AI Backend (Ubuntu WSL 2 CUDA) and React/Vite Frontend cleanly.
    Keeps Docker/CVAT services completely separate and leaves Windows Smart App Control intact.
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogsDir = Join-Path $ProjectRoot "artifacts\logs"
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

# Convert D:\Main_Project\cotton_leaf_ai -> /mnt/d/Main_Project/cotton_leaf_ai dynamically
$Drive = $ProjectRoot.Substring(0, 1).ToLower()
$RelRest = $ProjectRoot.Substring(2).Replace("\", "/")
$WslProjectPath = "/mnt/$Drive$RelRest"

Write-Host "`n==========================================================================" -ForegroundColor Green
Write-Host "STARTING COTTON LEAF AI PLATFORM (WSL 2 GPU BACKEND + VITE FRONTEND)" -ForegroundColor Green
Write-Host "==========================================================================" -ForegroundColor Green

# 1. Stop any stale app servers first without touching CVAT
$StopScript = Join-Path $ProjectRoot "stop_project.ps1"
if (Test-Path $StopScript) {
    & $StopScript -Quiet
}

# 2. Ensure Unix LF line endings and launch FastAPI Backend in hidden WSL process
Write-Host "[1/2] Launching FastAPI AI Backend inside Ubuntu WSL 2 (Port 8000)..."
wsl -d Ubuntu -- sed -i 's/\r$//' "$WslProjectPath/scripts/run_backend_wsl.sh" "$WslProjectPath/scripts/run_frontend_wsl.sh"
Start-Process -FilePath "wsl.exe" -ArgumentList "-d", "Ubuntu", "--", "bash", "$WslProjectPath/scripts/run_backend_wsl.sh" -WindowStyle Hidden

# 3. Launch React + Vite Frontend in hidden WSL process
Write-Host "[2/2] Launching React + Vite Frontend (Port 5173)..."
Start-Process -FilePath "wsl.exe" -ArgumentList "-d", "Ubuntu", "--", "bash", "$WslProjectPath/scripts/run_frontend_wsl.sh" -WindowStyle Hidden

# 4. Wait for readiness
$BackendReady = $false
$FrontendReady = $false
$MaxWaitSeconds = 45

Write-Host "Waiting for backend and frontend servers to initialize..."
for ($i = 0; $i -lt $MaxWaitSeconds; $i++) {
    if (-not $BackendReady) {
        try {
            $h = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 2 -ErrorAction Stop
            if ($h.status) { $BackendReady = $true }
        } catch {}
    }
    if (-not $FrontendReady) {
        try {
            $f = Invoke-WebRequest -Uri "http://127.0.0.1:5173" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($f.StatusCode -eq 200) { $FrontendReady = $true }
        } catch {}
    }
    if ($BackendReady -and $FrontendReady) { break }
    Start-Sleep -Seconds 1
}

Write-Host "`n--------------------------------------------------------------------------" -ForegroundColor Cyan
if ($BackendReady) {
    Write-Host "  Backend API Ready:   http://127.0.0.1:8000" -ForegroundColor Green
    Write-Host "  Swagger OpenAPI UI:  http://127.0.0.1:8000/docs" -ForegroundColor Green
} else {
    Write-Host "  Backend API NOT ready within ${MaxWaitSeconds}s. Check artifacts\logs\backend.log" -ForegroundColor Red
}

if ($FrontendReady) {
    Write-Host "  React + Vite UI:     http://localhost:5173" -ForegroundColor Green
} else {
    Write-Host "  Frontend NOT ready within ${MaxWaitSeconds}s. Check artifacts\logs\frontend.log" -ForegroundColor Red
}
Write-Host "  CVAT (Separate):     http://localhost:8080 (managed independently)" -ForegroundColor DarkGray
Write-Host "  Stop Services With:  .\stop_project.ps1" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------------------`n" -ForegroundColor Cyan

if (-not ($BackendReady -and $FrontendReady)) {
    exit 1
}
