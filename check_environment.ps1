<#
.SYNOPSIS
    Verifies the Cotton Leaf AI local deployment environment (Windows + WSL 2 + GPU + Node.js + Checkpoints).
#>

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "`n==========================================================================" -ForegroundColor Cyan
Write-Host "COTTON LEAF AI - ENVIRONMENT DIAGNOSTIC CHECK" -ForegroundColor Cyan
Write-Host "==========================================================================" -ForegroundColor Cyan

# 1. Windows .venv check (read-only)
$WinVenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
Write-Host "`n[1] Windows Host Environment" -ForegroundColor Yellow
Write-Host "  Project Root:  $ProjectRoot"
Write-Host "  Windows .venv: $(if (Test-Path $WinVenvPython) { 'Present (kept untouched; Smart App Control blocks Windows c10_cuda.dll)' } else { 'Not found' })"

# 2. Checkpoints & Database
Write-Host "`n[2] Model Checkpoints & Database" -ForegroundColor Yellow
$ClsCkpt = Join-Path $ProjectRoot "artifacts\checkpoints\classification\best_model.pt"
$DetCkpt = Join-Path $ProjectRoot "artifacts\checkpoints\detection\best_yolo.pt"
$SegCkpt = Join-Path $ProjectRoot "artifacts\checkpoints\segmentation\best_unet.pt"
$DbPath  = Join-Path $ProjectRoot "cotton_leaf_ai.db"

Write-Host "  EfficientNet-B0 Classifier: $(if (Test-Path $ClsCkpt) { 'READY (best_model.pt)' } else { 'MISSING' })"
Write-Host "  YOLO Lesion Detector:       $(if (Test-Path $DetCkpt) { 'READY (best_yolo.pt)' } else { 'UNAVAILABLE (awaiting CVAT bbox annotations)' })"
Write-Host "  U-Net 3-Class Segmentation: $(if (Test-Path $SegCkpt) { 'READY (best_unet.pt)' } else { 'UNAVAILABLE (awaiting CVAT masks)' })"
Write-Host "  SQLite Database:            $(if (Test-Path $DbPath)  { 'Present (' + (Get-Item $DbPath).Length + ' bytes)' } else { 'Will initialize on backend startup' })"

# 3. WSL 2 Ubuntu GPU & PyTorch Verification
Write-Host "`n[3] Ubuntu WSL 2 GPU & PyTorch Runtime" -ForegroundColor Yellow
$BashScript = @'
echo "  WSL Distro User: $USER"
grep PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d '"' | sed 's/^/  Ubuntu OS:       /'
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/  WSL GPU:         /'
else
  echo "  WSL GPU:         nvidia-smi unavailable"
fi
VENV_PY="$HOME/.venvs/cotton_leaf_ai/bin/python"
if [ -x "$VENV_PY" ]; then
  PY_VER=$("$VENV_PY" --version)
  echo "  WSL Venv Python: $PY_VER ($VENV_PY)"
  "$VENV_PY" -c "import torch; print('  PyTorch CUDA:', torch.__version__, '| Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); x=torch.ones(64,64,device='cuda:0' if torch.cuda.is_available() else 'cpu'); print('  Tensor Check:    OK on', x.device)"
else
  echo "  WSL Venv Python: MISSING ($VENV_PY)"
fi
if [ -x "$HOME/.local/node/bin/node" ]; then
  NODE_VER=$("$HOME/.local/node/bin/node" --version)
  echo "  WSL Node.js:     $NODE_VER"
else
  echo "  WSL Node.js:     MISSING (~/.local/node/bin/node)"
fi
'@
$BashScript = ($BashScript -replace "`r`n", "`n") + "`n"
$B64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($BashScript))
$WslCheck = wsl -d Ubuntu -- bash -c "echo $B64 | base64 -d | bash"
Write-Host ($WslCheck -join "`n")

# 4. Frontend Dependencies & CVAT Server Check
Write-Host "`n[4] Frontend Bundle & Separate CVAT Server" -ForegroundColor Yellow
$NodeModules = Join-Path $ProjectRoot "frontend\node_modules"
Write-Host "  Frontend node_modules: $(if (Test-Path $NodeModules) { 'Installed' } else { 'MISSING - run npm install in frontend/' })"

try {
    $cvatResp = Invoke-WebRequest -Uri "http://localhost:8080/api/server/about" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "  CVAT (localhost:8080): ONLINE (HTTP $($cvatResp.StatusCode) - Separate Docker stack)" -ForegroundColor Green
} catch {
    Write-Host "  CVAT (localhost:8080): OFFLINE (Start Docker Desktop & CVAT stack separately when annotating)" -ForegroundColor DarkYellow
}

Write-Host "`n==========================================================================`n" -ForegroundColor Cyan
