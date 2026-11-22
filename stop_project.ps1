<#
.SYNOPSIS
    Stops the Cotton Leaf AI WSL 2 Backend (uvicorn) and Vite Frontend processes cleanly.
    Leaves Docker Desktop and CVAT services completely untouched.
#>
param(
    [switch]$Quiet
)

$ErrorActionPreference = "SilentlyContinue"

if (-not $Quiet) {
    Write-Host "`nStopping Cotton Leaf AI Backend and Frontend services..." -ForegroundColor Yellow
}

$StopScript = @'
for pidfile in /tmp/cotton_leaf_backend.pid /tmp/cotton_leaf_frontend.pid; do
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill -- -"$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
done
pkill -f "uvicorn backend.app.main:app" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
pkill -f "esbuild" 2>/dev/null || true
'@
$B64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes(($StopScript -replace "`r`n", "`n") + "`n"))
wsl -d Ubuntu -- bash -c "echo $B64 | base64 -d | bash" | Out-Null

if (-not $Quiet) {
    Write-Host "Cotton Leaf AI services stopped cleanly (CVAT & Windows .venv untouched).`n" -ForegroundColor Green
}
