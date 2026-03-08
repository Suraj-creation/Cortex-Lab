# Cortex Lab - Start Both Servers (Windows)
$ErrorActionPreference = "Continue"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Definition

Write-Host ""
Write-Host "  Cortex Lab - Fine-Tuned DeepSeek-R1-7B - Agentic RAG"
Write-Host ""

# Pre-flight checks
$MergedModel = Join-Path $ROOT "fine_tuned\stage15_spin\merged\config.json"
if (Test-Path $MergedModel) {
    Write-Host "  [OK] Fine-tuned model found (stage15_spin/merged)"
}
else {
    Write-Host "  [WARN] Fine-tuned model not found - will use Gemini fallback"
}

# Kill stale processes on ports 8000 and 3000
foreach ($Port in 8000, 3000) {
    $Connections = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Where-Object { $_.State -eq 'Listen' }
    foreach ($Conn in $Connections) {
        Write-Host "  [WARN] Port $Port in use (PID $($Conn.OwningProcess)) - killing"
        Stop-Process -Id $Conn.OwningProcess -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }
}

# Check if torch is available to decide mode
$torchCheck = python -c "import torch; print('yes')" 2>$null
if ($torchCheck -ne 'yes') {
    Write-Host "  [INFO] PyTorch not found - using SKIP_LOCAL_MODEL (Gemini-only mode)"
    $env:SKIP_LOCAL_MODEL = "true"
}

# Environment
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:TOKENIZERS_PARALLELISM = "false"

# Start Backend
Write-Host ""
Write-Host "  [1/2] Starting Python backend (FastAPI + RAG Engine)..."
Write-Host "        -> http://localhost:8000"
Write-Host ""

$BackendDir = Join-Path $ROOT "backend"
$BackendProcess = Start-Process -FilePath "python" -ArgumentList "server.py" -WorkingDirectory $BackendDir -PassThru -NoNewWindow
Write-Host "  Backend PID: $($BackendProcess.Id)"

# Wait for backend
Write-Host "  Waiting for backend..."
$BackendReady = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:8000/api/health" -UseBasicParsing -TimeoutSec 3 -ErrorAction SilentlyContinue
        if ($resp.StatusCode -eq 200) {
            Write-Host "  [OK] Backend responding"
            $BackendReady = $true
            break
        }
    }
    catch { }
    Start-Sleep -Seconds 2
}

if (-not $BackendReady) {
    Write-Host "  [WARN] Backend did not respond within 120s - continuing anyway"
}

# Start Frontend
Write-Host ""
Write-Host "  [2/2] Starting Next.js frontend -> http://localhost:3000"
Write-Host ""

$FrontendDir = Join-Path $ROOT "frontend"
$FrontendProcess = Start-Process -FilePath "npm.cmd" -ArgumentList "run","dev" -WorkingDirectory $FrontendDir -PassThru -NoNewWindow

Write-Host ""
Write-Host "  Both servers starting. Open http://localhost:3000"
Write-Host "  Press Ctrl+C to stop."
Write-Host ""

try {
    while ($true) {
        if ($BackendProcess.HasExited -and $FrontendProcess.HasExited) {
            Write-Host "  Both processes exited."
            break
        }
        Start-Sleep -Seconds 2
    }
}
finally {
    Write-Host ""
    Write-Host "  Shutting down..."
    if (-not $BackendProcess.HasExited) {
        Stop-Process -Id $BackendProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if (-not $FrontendProcess.HasExited) {
        Stop-Process -Id $FrontendProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "  Stopped."
}
