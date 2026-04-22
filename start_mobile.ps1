$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$mobileDir = Join-Path $repoRoot "mobile"

Write-Host "Starting Cortex Mobile (Expo)..." -ForegroundColor Cyan

if (-not (Test-Path $mobileDir)) {
  throw "Mobile directory not found: $mobileDir"
}

Set-Location $mobileDir

if (-not (Test-Path "node_modules")) {
  Write-Host "Installing mobile dependencies..." -ForegroundColor Yellow
  npm install
}

if (-not (Test-Path ".env")) {
  if (Test-Path ".env.example") {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example. Update EXPO_PUBLIC_API_BASE_URL if needed." -ForegroundColor Yellow
  }
}

npm run start
