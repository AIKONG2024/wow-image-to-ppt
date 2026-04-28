param(
  [int]$Port = 8000,
  [string]$HfToken = $env:HF_TOKEN
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv-ai\Scripts\python.exe"
$OcrPython = Join-Path $ProjectRoot ".venv-ocr\Scripts\python.exe"
$CacheDir = Join-Path $ProjectRoot "data\cache"

if (-not (Test-Path $VenvPython)) {
  throw "AI runtime is not installed. Run scripts\setup-ai-runtime.ps1 first."
}
if (Test-Path $OcrPython) {
  $env:PPT_AGENT_PADDLEOCR_PYTHON = $OcrPython
}

New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
$env:XDG_CACHE_HOME = $CacheDir
$env:HOME = Join-Path $CacheDir "home"
$env:USERPROFILE = Join-Path $CacheDir "home"
$env:PADDLE_HOME = Join-Path $CacheDir "paddle"
$env:PADDLE_PDX_CACHE_HOME = Join-Path $CacheDir "paddlex"
$env:PADDLE_EXTENSION_DIR = Join-Path $CacheDir "paddle_extension"
$env:HF_HOME = Join-Path $CacheDir "huggingface"
$env:PIP_CACHE_DIR = Join-Path $CacheDir "pip"

if ($HfToken) {
  $env:HF_TOKEN = $HfToken
  $env:HUGGINGFACE_HUB_TOKEN = $HfToken
}

if (-not $env:PPT_AGENT_PADDLEOCR_DEVICE) {
  $env:PPT_AGENT_PADDLEOCR_DEVICE = "cpu"
}
Set-Location $ProjectRoot
& $VenvPython -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port $Port
