param(
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

$env:PYTHONPATH = Join-Path $ProjectRoot "backend"
& $VenvPython (Join-Path $ProjectRoot "scripts\check_environment.py")
& $VenvPython -c "import json; from app.analysis import runtime_status; print(json.dumps(runtime_status(), ensure_ascii=False, indent=2))"
if (Test-Path $OcrPython) {
  & $OcrPython -c "import json, importlib.util; print(json.dumps({'ocr_python': r'$OcrPython', 'paddleocr': importlib.util.find_spec('paddleocr') is not None, 'paddle': importlib.util.find_spec('paddle') is not None}, ensure_ascii=False, indent=2))"
}
