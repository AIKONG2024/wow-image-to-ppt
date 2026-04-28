param(
  [string]$HfToken = $env:HF_TOKEN,
  [string]$Python = "python",
  [switch]$SkipTorch,
  [switch]$SkipPaddle,
  [switch]$SkipSam3,
  [switch]$OcrGpu
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $ProjectRoot ".venv-ai"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvHf = Join-Path $VenvDir "Scripts\hf.exe"
$OcrVenvDir = Join-Path $ProjectRoot ".venv-ocr"
$OcrPython = Join-Path $OcrVenvDir "Scripts\python.exe"
$VendorDir = Join-Path $ProjectRoot "vendor"
$Sam3Dir = Join-Path $VendorDir "sam3"
$CacheDir = Join-Path $ProjectRoot "data\cache"

Set-Location $ProjectRoot
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
$env:XDG_CACHE_HOME = $CacheDir
$env:HOME = Join-Path $CacheDir "home"
$env:USERPROFILE = Join-Path $CacheDir "home"
$env:PADDLE_HOME = Join-Path $CacheDir "paddle"
$env:PADDLE_PDX_CACHE_HOME = Join-Path $CacheDir "paddlex"
$env:PADDLE_EXTENSION_DIR = Join-Path $CacheDir "paddle_extension"
$env:HF_HOME = Join-Path $CacheDir "huggingface"
$env:PIP_CACHE_DIR = Join-Path $CacheDir "pip"

if (-not (Test-Path $VenvPython)) {
  & $Python -m venv $VenvDir
  if ($LASTEXITCODE -ne 0) { throw "Failed to create AI virtual environment." }
}

& $VenvPython -m pip install --upgrade pip "setuptools<81" wheel
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip tooling." }
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "Failed to install base requirements." }

if (-not $SkipTorch) {
  & $VenvPython -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
  if ($LASTEXITCODE -ne 0) { throw "Failed to install CUDA PyTorch." }
}

if (-not $SkipPaddle) {
  if (-not (Test-Path $OcrPython)) {
    & $Python -m venv $OcrVenvDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to create OCR virtual environment." }
  }
  & $OcrPython -m pip install --upgrade pip "setuptools<81" wheel
  if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade OCR pip tooling." }
  & $OcrPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
  if ($LASTEXITCODE -ne 0) { throw "Failed to install OCR base requirements." }
  if ($OcrGpu) {
    & $OcrPython -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
    if ($LASTEXITCODE -ne 0) { throw "Failed to install PaddlePaddle GPU." }
  } else {
    & $OcrPython -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    if ($LASTEXITCODE -ne 0) { throw "Failed to install PaddlePaddle CPU." }
  }
  & $OcrPython -m pip install paddleocr
  if ($LASTEXITCODE -ne 0) { throw "Failed to install PaddleOCR." }
}

& $VenvPython -m pip install "huggingface_hub[cli]"
if ($LASTEXITCODE -ne 0) { throw "Failed to install Hugging Face tooling." }

if (-not $SkipSam3) {
  New-Item -ItemType Directory -Force -Path $VendorDir | Out-Null
  if (-not (Test-Path (Join-Path $Sam3Dir ".git"))) {
    git -c http.sslBackend=openssl clone https://github.com/facebookresearch/sam3.git $Sam3Dir
    if ($LASTEXITCODE -ne 0) { throw "Failed to clone SAM3." }
  } else {
    git -C $Sam3Dir -c http.sslBackend=openssl pull --ff-only
    if ($LASTEXITCODE -ne 0) { throw "Failed to update SAM3." }
  }
  & $VenvPython -m pip install --no-cache-dir -e $Sam3Dir
  if ($LASTEXITCODE -ne 0) { throw "Failed to install SAM3." }
  & $VenvPython -m pip install einops triton-windows pycocotools
  if ($LASTEXITCODE -ne 0) { throw "Failed to install SAM3 runtime extras." }
}

if ($HfToken) {
  $env:HF_TOKEN = $HfToken
  $env:HUGGINGFACE_HUB_TOKEN = $HfToken
  if (Test-Path $VenvHf) {
    & $VenvHf auth login --token $HfToken
  }
}

& (Join-Path $PSScriptRoot "check-ai-runtime.ps1")
