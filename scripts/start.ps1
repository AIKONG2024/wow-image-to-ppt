$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
