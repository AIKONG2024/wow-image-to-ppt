$ErrorActionPreference = "Stop"
Set-Location "C:\Users\ust21\Documents\ppt-agent-studio"
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
