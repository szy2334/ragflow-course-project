@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 > var\backend.log 2>&1
