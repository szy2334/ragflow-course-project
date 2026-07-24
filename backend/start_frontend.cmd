@echo off
cd /d "%~dp0\..\frontend"
call npm run dev -- --host 127.0.0.1 --port 5173 > ..\backend\var\frontend.log 2>&1
