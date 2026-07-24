@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "EXIT_CODE=0"

echo Stopping the paper understanding application...

call :stop_port 5173 frontend
call :stop_port 8000 backend

if "%EXIT_CODE%"=="0" (
    echo.
    echo Frontend and backend are stopped.
)

if /I "%~1"=="--no-pause" exit /b %EXIT_CODE%
echo.
pause
exit /b %EXIT_CODE%

:stop_port
set "TARGET_PORT=%~1"
set "SERVICE_NAME=%~2"
set "TARGET_PID="

for /f "delims=" %%P in ('powershell.exe -NoProfile -Command ^
    "$connections = @(Get-NetTCPConnection -State Listen -LocalPort !TARGET_PORT! -ErrorAction SilentlyContinue); if ($connections.Count -gt 0) { $connections[0].OwningProcess }"') do set "TARGET_PID=%%P"

if not defined TARGET_PID (
    echo [!SERVICE_NAME!] Already stopped; port !TARGET_PORT! is free.
    exit /b 0
)

powershell.exe -NoProfile -Command ^
    "$process = Get-CimInstance Win32_Process -Filter 'ProcessId = !TARGET_PID!'; $command = [string]$process.CommandLine; if ('!TARGET_PORT!' -eq '8000' -and $command -match 'uvicorn' -and $command -match 'app\.main:app') { exit 0 }; if ('!TARGET_PORT!' -eq '5173' -and $command -match 'vite') { exit 0 }; exit 1"

if errorlevel 1 (
    echo [ERROR] Port !TARGET_PORT! is owned by an unrecognized process ^(PID !TARGET_PID!^).
    echo         It was not stopped.
    set "EXIT_CODE=1"
    exit /b 0
)

taskkill.exe /PID !TARGET_PID! /T /F >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Failed to stop !SERVICE_NAME! ^(PID !TARGET_PID!^).
    set "EXIT_CODE=1"
    exit /b 0
)

powershell.exe -NoProfile -Command "Start-Sleep -Seconds 1"
powershell.exe -NoProfile -Command ^
    "if (Get-NetTCPConnection -State Listen -LocalPort !TARGET_PORT! -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }"
if errorlevel 1 (
    echo [ERROR] !SERVICE_NAME! still occupies port !TARGET_PORT!.
    set "EXIT_CODE=1"
) else (
    echo [!SERVICE_NAME!] Stopped; port !TARGET_PORT! is free.
)
exit /b 0
