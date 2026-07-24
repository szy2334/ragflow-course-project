@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "PROJECT_ROOT=%CD%"
set "BACKEND_URL=http://127.0.0.1:8000/api/v1/health"
set "FRONTEND_URL=http://127.0.0.1:5173/"
set "EXIT_CODE=0"

echo Starting the paper understanding application...

if not exist "%PROJECT_ROOT%\backend\.env" (
    echo [ERROR] Missing backend\.env.
    set "EXIT_CODE=1"
    goto :finish
)

if not exist "%PROJECT_ROOT%\backend\.venv\Scripts\python.exe" (
    echo [ERROR] Missing backend virtual environment.
    echo         Run backend\install_deps.cmd first.
    set "EXIT_CODE=1"
    goto :finish
)

if not exist "%PROJECT_ROOT%\frontend\node_modules\vite\bin\vite.js" (
    echo [ERROR] Missing frontend dependencies.
    echo         Run npm install in the frontend directory first.
    set "EXIT_CODE=1"
    goto :finish
)

if not exist "%PROJECT_ROOT%\backend\var" mkdir "%PROJECT_ROOT%\backend\var"

call :port_in_use 8000
if errorlevel 1 (
    echo [BACKEND] Port 8000 is already in use; skipping duplicate start.
) else (
    echo [BACKEND] Starting on http://127.0.0.1:8000 ...
    start "Paper Agent Backend" /min /D "%PROJECT_ROOT%\backend" "%ComSpec%" /d /c call start_backend.cmd
)

call :wait_for_url "%BACKEND_URL%"
if errorlevel 1 (
    echo [ERROR] Backend health check failed. See backend\var\backend.log.
    set "EXIT_CODE=1"
) else (
    echo [BACKEND] Ready.
)

call :port_in_use 5173
if errorlevel 1 (
    echo [FRONTEND] Port 5173 is already in use; skipping duplicate start.
) else (
    echo [FRONTEND] Starting on http://127.0.0.1:5173 ...
    start "Paper Agent Frontend" /min /D "%PROJECT_ROOT%\backend" "%ComSpec%" /d /c call start_frontend.cmd
)

call :wait_for_url "%FRONTEND_URL%"
if errorlevel 1 (
    echo [ERROR] Frontend health check failed. See backend\var\frontend.log.
    set "EXIT_CODE=1"
) else (
    echo [FRONTEND] Ready.
)

if "%EXIT_CODE%"=="0" (
    echo.
    echo Application started successfully:
    echo   Frontend: %FRONTEND_URL%
    echo   Backend:  http://127.0.0.1:8000
)

:finish
if /I "%~1"=="--no-pause" exit /b %EXIT_CODE%
echo.
pause
exit /b %EXIT_CODE%

:port_in_use
powershell.exe -NoProfile -Command ^
    "if (Get-NetTCPConnection -State Listen -LocalPort %~1 -ErrorAction SilentlyContinue) { exit 1 } else { exit 0 }"
exit /b %ERRORLEVEL%

:wait_for_url
for /L %%I in (1,1,30) do (
    powershell.exe -NoProfile -Command ^
        "try { $response = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri '%~1'; if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) { exit 0 }; exit 1 } catch { exit 1 }"
    if not errorlevel 1 exit /b 0
    powershell.exe -NoProfile -Command "Start-Sleep -Seconds 1"
)
exit /b 1
