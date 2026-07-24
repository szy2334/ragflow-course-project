@echo off
cd /d "%~dp0"
call .venv\Scripts\python.exe -m pip install -e ".[test]" > var\venv_pip_out.txt 2> var\venv_pip_err.txt
echo EXIT=%ERRORLEVEL%>> var\venv_pip_out.txt
