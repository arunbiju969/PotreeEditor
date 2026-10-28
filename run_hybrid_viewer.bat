@echo off
setlocal enableextensions
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo =================================================================
echo   Starting Potree Hybrid Point Cloud Editor (Experimental)
echo =================================================================

:: Check for existing TreeAIBox venv or local venv
if exist "%~dp0..\TreeAIBox\.venv\Scripts\activate.bat" (
    echo [INFO] Activating Python environment from TreeAIBox\.venv...
    call "%~dp0..\TreeAIBox\.venv\Scripts\activate.bat"
) else if exist "%~dp0.venv\Scripts\activate.bat" (
    echo [INFO] Activating local .venv...
    call "%~dp0.venv\Scripts\activate.bat"
) else (
    echo [WARNING] No virtual environment found. Using system Python...
)

echo [INFO] Launching Hybrid Editor...
python "%~dp0main.py" %*

if %errorlevel% neq 0 (
    echo [ERROR] Application exited with error code %errorlevel%.
    pause
)
