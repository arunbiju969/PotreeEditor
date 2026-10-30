@echo off
setlocal enableextensions
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo =================================================================
echo   Starting Potree Hybrid Point Cloud Editor (Experimental)
echo =================================================================

REM  Everything this app needs lives inside this folder:
REM     .\.venv    Python environment (PyQt6, laspy, lazrs, NumPy, SciPy, Matplotlib)
REM     .\potree   renderer, pinned in potree.version  (build/ must be built once)
REM  No sibling projects and no environment variables are consulted.

REM ---- 1. Python environment must be the local one ---------------------
if not exist "%~dp0.venv\Scripts\activate.bat" (
    echo [ERROR] No local virtual environment found at .\.venv
    echo         Create it, from this folder:
    echo             python -m venv .venv
    echo             .venv\Scripts\activate ^&^& pip install -r requirements.txt
    pause
    exit /b 1
)
echo [INFO] Activating local .venv...
call "%~dp0.venv\Scripts\activate.bat"

REM ---- 2. Renderer assets must have been built -------------------------
if not exist "%~dp0potree\build\potree\potree.js" (
    echo [ERROR] Renderer not built: .\potree\build\potree\potree.js is missing.
    echo         Run the one-time setup, which clones, runs npm install and verifies:
    echo             setup_potree_assets.bat
    pause
    exit /b 1
)

echo [INFO] Launching Hybrid Editor...
python "%~dp0main.py" %*

if %errorlevel% neq 0 (
    echo [ERROR] Application exited with error code %errorlevel%.
    pause
)
