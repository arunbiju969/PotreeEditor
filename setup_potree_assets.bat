@echo off
setlocal enableextensions
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

REM =====================================================================
REM  First-time setup: obtain and build the vendored Potree renderer.
REM  The editor renders from  .\potree\build\potree\potree.js
REM  which is NOT tracked by the potree repository, so it must be built.
REM  Reads the pinned revision from  .\potree.version
REM =====================================================================

set "POTREE_URL=https://github.com/arunbiju969/potree.git"
set "POTREE_SHA=57473258d220bc75f368377e99348d1306233e39"

echo =================================================================
echo   PotreeEditor :: vendored renderer setup
echo =================================================================

REM ---- 1. toolchain preflight -----------------------------------------
where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] git not found on PATH. Install Git for Windows first.
    pause & exit /b 1
)
where npm >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm not found on PATH. Install Node.js ^(LTS^) first.
    echo         The renderer is built with gulp + rollup via "npm install".
    pause & exit /b 1
)

REM ---- 2. clone at the pinned SHA (only if missing) -------------------
if not exist "potree\build\potree\potree.js" (
    if not exist "potree\.git" (
        echo [INFO] Cloning %POTREE_URL% at %POTREE_SHA% ...
        git clone --no-checkout "%POTREE_URL%" potree
        if errorlevel 1 ( echo [ERROR] clone failed. & pause & exit /b 1 )
    )
    echo [INFO] Checking out %POTREE_SHA% ...
    git -C potree fetch origin
    git -C potree checkout --detach %POTREE_SHA%
    if errorlevel 1 ( echo [ERROR] checkout of pinned SHA failed. & pause & exit /b 1 )
)

REM ---- 3. build the assets (postinstall runs: npm run build) ----------
if not exist "potree\build\potree\potree.js" (
    echo [INFO] Installing dependencies and building assets ...
    pushd potree
    call npm install --no-audit --no-fund
    if errorlevel 1 ( popd & echo [ERROR] npm install/build failed. & pause & exit /b 1 )
    popd
)

REM ---- 4. verify the artifacts the editor actually loads --------------
set "OK=1"
for %%F in (
    "potree\build\potree\potree.js"
    "potree\build\potree\potree.css"
    "potree\libs\three.js\build\three.module.js"
    "potree\libs\jquery\jquery-3.1.1.min.js"
    "potree\libs\PotreeConverter2\PotreeConverter.exe"
) do (
    if not exist %%F (
        echo [ERROR] missing required asset: %%~F
        set "OK=0"
    )
)

if "%OK%"=="0" (
    echo.
    echo [ERROR] Renderer is incomplete. Re-run this script, or inside .\potree :
    echo             npm install
    echo             npm run build
    pause & exit /b 1
)

echo [INFO] Renderer revision:
git -C potree rev-parse HEAD
echo [INFO] Renderer assets ready.
echo [INFO] Launch with: run_potree_editor.bat
endlocal
