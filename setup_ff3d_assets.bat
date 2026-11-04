@echo off
setlocal enableextensions
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

REM =====================================================================
REM  First-time setup: obtain the AI segmentation engine (ForestFormer3D)
REM  and its weights, so the "Segment (ForestFormer3D)" button works.
REM
REM    1. vendor\ForestFormer3D  - git submodule at the revision pinned in
REM                                forestformer3d.version
REM    2. model weights          - clean_forestformer.zip from Zenodo (220 MB)
REM    3. verify                 - import ff3d_api from the 'ff3d' conda env
REM
REM  Safe to re-run: completed steps are skipped.
REM  Usage:  setup_ff3d_assets.bat [--skip-weights] [--smoke]
REM =====================================================================

set "SUB=vendor\ForestFormer3D"
set "PINFILE=forestformer3d.version"
set "CKPT=%SUB%\work_dirs\clean_forestformer\epoch_3000_fix.pth"
set "WEIGHTS_URL=https://zenodo.org/api/records/16742708/files/clean_forestformer.zip/content"
set "PYFF3D="

set "SKIPW=0"
set "SMOKE=0"
:args
if "%~1"=="" goto args_done
if /i "%~1"=="--skip-weights" set "SKIPW=1"
if /i "%~1"=="--smoke" set "SMOKE=1"
shift
goto args
:args_done

echo =================================================================
echo   PotreeEditor :: ForestFormer3D AI engine setup
echo =================================================================

REM ---- 0. toolchain preflight ---------------------------------------
where git >nul 2>&1
if errorlevel 1 ( echo [ERROR] git not found on PATH. Install Git for Windows first. & pause & exit /b 1 )

REM ---- 1. submodule: sources at the pinned revision ------------------
echo [INFO] Step 1/3  Sources -^> %SUB%
if not exist "%SUB%\.git" (
    echo [INFO] Initialising submodule ...
    git submodule update --init --recursive "%SUB%"
    if errorlevel 1 ( echo [ERROR] submodule init failed. Is the network reachable? & pause & exit /b 1 )
)
if not exist "%SUB%\ff3d_api.py" (
    echo [ERROR] %SUB% has no ff3d_api.py - submodule revision predates the Python API.
    echo         Fix: git submodule update --remote %SUB%
    pause & exit /b 1
)
for /f "delims=" %%H in ('git -C "%SUB%" rev-parse HEAD') do echo [INFO] Sources at %%H

REM ---- 2. model weights (gitignored inside the submodule) ------------
echo [INFO] Step 2/3  Weights
if exist "%CKPT%" (
    echo [INFO] Checkpoint present: work_dirs\clean_forestformer\epoch_3000_fix.pth
    goto weights_done
)
if "%SKIPW%"=="1" (
    echo [WARN] No checkpoint. Fetch it manually:
    echo            curl -L -o %TEMP%\clean_forestformer.zip "%WEIGHTS_URL%"
    echo            tar -xf %TEMP%\clean_forestformer.zip -d %SUB%\work_dirs
    goto weights_done
)
where curl >nul 2>&1
if errorlevel 1 ( echo [ERROR] curl not found - needed for the Zenodo download. & pause & exit /b 1 )
if not exist "%SUB%\work_dirs" mkdir "%SUB%\work_dirs"
echo [INFO] Downloading clean_forestformer.zip (~190 MB) from Zenodo ...
curl -L --fail -C - -o "%TEMP%\clean_forestformer.zip" "%WEIGHTS_URL%"
if errorlevel 1 ( echo [ERROR] Zenodo download failed. & pause & exit /b 1 )
echo [INFO] Unpacking to %SUB%\work_dirs ...
tar -xf "%TEMP%\clean_forestformer.zip" -C "%SUB%\work_dirs"
if errorlevel 1 ( echo [ERROR] unpack failed. & pause & exit /b 1 )
if not exist "%CKPT%" ( echo [ERROR] unpacked, but %CKPT% is still missing. & pause & exit /b 1 )
del "%TEMP%\clean_forestformer.zip" >nul 2>&1
echo [INFO] Checkpoint ready.
:weights_done

REM ---- 3. verify with the ff3d torch env -----------------------------
echo [INFO] Step 3/3  Verify
if defined FF3D_PYTHON (
    set "PYFF3D=%FF3D_PYTHON%"
) else (
    where conda >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] No 'ff3d' conda env found and conda is not on PATH.
        echo         Create it from %SUB%\SETUP_NOTES.md ^(conda env ^'ff3d^'^),
        echo         then re-run, or set FF3D_PYTHON=^<env^>\Scripts\python.exe
        pause & exit /b 1
    )
    for /f "delims=" %%P in ('conda run -n ff3d python -c "import sys;print(sys.executable)"') do set "PYFF3D=%%P"
)
if not defined PYFF3D ( echo [ERROR] Could not resolve the ff3d interpreter. & pause & exit /b 1 )
echo [INFO] Interpreter: %PYFF3D%
pushd "%SUB%"
set "FF3D_REPO=%CD%"
set "PYTHONPATH=%CD%"
"%PYFF3D%" -c "import os,torch,ff3d_api as f; print('[INFO] ff3d_api  :', os.path.dirname(os.path.abspath(f.__file__))); print('[INFO] torch     :', torch.__version__, 'cuda:', torch.cuda.is_available()); print('[INFO] ckpt    :', 'ok' if os.path.isfile(f._DEFAULT_CKPT) else 'MISSING'); raise SystemExit(0 if os.path.isfile(f._DEFAULT_CKPT) and torch.cuda.is_available() else 1)"
if errorlevel 1 ( popd & echo [ERROR] ForestFormer3D verification failed. & pause & exit /b 1 )
if "%SMOKE%"=="1" (
    echo [INFO] Smoke test - 5000-point inference ...
    "%PYFF3D%" -c "import numpy as np,time,ff3d_api as f; r=np.random.default_rng(0); x=(r.random((5000,3),dtype=np.float32)*[10,10,6]); t=time.time(); ff=f.ForestFormer3D(); s,i,sc=ff.predict_points(x); print('[INFO] load %.1fs inference %.2fs trees %d' % (time.time()-t, getattr(ff,'_last_infer_time',0), len(np.unique(i[i>=0]))))"
    if errorlevel 1 ( popd & echo [ERROR] Smoke inference failed. & pause & exit /b 1 )
)
popd

echo.
echo [INFO] AI segmentation engine ready. Launch with: run_potree_editor.bat
endlocal
