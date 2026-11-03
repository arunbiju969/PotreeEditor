#!/usr/bin/env bash
# Linux equivalent of run_potree_editor.bat — launches PotreeEditor on the NVIDIA GPU.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

if [ ! -d .venv ]; then
    echo "[ERROR] .venv missing. Create it with:"
    echo "    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi
if [ ! -f potree/build/potree/potree.js ]; then
    echo "[ERROR] Renderer not built. Run:"
    echo "    cd potree && npm install"
    exit 1
fi

# PRIME offload: force the discrete NVIDIA GPU (RTX 5090) instead of the Intel iGPU.
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export DRI_PRIME=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.mplcache}"

exec .venv/bin/python main.py "$@"
