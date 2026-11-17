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

# QtWebEngine flicker workaround (Wayland): Qt 6's native Wayland backend
# composites QWebEngineView through a separate sub-surface whose damage
# commits race with the Qt frame, flickering the 3D viewport (QTBUG-111836).
# Run on XWayland instead — the view is embedded via X11 reparenting there,
# no surface race, and NVIDIA PRIME GLX offload is fully native on X11.
# Side effect: the "GBM is not supported... Fallback to Vulkan" Chromium line
# also disappears (ozone-x11 does not use GBM). Delete this line to go back
# to native Wayland.
export QT_QPA_PLATFORM=xcb

# KDE theme + native file dialogs (Linux). PyQt6 bundles its own Qt and only
# searches its private plugin dir, which ships GTK/portal themes but NOT
# KDEPlasmaPlatformTheme6.so. Without the system plugin path Qt falls back to
# the Fusion style and draws its own file picker with no Places/quick folders.
# Adding the distro dir lets Qt load the real KDE theme, so Open/Save use
# Dolphin's native picker (with bookmarks, Recent, Devices). Harmless when the
# directory is absent (non-KDE desktops, Windows/macOS); override by exporting
# POTREEEDITOR_KEEP_QT_PLUGINS=1 before launching.
if [ "${POTREEEDITOR_KEEP_QT_PLUGINS:-0}" != "1" ]; then
    for _qt6_dir in /usr/lib/qt6/plugins /usr/lib/x86_64-linux-gnu/qt6/plugins; do
        if [ -d "$_qt6_dir" ]; then
            export QT_PLUGIN_PATH="${QT_PLUGIN_PATH:+$QT_PLUGIN_PATH:}$_qt6_dir"
        fi
    done
fi

exec .venv/bin/python main.py "$@"
