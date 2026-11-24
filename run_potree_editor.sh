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

# Native file dialogs (Linux). PyQt6 bundles its own Qt and only searches its
# private plugin dir, which ships GTK/portal themes but NOT KDEPlasmaPlatformTheme6.so.
# When the system Qt6 minor version matches PyQt6's bundled Qt version, adding the
# distro plugin dir lets Qt load the native KDE/Dolphin theme. When system Qt6 has
# upgraded past PyQt6's bundled Qt (incompatible minor versions), or on desktops
# without KDEPlasmaPlatformTheme6.so (such as Ubuntu/GNOME), Qt rejects the system
# plugins and would otherwise fall back to Fusion's bare dialog; in that case use
# PyQt6's own version-matched xdgdesktopportal plugin over D-Bus instead.
# Override by exporting POTREEEDITOR_KEEP_QT_PLUGINS=1 before launching.
if [ "${POTREEEDITOR_KEEP_QT_PLUGINS:-0}" != "1" ]; then
    _venv_qt="$(.venv/bin/python -c 'from PyQt6.QtCore import QT_VERSION_STR; print(".".join(QT_VERSION_STR.split(".")[:2]))' 2>/dev/null || true)"
    _sys_qt=""
    if command -v qmake6 >/dev/null 2>&1; then
        _sys_qt="$(qmake6 -query QT_VERSION 2>/dev/null | cut -d. -f1,2 || true)"
    elif command -v qtpaths6 >/dev/null 2>&1; then
        _sys_qt="$(qtpaths6 --qt-version 2>/dev/null | cut -d. -f1,2 || true)"
    fi

    _loaded_sys_kde=0
    if [ -n "$_venv_qt" ] && [ "$_venv_qt" = "$_sys_qt" ]; then
        for _qt6_dir in /usr/lib/qt6/plugins /usr/lib/x86_64-linux-gnu/qt6/plugins; do
            if [ -f "$_qt6_dir/platformthemes/KDEPlasmaPlatformTheme6.so" ]; then
                export QT_PLUGIN_PATH="${QT_PLUGIN_PATH:+$QT_PLUGIN_PATH:}$_qt6_dir"
                _loaded_sys_kde=1
            fi
        done
    fi

    if [ "$_loaded_sys_kde" -eq 0 ] && [ -z "${QT_QPA_PLATFORMTHEME:-}" ]; then
        export QT_QPA_PLATFORMTHEME=xdgdesktopportal
    fi
fi

exec .venv/bin/python main.py "$@"
