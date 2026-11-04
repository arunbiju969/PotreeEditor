#!/usr/bin/env bash
# =============================================================================
#  PotreeEditor :: ForestFormer3D AI segmentation engine setup
#
#  Brings the editor's AI dependency to a working state in one run:
#
#    1. vendor/ForestFormer3D  — submodule, checked out at the pinned revision
#                                recorded in forestformer3d.version
#    2. model weights          — clean_forestformer.zip from Zenodo (220 MB .pth)
#    3. torch env              — ~/.venvs/ff3d (opt-in: --venv)
#    4. verify                 — import ff3d_api, report CUDA + checkpoint
#
#  Safe to re-run: every step is skipped when already satisfied.
#
#  Usage:  ./setup_ff3d.sh [--venv] [--weights] [--skip-weights] [--smoke]
#            --venv           create the torch env with the vendored setup_linux.sh
#            --weights        re-download the checkpoint even if present
#            --skip-weights   do not fetch the checkpoint (print instructions)
#            --smoke          run a 5 000-point inference to prove the stack works
# =============================================================================
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SUB_PATH="vendor/ForestFormer3D"
PIN_FILE="forestformer3d.version"
VENV="${FF3D_VENV:-$HOME/.venvs/ff3d}"
CKPT_REL="work_dirs/clean_forestformer/epoch_3000_fix.pth"
WEIGHTS_URL="https://zenodo.org/api/records/16742708/files/clean_forestformer.zip/content"

C_CYN='\033[1;36m'; C_OK='\033[1;32m'; C_WARN='\033[1;33m'; C_ERR='\033[1;31m'; C_OFF='\033[0m'
ts()   { printf "\n$C_CYN[%s] ══ %s$C_OFF\n" "$(date +%H:%M:%S)" "$*"; }
ok()   { printf "$C_OK[%s] ✔ %s$C_OFF\n" "$(date +%H:%M:%S)" "$*"; }
warn() { printf "$C_WARN[%s] ⚠ %s$C_OFF\n" "$(date +%H:%M:%S)" "$*"; }
die()  { printf "$C_ERR[%s] ✘ %s$C_OFF\n" "$(date +%H:%M:%S)" "$*"; exit 1; }

MAKE_VENV=0; FETCH_WEIGHTS=1; SMOKE=0
for arg in "$@"; do
    case "$arg" in
        --venv)          MAKE_VENV=1 ;;
        --weights)       FETCH_WEIGHTS=1 ;;
        --skip-weights)  FETCH_WEIGHTS=0 ;;
        --smoke)         SMOKE=1 ;;
        -h|--help)       sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)               die "unknown option: $arg (see --help)" ;;
    esac
done

echo "================================================================="
echo "  PotreeEditor :: ForestFormer3D setup"
echo "================================================================="

# ---- 0. toolchain preflight ------------------------------------------------
ts "STEP 0/4  Preflight"
command -v git >/dev/null 2>&1 || die "git not found on PATH."
PIN_URL=$(sed -n 's/^url[[:space:]]*=[[:space:]]*//p'    "$PIN_FILE" 2>/dev/null)
PIN_SHA=$(sed -n 's/^commit[[:space:]]*=[[:space:]]*//p' "$PIN_FILE" 2>/dev/null)
[ -n "$PIN_URL" ] && [ -n "$PIN_SHA" ] || die "cannot read url/commit from $PIN_FILE"
ok "pinned revision: ${PIN_SHA:0:12} ($PIN_URL)"

# ---- 1. submodule: sources at the pinned revision --------------------------
ts "STEP 1/4  Sources → $SUB_PATH"
if [ ! -d "$SUB_PATH/.git" ] && [ ! -f "$SUB_PATH/.git" ]; then
    echo "[INFO] Initialising submodule ..."
    git submodule update --init --recursive "$SUB_PATH" \
        || die "submodule init failed. Is the network reachable?"
fi
if [ -n "$PIN_SHA" ] && [ "$(git -C "$SUB_PATH" rev-parse HEAD 2>/dev/null)" != "$PIN_SHA" ]; then
    echo "[INFO] Superproject pins ${PIN_SHA:0:12}; submodule is at $(git -C "$SUB_PATH" rev-parse --short HEAD). Syncing ..."
    git submodule update --init --recursive "$SUB_PATH" \
        || die "submodule sync failed."
fi
[ -f "$SUB_PATH/ff3d_api.py" ] \
    || die "$SUB_PATH has no ff3d_api.py — submodule revision predates the Python API.
       Fix: git submodule update --remote $SUB_PATH"
[ -f "$SUB_PATH/configs/oneformer3d_qs_radius16_qp300_2many.py" ] \
    || die "inference config missing in $SUB_PATH — unexpected submodule content."
ok "sources ready at $(git -C "$SUB_PATH" rev-parse --short HEAD)"

# ---- 2. model weights (gitignored inside the submodule) --------------------
ts "STEP 2/4  Weights"
CKPT="$SUB_PATH/$CKPT_REL"
if [ -f "$CKPT" ]; then
    ok "checkpoint present: $CKPT_REL ($(du -h "$CKPT" | cut -f1))"
elif [ "$FETCH_WEIGHTS" -eq 0 ]; then
    warn "no checkpoint. Fetch it manually:"
    echo "        curl -L -C - -o /tmp/clean_forestformer.zip '$WEIGHTS_URL'"
    echo "        unzip -o /tmp/clean_forestformer.zip -d $SUB_PATH/work_dirs/"
else
    command -v curl >/dev/null 2>&1 || die "curl not found — needed for the Zenodo download."
    command -v unzip >/dev/null 2>&1 || die "unzip not found — needed to unpack the checkpoint."
    ZIP="$SUB_PATH/work_dirs/clean_forestformer.zip"
    mkdir -p "$(dirname "$CKPT")"
    echo "[INFO] Downloading clean_forestformer.zip (~190 MB) from Zenodo ..."
    curl -L --fail -C - -o "$ZIP" "$WEIGHTS_URL" || die "Zenodo download failed."
    echo "[INFO] Unpacking → $CKPT_REL ..."
    unzip -o -q "$ZIP" -d "$SUB_PATH/work_dirs/" || die "unzip failed."
    [ -f "$CKPT" ] || die "unpacked, but $CKPT_REL is still missing — check the zip layout."
    rm -f "$ZIP"
    ok "checkpoint ready ($(du -h "$CKPT" | cut -f1))"
fi

# ---- 3. torch interpreter --------------------------------------------------
ts "STEP 3/4  Torch env (${VENV/#$HOME/~})"
PY=""
for c in "${FF3D_PYTHON:-}" "$VENV/bin/python" "$VENV/Scripts/python.exe"; do
    [ -n "$c" ] && [ -x "$c" ] && { PY="$c"; break; }
done
if [ -z "$PY" ] && [ "$MAKE_VENV" -eq 1 ]; then
    echo "[INFO] Building the env with the vendored recipe (long: torch + spconv + kernels) ..."
    FF3D_REPO="$PWD/$SUB_PATH" bash "$SUB_PATH/setup_linux.sh" || die "setup_linux.sh failed."
    for c in "$VENV/bin/python" "$VENV/Scripts/python.exe"; do
        [ -x "$c" ] && { PY="$c"; break; }
    done
fi
[ -n "$PY" ] || die "No torch interpreter. Re-run with --venv to build ~/.venvs/ff3d,
       or export FF3D_PYTHON=/path/to/env/bin/python (Windows: conda env 'ff3d')."
ok "interpreter: ${PY/#$HOME/~}"

# ---- 4. verify the pieces the editor actually calls ------------------------
ts "STEP 4/4  Verify"
FF3D_REPO="$PWD/$SUB_PATH" PYTHONPATH="$PWD/$SUB_PATH" "$PY" - <<'PYEOF' || die "ff3d_api smoke import failed."
import os
import torch
import ff3d_api
cfg = ff3d_api._DEFAULT_CONFIG
ckp = ff3d_api._DEFAULT_CKPT
print(f"  ff3d_api     : {os.path.dirname(os.path.abspath(ff3d_api.__file__))}")
print(f"  torch        : {torch.__version__}   cuda: {torch.cuda.is_available()} "
      f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu only'})")
print(f"  config       : {'ok' if os.path.isfile(cfg) else 'MISSING'}  {os.path.relpath(cfg)}")
print(f"  checkpoint   : {'ok' if os.path.isfile(ckp) else 'MISSING'}  {os.path.relpath(ckp)}")
if not (os.path.isfile(cfg) and os.path.isfile(ckp) and torch.cuda.is_available()):
    raise SystemExit(1)
PYEOF
ok "ForestFormer3D is importable and complete."

if [ "$SMOKE" -eq 1 ]; then
    ts "SMOKE  5 000-point inference"
    FF3D_REPO="$PWD/$SUB_PATH" PYTHONPATH="$PWD/$SUB_PATH" "$PY" - <<'PYEOF' || die "smoke inference failed."
import numpy as np, time, ff3d_api
rng = np.random.default_rng(0)
xyz = (rng.random((5000, 3), dtype=np.float32) * [10, 10, 6])
t0 = time.time(); ff = ff3d_api.ForestFormer3D(); load = time.time() - t0
t0 = time.time(); sem, inst, scores = ff.predict_points(xyz); infer = time.time() - t0
print(f"  model load {load:.1f}s · inference {infer:.2f}s · "
      f"{len(np.unique(inst[inst >= 0]))} tree instance(s) over {len(xyz):,} points")
PYEOF
fi

echo
echo "================================================================="
echo "  Ready. AI segmentation = the 'Segment (ForestFormer3D)' button."
echo "  Launch the editor with ./start.sh"
echo "================================================================="
