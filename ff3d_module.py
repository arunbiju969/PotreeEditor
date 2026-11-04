"""ForestFormer3D as an in-app segmentation module.

Runs ForestFormer3D's Python API (ff3d_api.ForestFormer3D.predict_points) in a
persistent subprocess of the ff3d torch env — the app's own interpreter has
no torch, and the model (~10 GB VRAM when warm) is built once and reused for
every click. Protocol: one JSON line in {"in": npz_path, "out": npz_path},
one JSON line out {"ok": ..., "n_trees": ...}.

The model sources are the vendored git submodule in ./vendor/ForestFormer3D
(pinned revision: forestformer3d.version; bootstrap: setup_ff3d.sh). Both
halves can be pointed elsewhere with environment variables, no code edit:

    FF3D_REPO    ForestFormer3D checkout   (default ./vendor/ForestFormer3D)
    FF3D_VENV    torch env prefix          (default ~/.venvs/ff3d)
    FF3D_PYTHON  interpreter in that env   (default <FF3D_VENV>/bin/python)

The checkpoint itself (work_dirs/clean_forestformer/epoch_3000_fix.pth, 220 MB)
is gitignored inside the submodule — see the setup script for fetching it.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _candidates(paths):
    """Yield existing, absolute paths in priority order."""
    for p in paths:
        if not p:
            continue
        p = os.path.expanduser(p)
        if os.path.exists(p):
            return p
    return None


def _repo_root():
    return _candidates([
        os.environ.get("FF3D_REPO"),
        os.path.join(_HERE, "vendor", "ForestFormer3D"),
        "/run/media/neo/Omen/Projects/ForestFormer3D",   # legacy external checkout
    ])


def _interpreter():
    env = os.environ.get("FF3D_PYTHON")
    venv = os.environ.get("FF3D_VENV", os.path.expanduser("~/.venvs/ff3d"))
    return _candidates([
        env,
        os.path.join(venv, "bin", "python"),             # Linux / macOS
        os.path.join(venv, "Scripts", "python.exe"),     # Windows
    ])


FF3D_REPO = _repo_root()
FF3D_PYTHON = _interpreter()

_RUNNER = '''
import sys, json
import numpy as np
from ff3d_api import ForestFormer3D

ff = ForestFormer3D()
print("##FF3D## " + json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    try:
        d = np.load(req["in"])
        sem, inst, scores = ff.predict_points(d["xyz"])
        np.savez(req["out"], sem=sem.astype(np.int16), inst=inst.astype(np.int32),
                 scores=scores.astype(np.float32))
        out = {"ok": True, "n_trees": int(len(np.unique(inst[inst >= 0])))}
    except Exception as e:
        out = {"ok": False, "error": str(e)}
    print("##FF3D## " + json.dumps(out), flush=True)
'''


def _read_tagged(stdout):
    """Read stdout lines until one carries our tag; model chatter is ignored."""
    for line in stdout:
        line = line.strip()
        if line.startswith("##FF3D## "):
            return json.loads(line[9:])
    raise RuntimeError("ForestFormer3D worker closed the pipe")


class ForestFormer3DModule:
    """Persistent ForestFormer3D worker; predict(xyz) blocks ~2-10 s."""

    def __init__(self):
        self._proc = None
        self._tmp = tempfile.mkdtemp(prefix="ff3d_")
        runner = os.path.join(self._tmp, "_ff3d_runner.py")
        with open(runner, "w", encoding="utf-8") as f:
            f.write(_RUNNER)
        self._runner = runner

    def _ensure(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        if FF3D_REPO is None:
            raise RuntimeError(
                "ForestFormer3D sources not found. Run ./setup_ff3d.sh, or "
                "git submodule update --init vendor/ForestFormer3D, or set FF3D_REPO.")
        if FF3D_PYTHON is None:
            raise RuntimeError(
                "ForestFormer3D torch env not found. Create ~/.venvs/ff3d "
                "(see vendor/ForestFormer3D/setup_linux.sh) or set FF3D_PYTHON.")
        if not os.path.isfile(os.path.join(FF3D_REPO, "ff3d_api.py")):
            raise RuntimeError(
                f"{FF3D_REPO} has no ff3d_api.py — the submodule is at a revision "
                "predating the Python API; run: git submodule update --remote vendor/ForestFormer3D")
        env = dict(os.environ, PYTHONPATH=FF3D_REPO)
        self._proc = subprocess.Popen(
            [FF3D_PYTHON, self._runner],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, env=env,
            cwd=FF3D_REPO)
        ready = _read_tagged(self._proc.stdout)   # blocks through model load (~6 s)
        if not ready.get("ready"):
            raise RuntimeError("ForestFormer3D worker died on startup")

    def predict(self, xyz: np.ndarray):
        """xyz (N,3) float32 -> (sem int16, inst int32, scores float32)."""
        self._ensure()
        tag = f"{int(time.time()*1000)%10**9}"
        in_path = os.path.join(self._tmp, f"{tag}_in.npz")
        out_path = os.path.join(self._tmp, f"{tag}_out.npz")
        np.savez(in_path, xyz=np.asarray(xyz, dtype=np.float32))
        self._proc.stdin.write(json.dumps({"in": in_path, "out": out_path}) + "\n")
        self._proc.stdin.flush()
        res = _read_tagged(self._proc.stdout)
        if not res.get("ok"):
            raise RuntimeError(f"ForestFormer3D failed: {res}")
        d = np.load(out_path)
        os.remove(in_path)
        os.remove(out_path)
        return d["sem"], d["inst"], d["scores"]

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.stdin.close()
            self._proc.terminate()
            self._proc = None


# One shared module per app process.
_module = None


def get_module() -> ForestFormer3DModule:
    global _module
    if _module is None:
        _module = ForestFormer3DModule()
    return _module
