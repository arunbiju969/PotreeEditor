"""ForestFormer3D as an in-app segmentation module.

Runs ForestFormer3D's Python API (ff3d_api.ForestFormer3D.predict_points) in a
persistent subprocess of the ff3d conda/uv env — the app's own interpreter has
no torch, and the model (~10 GB VRAM when warm) is built once and reused for
every click. Protocol: one JSON line in {"in": npz_path, "out": npz_path},
one JSON line out {"ok": ..., "n_trees": ...}.
"""
import json
import os
import subprocess
import tempfile
import time

import numpy as np

FF3D_PYTHON = os.path.expanduser("~/.venvs/ff3d/bin/python")
FF3D_REPO = "/run/media/neo/Omen/Projects/ForestFormer3D"

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
