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
from collections import deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# Exit codes that explain themselves once you know the cause.
_EXIT_HINTS = {
    -9: "SIGKILL, typically the kernel reclaiming memory — try Tiled Tract mode",
    -11: "segmentation fault in the CUDA path",
    9: "typically out of memory — try Tiled Tract mode",
    137: "SIGKILL (128+9), typically the kernel reclaiming memory — try Tiled Tract mode",
    139: "segmentation fault (128+11) in the CUDA path",
}


class ForestFormer3DError(RuntimeError):
    """Engine failure; `exit_code` is the child's status when it died outright."""

    def __init__(self, message: str, exit_code=None):
        super().__init__(message)
        self.exit_code = exit_code


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


def repo_root():
    """Absolute path of the resolved ForestFormer3D checkout (None if missing)."""
    return _repo_root()


FF3D_REPO = _repo_root()
FF3D_PYTHON = _interpreter()

_RUNNER = '''
import os, sys, json
import numpy as np
from ff3d_api import ForestFormer3D

_model = None
_model_key = None


def model_for(req):
    """Build the model once and reuse it; rebuild only when a model-level
    parameter (weights, device, voxel radius/step, score threshold) changes."""
    global _model, _model_key
    key = (req.get("config") or "", req.get("checkpoint") or "",
           req.get("device") or "cuda:0", float(req.get("radius", 16.0)),
           float(req.get("step_size", 6.0)), float(req.get("score_th", 0.4)))
    if _model is not None and _model_key == key:
        return _model
    kw = {}
    if req.get("config"):
        kw["config"] = req["config"]
    if req.get("checkpoint"):
        kw["checkpoint"] = req["checkpoint"]
    if req.get("device"):
        kw["device"] = req["device"]
    for name in ("radius", "step_size", "score_th"):
        if req.get(name) is not None:
            kw[name] = float(req[name])
    _model = ForestFormer3D(**kw)
    _model_key = key
    return _model


print("##FF3D## " + json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    try:
        mode = req.get("mode", "points")
        if mode == "points":
            ff = model_for(req)
            d = np.load(req["in"])
            sem, inst, scores = ff.predict_points(d["xyz"])
            np.savez(req["out"], sem=sem.astype(np.int16), inst=inst.astype(np.int32),
                     scores=scores.astype(np.float32))
            out = {"ok": True, "n_trees": int(len(np.unique(inst[inst >= 0])))}
        elif mode == "tract":
            ff = model_for(req)
            las_out = ff.predict_tract(
                req["in"], req["out_dir"],
                tile_size=float(req.get("tile_size", 70.0)),
                overlap=float(req.get("overlap", 10.0)))
            out = {"ok": True, "out_las": las_out}
        else:  # single: one cylinder pass over the whole file
            ff = model_for(req)
            roi = req.get("roi")
            res = ff.predict(req["in"], roi=tuple(roi) if roi else None)
            stem = os.path.splitext(os.path.basename(req["in"]))[0]
            las_out = os.path.join(req["out_dir"], stem + "_ff3d_segmented.las")
            ForestFormer3D.save_las(req["in"], res, las_out)
            out = {"ok": True, "out_las": las_out, "n_trees": int(res.n_trees)}
    except Exception as e:
        # str(e) alone hides which frame broke; keep the last frames so the
        # console message is actionable.
        import traceback
        tail = " | ".join(traceback.format_exc().strip().splitlines()[-4:])
        out = {"ok": False, "error": tail}
    print("##FF3D## " + json.dumps(out), flush=True)
'''


def _read_tagged(stdout, on_line=None, proc=None):
    """Read stdout lines until one carries our tag.

    Everything else is the engine's own chatter (model build, per-tile lines,
    merge/stitch notes); `on_line` forwards it so a UI can show live progress.
    When the child dies, the message carries its exit code and the last line
    it printed — a bare "closed the pipe" is not actionable.
    """
    tail = deque(maxlen=2)
    for line in stdout:
        line = line.strip()
        if line.startswith("##FF3D## "):
            return json.loads(line[9:])
        if line:
            tail.append(line)
            if on_line is not None:
                on_line(line)
    code = proc.poll() if proc is not None else None
    hint = _EXIT_HINTS.get(code)
    detail = ""
    if code is not None:
        detail = f" (exit {code}" + (f": {hint}" if hint else "") + ")"
    last = tail[-1] if tail else ""
    raise ForestFormer3DError(
        f"ForestFormer3D engine stopped{detail}"
        + (f" — last line: {last[:150]}" if last else ""), exit_code=code)


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
            stderr=subprocess.STDOUT, text=True, env=env,
            cwd=FF3D_REPO)
        ready = _read_tagged(self._proc.stdout, proc=self._proc)   # blocks through model load (~6 s)
        if not ready.get("ready"):
            raise RuntimeError("ForestFormer3D worker died on startup")

    def predict(self, xyz: np.ndarray, on_output=None):
        """xyz (N,3) float32 -> (sem int16, inst int32, scores float32)."""
        self._ensure()
        tag = f"{int(time.time()*1000)%10**9}"
        in_path = os.path.join(self._tmp, f"{tag}_in.npz")
        out_path = os.path.join(self._tmp, f"{tag}_out.npz")
        np.savez(in_path, xyz=np.asarray(xyz, dtype=np.float32))
        self._proc.stdin.write(json.dumps({"in": in_path, "out": out_path}) + "\n")
        self._proc.stdin.flush()
        res = _read_tagged(self._proc.stdout, on_output, proc=self._proc)
        if not res.get("ok"):
            raise RuntimeError(f"ForestFormer3D failed: {res}")
        d = np.load(out_path)
        os.remove(in_path)
        os.remove(out_path)
        return d["sem"], d["inst"], d["scores"]

    def predict_file(self, in_las: str, out_dir: str, mode: str = "tract", *,
                     tile_size: float = 70.0, overlap: float = 10.0,
                     roi=None, model: dict = None, on_output=None) -> dict:
        """Full-pipeline run over a LAS/LAZ file (blocks for minutes on a tract).

        mode="tract"  tiles the cloud, runs each tile, merges and seam-stitches
                      -> <out_dir>/<stem>_entire_tract_stitched.las
        mode="single" one cylinder pass over the whole file
                      -> <out_dir>/<stem>_ff3d_segmented.las

        `model` optionally carries ForestFormer3D constructor overrides
        (config, checkpoint, device, radius, step_size, score_th); changing one
        rebuilds the warm model on the next request. `on_output` receives each
        engine log line — tile progress, merge, stitch — as it is printed.
        Returns the worker's reply: {"ok": True, "out_las": path, "n_trees": int|None}.
        """
        self._ensure()
        os.makedirs(out_dir, exist_ok=True)
        payload = {"mode": mode,
                   "in": os.path.abspath(in_las),
                   "out_dir": os.path.abspath(out_dir),
                   "tile_size": float(tile_size),
                   "overlap": float(overlap)}
        if roi:
            payload["roi"] = [float(v) for v in roi]
        for k, v in (model or {}).items():
            if v not in (None, ""):
                payload[k] = v
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()
        res = _read_tagged(self._proc.stdout, on_output, proc=self._proc)
        if not res.get("ok"):
            raise ForestFormer3DError(f"ForestFormer3D {mode} run failed: {res.get('error') or res}")
        return res

    def close(self):
        """Terminate the CUDA child so the GPU is actually released."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
        except OSError:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


# One shared module per app process.
_module = None


def get_module() -> ForestFormer3DModule:
    global _module
    if _module is None:
        _module = ForestFormer3DModule()
    return _module


def shutdown():
    """Stop the shared worker. Called from the app's closeEvent: the model runs
    in its own interpreter, so without this a run started seconds before
    quitting keeps the child (and ~6 GB of VRAM) alive after the window is gone.
    """
    global _module
    if _module is not None:
        _module.close()
        _module = None
