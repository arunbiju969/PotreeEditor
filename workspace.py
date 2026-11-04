"""Per-master editing workspace.

Opening ``tract.laz`` creates a sidecar folder ``tract.potreeedit/`` beside it,
so one dataset keeps everything in one place:

    tract.laz
    tract.potreeedit/
    ├── workspace.json     manifest: master, created, last commit, commit count
    ├── edits.db           edit history, split tree-ID routes, staged edits
    ├── segmented/         ForestFormer3D split clouds (tree_<itc>_ff3d_segmented.las)
    └── committed.laz      cumulative committed cloud — every commit is applied
                           on top of the previous one, so this single file is
                           "the edited point cloud"

The master LAS is never rewritten; ``committed.laz`` is its edited twin.
Uncommitted staged edits are snapshotted into ``edits.db`` so the sidebar
table is still populated the next time the master is opened.
"""
from __future__ import annotations

import json
import os
import time

from settings_store import SettingsStore

SUFFIX = ".potreeedit"


def workspace_path_for(master_path: str) -> str:
    """Sidecar folder path for a master LAS/LAZ (no filesystem side effects)."""
    stem, _ = os.path.splitext(os.path.abspath(master_path))
    return stem + SUFFIX


class Workspace:
    """Editing workspace that belongs to exactly one master point cloud."""

    def __init__(self, master_path: str):
        self.master = os.path.abspath(master_path)
        self.root = workspace_path_for(self.master)
        self.segmented_dir = os.path.join(self.root, "segmented")
        self.committed_laz = os.path.join(self.root, "committed.laz")
        self.db_path = os.path.join(self.root, "edits.db")
        self.manifest_path = os.path.join(self.root, "workspace.json")
        self._store: SettingsStore | None = None

    # ------------------------------------------------------------------ #
    def ensure(self) -> "Workspace":
        """Create the folder skeleton; safe to call on every open."""
        os.makedirs(self.segmented_dir, exist_ok=True)
        if not os.path.isfile(self.manifest_path):
            self._write_manifest({
                "master": self.master,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "commit_count": 0,
                "last_commit": None,
            })
        return self

    @property
    def store(self) -> SettingsStore:
        """SQLite store living inside the workspace (edit history, routes, staged)."""
        if self._store is None:
            self._store = SettingsStore(self.db_path)
        return self._store

    @property
    def has_committed(self) -> bool:
        return os.path.isfile(self.committed_laz)

    @property
    def commit_source(self) -> str:
        """File the next commit should read: the cumulative output if it exists."""
        return self.committed_laz if self.has_committed else self.master

    def close(self):
        if self._store is not None:
            self._store.close()
            self._store = None

    # ------------------------------------------------------------------ #
    # Staged (uncommitted) edits — restored when the master is reopened
    # ------------------------------------------------------------------ #
    def save_staged(self, edits: list):
        self.store.save_staged_edits(list(edits))

    def load_staged(self) -> list:
        return self.store.load_staged_edits()

    # ------------------------------------------------------------------ #
    # Manifest
    # ------------------------------------------------------------------ #
    def manifest(self) -> dict:
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def record_commit(self, output_path: str, edits_count: int = 0,
                      modified_points: int = 0, elapsed_seconds: float = 0.0,
                      potree_dir: str = None) -> dict:
        """Append a commit summary to workspace.json (history stays in edits.db)."""
        data = self.manifest()
        data["master"] = self.master
        data["commit_count"] = int(data.get("commit_count", 0)) + 1
        data["last_commit"] = {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "file": output_path,
            "edits": int(edits_count),
            "modified_points": int(modified_points),
            "elapsed_seconds": round(float(elapsed_seconds), 2),
            "potree_dir": potree_dir,
        }
        self._write_manifest(data)
        return data

    def _write_manifest(self, data: dict):
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, self.manifest_path)
