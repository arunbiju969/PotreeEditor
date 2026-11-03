"""
Point Cloud Data Manager for the Hybrid Viewer.
Handles loading LAS/LAZ files, spatial queries, optimistic visual staging,
attribute/classification/ITC modifications, and high-performance streaming commit.
"""

import copy
import functools
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable
import numpy as np
import laspy
from matplotlib.path import Path as MplPath


def _synchronized(method):
    """Serialize access to shared state across the GUI thread, HTTP handler
    threads and the commit QThread (all mutate the same manager)."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class PointCloudManager:
    def __init__(self):
        self._lock = threading.RLock()
        self.las_path: Optional[str] = None
        self.header_info: Optional[Dict[str, Any]] = None
        self.las_data: Optional[laspy.LasData] = None
        self.selected_indices: np.ndarray = np.array([], dtype=int)
        self.has_unsaved_changes: bool = False
        self.staged_edits: List[Dict[str, Any]] = []
        self.current_selection: Optional[Dict[str, Any]] = None
        self._tree_cache: Dict[int, Dict[str, Any]] = {}
        self._chunk_index: Optional[List[Dict[str, Any]]] = None
        self._tree_index: Optional[Dict[int, List[int]]] = None
        self._indexing_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._is_indexing: bool = False
        self._indexing_stage: str = "idle"
        self._last_indexing_event: Optional[Dict[str, Any]] = None

    def add_indexing_callback(self, cb: Callable[[Dict[str, Any]], None]):
        """Register a callback for automatic background indexing lifecycle and progress events."""
        if cb not in self._indexing_callbacks:
            self._indexing_callbacks.append(cb)
            if self._last_indexing_event:
                try:
                    cb(self._last_indexing_event)
                except Exception:
                    pass

    def _notify_indexing(self, event_type: str, **kwargs):
        """Dispatch indexing event payload to all registered callbacks."""
        payload = {"type": event_type, "is_indexing": self._is_indexing, **kwargs}
        self._last_indexing_event = payload
        for cb in list(self._indexing_callbacks):
            try:
                cb(payload)
            except Exception as e:
                print(f"[PointCloudManager] Error in indexing callback: {e}")

    def get_indexing_status(self) -> Dict[str, Any]:
        """Return the current background indexing status."""
        return {
            "is_indexing": getattr(self, "_is_indexing", False),
            "stage": getattr(self, "_indexing_stage", "idle"),
            "has_chunk_index": self._chunk_index is not None,
            "has_tree_index": self._tree_index is not None,
            "chunk_count": len(self._chunk_index) if self._chunk_index else 0,
            "tree_count": len(self._tree_index) if self._tree_index else 0,
            "last_event": getattr(self, "_last_indexing_event", None)
        }

    @property
    def xyz(self) -> Optional[np.ndarray]:
        """Backward-compatible on-demand coordinate array. Not stored permanently in RAM."""
        if self.las_data is None:
            return None
        return np.vstack((self.las_data.x, self.las_data.y, self.las_data.z)).T

    @_synchronized
    def load_las_header(self, path: str) -> Dict[str, Any]:
        """
        Instantly read LAS/LAZ header metadata in milliseconds without decompressing point records.
        Enables instant file opening (<0.1s) and low-memory footprint (~200MB) for massive point clouds.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        print(f"[PointCloudManager] Reading LAS header: {path}...")
        with laspy.open(path) as reader:
            hdr = reader.header
            total_points = hdr.point_count
            bounds = {
                "min": [float(hdr.x_min), float(hdr.y_min), float(hdr.z_min)],
                "max": [float(hdr.x_max), float(hdr.y_max), float(hdr.z_max)]
            }
            scales = [float(s) for s in hdr.scales]
            offsets = [float(o) for o in hdr.offsets]

            extra_dims = [dim.name for dim in hdr.point_format.extra_dimensions]
            has_itc = ("itc" in extra_dims) or ("treeID" in extra_dims) or ("tree_id" in extra_dims)

            info = {
                "path": path,
                "filename": os.path.basename(path),
                "total_points": total_points,
                "bounds": bounds,
                "scales": scales,
                "offsets": offsets,
                "has_itc": has_itc,
                "extra_dimensions": extra_dims,
                "dimensions": [d.name for d in hdr.point_format.dimensions],
                "is_header_only": True
            }

            self.las_path = path
            self.header_info = info
            self.las_data = None
            self.selected_indices = np.array([], dtype=int)
            self._tree_cache.clear()
            idx_file = path + ".chunk_idx.json"
            if os.path.exists(idx_file):
                try:
                    with open(idx_file, "r") as f:
                        self._chunk_index = json.load(f)
                    print(f"[PointCloudManager] Loaded chunk index with {len(self._chunk_index)} chunks.")
                except Exception as e:
                    print(f"[PointCloudManager] Error loading chunk index: {e}")
                    self._chunk_index = None
            else:
                self._chunk_index = None

            # Load exact tree-to-chunks index if available (0ms lookup of candidate chunks)
            tree_idx_file = path + ".tree_idx.json"
            if os.path.exists(tree_idx_file):
                try:
                    with open(tree_idx_file, "r") as f:
                        self._tree_index = {int(k): v for k, v in json.load(f).items()}
                    print(f"[PointCloudManager] Loaded exact tree index with {len(self._tree_index):,} trees.")
                except Exception as e:
                    print(f"[PointCloudManager] Error loading tree index: {e}")
                    self._tree_index = None
            else:
                self._tree_index = None

            # Check if sidecars are already ready or if background indexing is required
            if self._chunk_index is not None and (not has_itc or self._tree_index is not None):
                self._is_indexing = False
                self._indexing_stage = "ready"
                tree_cnt = len(self._tree_index) if self._tree_index else 0
                chunk_cnt = len(self._chunk_index)
                ready_msg = (
                    f"Tree spatial index ready: {tree_cnt:,} trees in {chunk_cnt} chunks."
                    if tree_cnt > 0 else f"Spatial chunk index ready ({chunk_cnt} chunks)."
                )
                self._notify_indexing("ready", stage="done", chunks=chunk_cnt, trees=tree_cnt, msg=ready_msg)
            elif self._chunk_index is None:
                self._is_indexing = True
                self._indexing_stage = "chunks"
                self._notify_indexing(
                    "started",
                    stage="chunks",
                    file=os.path.basename(path),
                    total_points=total_points,
                    msg="Automatic Background Indexing: Scanning spatial chunks (streaming fallback active)..."
                )
                threading.Thread(target=self._build_chunk_index_in_background, args=(path,), daemon=True).start()
            elif has_itc and self._tree_index is None:
                self._is_indexing = True
                self._indexing_stage = "trees"
                self._notify_indexing(
                    "started",
                    stage="trees",
                    file=os.path.basename(path),
                    chunks=len(self._chunk_index),
                    msg=f"Automatic Background Indexing: Indexing tree IDs across {len(self._chunk_index)} chunks..."
                )
                threading.Thread(target=self._build_tree_index_in_background, args=(path,), daemon=True).start()

            print(f"[PointCloudManager] Header read complete: {total_points:,} points. Has ITC: {has_itc}.")
            return info

    def _build_chunk_index_in_background(self, path: str):
        """Scans chunks in the background to build spatial bounding boxes and ITC extents."""
        try:
            chunks = []
            chunk_size = 5_000_000
            with laspy.open(path) as r:
                extra_dims = [d.name for d in r.header.point_format.dimensions]
                itc_name = next((n for n in ("itc", "treeID", "tree_id") if n in extra_dims), None)
                total_points = r.header.point_count
                total_chunks = max(1, math.ceil(total_points / chunk_size))

                self._notify_indexing(
                    "started",
                    stage="chunks",
                    file=os.path.basename(path),
                    total_points=total_points,
                    total_chunks=total_chunks,
                    msg=f"Automatic Background Indexing: Scanning spatial chunks ({total_chunks} chunks)..."
                )

                for i, chunk in enumerate(r.chunk_iterator(chunk_size)):
                    if self.las_path != path:
                        print(f"[PointCloudManager] Background indexing cancelled for {path} (active cloud changed)")
                        return
                    pt_off = i * chunk_size
                    cx_arr = np.asarray(chunk.x, dtype=np.float64)
                    cy_arr = np.asarray(chunk.y, dtype=np.float64)
                    c_info = {
                        "chunk_idx": i,
                        "point_offset": pt_off,
                        "count": len(chunk),
                        "min_x": float(cx_arr.min()),
                        "max_x": float(cx_arr.max()),
                        "min_y": float(cy_arr.min()),
                        "max_y": float(cy_arr.max()),
                    }
                    if itc_name:
                        tids = np.asarray(getattr(chunk, itc_name))
                        valid = tids[tids > 0]
                        c_info["min_itc"] = int(valid.min()) if len(valid) > 0 else 0
                        c_info["max_itc"] = int(valid.max()) if len(valid) > 0 else 0
                    chunks.append(c_info)

                    pct = int(round((i + 1) / total_chunks * 100))
                    self._notify_indexing(
                        "progress",
                        stage="chunks",
                        current=i + 1,
                        total=total_chunks,
                        percent=pct,
                        msg=f"Automatic Background Indexing: Spatial chunk {i + 1}/{total_chunks} ({pct}%)"
                    )

            if chunks:
                with self._lock:
                    self._chunk_index = chunks
                try:
                    with open(path + ".chunk_idx.json", "w") as f:
                        json.dump(chunks, f, indent=2)
                    print(f"[PointCloudManager] Built spatial chunk index ({len(chunks)} chunks).")
                except Exception as e:
                    print(f"[PointCloudManager] Error saving chunk index: {e}")

                if itc_name:
                    self._build_tree_index_in_background(path)
                else:
                    self._is_indexing = False
                    self._indexing_stage = "ready"
                    self._notify_indexing(
                        "completed",
                        stage="done",
                        chunks=len(chunks),
                        trees=0,
                        msg=f"Spatial chunk index complete ({len(chunks)} chunks). Stand queries are now instant!"
                    )
        except Exception as e:
            self._is_indexing = False
            self._indexing_stage = "error"
            self._notify_indexing("error", stage="error", error=str(e), msg=f"Background indexing warning: {e}")
            print(f"[PointCloudManager] Background chunk index building exception: {e}")

    def _build_tree_index_in_background(self, path: str):
        """Scans chunks in the background to build an exact tree_id -> chunk_idx index file."""
        try:
            if not self._chunk_index:
                return
            tree_to_chunks: Dict[int, List[int]] = {}
            total_chunks = len(self._chunk_index)
            self._notify_indexing(
                "started",
                stage="trees",
                file=os.path.basename(path),
                total=total_chunks,
                msg=f"Automatic Background Indexing: Indexing tree IDs across {total_chunks} chunks..."
            )
            with laspy.open(path) as r:
                extra_dims = [d.name for d in r.header.point_format.dimensions]
                itc_name = next((n for n in ("itc", "treeID", "tree_id") if n in extra_dims), None)
                if not itc_name:
                    self._is_indexing = False
                    self._indexing_stage = "ready"
                    return
                for idx, c in enumerate(self._chunk_index):
                    if self.las_path != path:
                        print(f"[PointCloudManager] Background tree indexing aborted for {path} (active cloud changed)")
                        return
                    r.seek(c["point_offset"])
                    pts = r.read_points(c["count"])
                    tids = np.asarray(getattr(pts, itc_name))
                    valid = tids[tids > 0]
                    if len(valid) > 0:
                        u_tids = np.unique(valid)
                        c_idx = c["chunk_idx"]
                        for tid in u_tids:
                            tid_int = int(tid)
                            if tid_int not in tree_to_chunks:
                                tree_to_chunks[tid_int] = []
                            tree_to_chunks[tid_int].append(c_idx)

                    pct = int(round((idx + 1) / total_chunks * 100))
                    self._notify_indexing(
                        "progress",
                        stage="trees",
                        current=idx + 1,
                        total=total_chunks,
                        percent=pct,
                        trees_found=len(tree_to_chunks),
                        msg=f"Automatic Background Indexing: Tree IDs in chunk {idx + 1}/{total_chunks} ({len(tree_to_chunks):,} trees)..."
                    )

            out_file = path + ".tree_idx.json"
            with open(out_file, "w") as f:
                json.dump(tree_to_chunks, f)
            with self._lock:
                self._tree_index = tree_to_chunks
            self._is_indexing = False
            self._indexing_stage = "ready"
            print(f"[PointCloudManager] Finished background tree index build: {len(tree_to_chunks):,} trees saved to {out_file}")
            self._notify_indexing(
                "completed",
                stage="done",
                chunks=len(self._chunk_index),
                trees=len(tree_to_chunks),
                msg=f"Tree spatial index ready: {len(tree_to_chunks):,} trees indexed in {len(self._chunk_index)} chunks. Stand queries are now instant!"
            )
        except Exception as e:
            self._is_indexing = False
            self._indexing_stage = "error"
            self._notify_indexing("error", stage="error", error=str(e), msg=f"Background tree index build warning: {e}")
            print(f"[PointCloudManager] Background tree index build failed: {e}")

    def prefetch_tree(self, itc_value: int, x: Optional[float] = None, y: Optional[float] = None):
        """Pre-warm tree cache in a background thread so queries return instantaneously."""
        itc_val = int(itc_value)
        if itc_val <= 0 or itc_val in self._tree_cache:
            return

        def _worker():
            try:
                self.get_tree_points(itc_val, x, y)
            except Exception:
                pass
        threading.Thread(target=_worker, daemon=True).start()

    @_synchronized
    def load_las(self, path: str) -> Dict[str, Any]:
        """Load a LAS/LAZ file fully into memory (optional for small clouds)."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        print(f"[PointCloudManager] Reading LAS file into RAM: {path}...")
        self.las_data = laspy.read(path)
        self.las_path = path
        self.selected_indices = np.array([], dtype=int)
        self.has_unsaved_changes = False
        self._tree_cache.clear()

        total_points = len(self.las_data)
        hdr = self.las_data.header
        bounds = {
            "min": [float(hdr.x_min), float(hdr.y_min), float(hdr.z_min)],
            "max": [float(hdr.x_max), float(hdr.y_max), float(hdr.z_max)]
        }

        extra_dims = [dim.name for dim in self.las_data.point_format.extra_dimensions]
        has_itc = "itc" in extra_dims or "treeID" in extra_dims or "tree_id" in extra_dims

        unique_classes = np.unique(self.las_data.classification).tolist() if hasattr(self.las_data, "classification") else []

        unique_itc = []
        if has_itc:
            itc_attr = "itc" if "itc" in extra_dims else ("treeID" if "treeID" in extra_dims else "tree_id")
            itc_vals = np.array(getattr(self.las_data, itc_attr))
            unique_itc = np.unique(itc_vals).tolist()

        info = {
            "path": path,
            "filename": os.path.basename(path),
            "total_points": total_points,
            "bounds": bounds,
            "has_itc": has_itc,
            "extra_dimensions": extra_dims,
            "dimensions": [d.name for d in self.las_data.point_format.dimensions],
            "classes": unique_classes,
            "unique_itc_count": len(unique_itc),
            "is_header_only": False
        }
        self.header_info = info
        self._is_indexing = False
        self._indexing_stage = "ready"
        self._notify_indexing(
            "ready",
            stage="memory",
            chunks=1,
            trees=len(unique_itc),
            msg=f"In-memory cloud ready: {len(unique_itc):,} trees ({total_points:,} points)."
        )
        print(f"[PointCloudManager] Loaded {total_points:,} points. Has ITC: {has_itc}. Classes: {unique_classes}")
        return info

    def _ensure_itc_dimension(self, default_val: int = 0):
        """Ensure the 'itc' extra dimension exists on the LasData object when in full memory mode."""
        if self.las_data is None:
            return

        extra_dims = [dim.name for dim in self.las_data.point_format.extra_dimensions]
        if "itc" in extra_dims:
            return

        print("[PointCloudManager] Adding 'itc' extra dimension to LAS file...")
        self.las_data.add_extra_dim(laspy.ExtraBytesParams(name="itc", type=np.int32))
        self.las_data.itc[:] = default_val

    def get_max_itc(self) -> int:
        """Find the maximum ITC/tree-id in the current point cloud."""
        if self.las_path:
            chunk_idx_path = f"{self.las_path}.chunk_idx.json"
            if os.path.exists(chunk_idx_path):
                try:
                    with open(chunk_idx_path, "r", encoding="utf-8") as f:
                        chunks = json.load(f)
                    return max(c.get("max_itc", 0) for c in chunks)
                except Exception:
                    pass
            tree_idx_path = f"{self.las_path}.tree_idx.json"
            if os.path.exists(tree_idx_path):
                try:
                    with open(tree_idx_path, "r", encoding="utf-8") as f:
                        tdata = json.load(f)
                    return max(int(k) for k in tdata.keys() if str(k).isdigit())
                except Exception:
                    pass
        if self.las_data is not None:
            extra_dims = [d.name for d in self.las_data.point_format.dimensions]
            itc_name = next((n for n in ("itc", "treeID", "tree_id") if n in extra_dims), None)
            if itc_name:
                arr = np.asarray(getattr(self.las_data, itc_name))
                return int(np.max(arr)) if len(arr) > 0 else 0
        return 10000

    # -------------------------------------------------------------------------
    # Optimistic Visual Staging System
    # -------------------------------------------------------------------------
    @_synchronized
    def add_staged_edit(self, edit: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage an edit operation (polygon crown delineation, noise classification, etc.).
        Edits are held in memory as lightweight geometric definitions and rendered visually in 3D.
        """
        if "id" not in edit or not edit["id"]:
            edit["id"] = f"edit_{len(self.staged_edits) + 1}_{int(time.time() * 1000) % 1000000}"

        action = edit.get("action", "assign_itc")
        if "color" not in edit or not edit["color"]:
            if action == "mark_noise":
                edit["color"] = "#ef4444"
            elif action == "assign_itc":
                edit["color"] = "#10b981"
            elif action == "ff3d_split":
                edit["color"] = "#06b6d4"
            elif action == "set_classification":
                edit["color"] = "#3b82f6"
            elif action == "delete":
                edit["color"] = "#f59e0b"
            else:
                edit["color"] = "#10b981"

        if "label" not in edit or not edit["label"]:
            if action == "assign_itc":
                edit["label"] = f"Tree #{edit.get('value', 1)}"
            elif action == "ff3d_split":
                edit["label"] = f"FF3D Split Tree #{edit.get('target_itc', '?')}"
            elif action == "mark_noise":
                edit["label"] = "Noise (7)"
            elif action == "set_classification":
                edit["label"] = f"Class {edit.get('value', 0)}"
            elif action == "delete":
                edit["label"] = "Deleted"
            else:
                edit["label"] = f"Edit #{len(self.staged_edits) + 1}"

        self.staged_edits.append(edit)
        self.has_unsaved_changes = True
        print(f"[PointCloudManager] Staged edit added: {edit['id']} -> {edit['label']} ({action})")
        return edit

    @_synchronized
    def remove_staged_edit(self, edit_id: str) -> bool:
        """Remove a staged edit by ID."""
        orig_len = len(self.staged_edits)
        self.staged_edits = [e for e in self.staged_edits if e.get("id") != edit_id]
        found = len(self.staged_edits) < orig_len
        self.has_unsaved_changes = bool(self.staged_edits)
        if found:
            print(f"[PointCloudManager] Staged edit removed: {edit_id}")
        return found

    @_synchronized
    def clear_staged_edits(self):
        """Clear all active staged edits."""
        self.staged_edits.clear()
        self.has_unsaved_changes = False
        print("[PointCloudManager] All staged edits cleared.")

    @_synchronized
    def get_staged_edits(self) -> List[Dict[str, Any]]:
        """Return a copy of all current staged edits."""
        return list(self.staged_edits)

    # -------------------------------------------------------------------------
    # High-Performance Deferred Commit (Streaming Chunks)
    # -------------------------------------------------------------------------
    @_synchronized
    def commit_staged_edits(
        self,
        input_path: str,
        output_path: str,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        chunk_size: int = 1_000_000
    ) -> Dict[str, Any]:
        """
        Commit all staged edits in a single streaming pass through the LAS/LAZ file.
        Uses raw coordinate integer bounding-box prefiltering and Matplotlib Path inside checks.
        Never loads the entire file into RAM, achieving minimal RAM footprint (<100MB)
        even for massive 200M+ point clouds.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input point cloud not found: {input_path}")

        # Snapshot the registry: staging/removal during the streaming pass must not
        # change which edits land in this commit.
        edits = self.get_staged_edits()
        if not edits:
            raise ValueError("No staged edits to commit.")

        t_start = time.time()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        print(f"[PointCloudManager] Committing {len(edits)} staged edits to: {output_path}...")

        with laspy.open(input_path) as reader:
            src_hdr = reader.header
            total_points = src_hdr.point_count

            # Determine if target header needs 'itc' added
            orig_extra_dims = [dim.name for dim in src_hdr.point_format.extra_dimensions]
            itc_fields = [f for f in ["itc", "treeID", "tree_id"] if f in orig_extra_dims]
            needs_add_itc = any(e.get("action") in ("assign_itc", "mark_noise", "delete") for e in edits) and (not itc_fields)

            out_hdr = src_hdr.copy()

            # Strip COPC VLRs to allow standard LAS/LAZ serialization
            if hasattr(out_hdr, "vlrs") and out_hdr.vlrs:
                out_hdr.vlrs = [v for v in out_hdr.vlrs if getattr(v, "user_id", "") != "copc"]

            if needs_add_itc:
                out_hdr.add_extra_dim(laspy.ExtraBytesParams(name="itc", type=np.int32))
                itc_fields = ["itc"]

            sx, sy = float(out_hdr.scales[0]), float(out_hdr.scales[1])
            ox, oy = float(out_hdr.offsets[0]), float(out_hdr.offsets[1])
            sz, oz = float(out_hdr.scales[2]), float(out_hdr.offsets[2])

            # Pre-compile spatial test structures for each staged edit
            compiled_edits = []
            for edit in edits:
                etype = edit.get("type", "polygon")
                if etype == "polygon" and "coords" in edit and len(edit["coords"]) >= 3:
                    pts = np.array(edit["coords"])
                    min_x, max_x = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
                    min_y, max_y = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
                    rx0 = int(np.floor((min_x - ox) / sx))
                    rx1 = int(np.ceil((max_x - ox) / sx))
                    ry0 = int(np.floor((min_y - oy) / sy))
                    ry1 = int(np.ceil((max_y - oy) / sy))
                    compiled_edits.append({
                        "type": "polygon",
                        "edit": edit,
                        "rx0": rx0, "rx1": rx1,
                        "ry0": ry0, "ry1": ry1,
                        "path": MplPath(pts[:, :2])
                    })
                elif etype == "box" and "min" in edit and "max" in edit:
                    min_b = edit["min"]
                    max_b = edit["max"]
                    rx0 = int(np.floor((min_b[0] - ox) / sx))
                    rx1 = int(np.ceil((max_b[0] - ox) / sx))
                    ry0 = int(np.floor((min_b[1] - oy) / sy))
                    ry1 = int(np.ceil((max_b[1] - oy) / sy))
                    rz0 = int(np.floor((min_b[2] - oz) / sz))
                    rz1 = int(np.ceil((max_b[2] - oz) / sz))
                    compiled_edits.append({
                        "type": "box",
                        "edit": edit,
                        "rx0": rx0, "rx1": rx1,
                        "ry0": ry0, "ry1": ry1,
                        "rz0": rz0, "rz1": rz1,
                    })
                elif etype == "ff3d_split" and "segmented_las" in edit and os.path.exists(edit["segmented_las"]):
                    from scipy.spatial import cKDTree
                    seg_data = laspy.read(edit["segmented_las"])
                    seg_xyz = np.column_stack([
                        np.asarray(seg_data.x, dtype=np.float64),
                        np.asarray(seg_data.y, dtype=np.float64),
                        np.asarray(seg_data.z, dtype=np.float64)
                    ])
                    extra_names = [d.name for d in seg_data.point_format.extra_dimensions]
                    if "ff3d_tree_id" in extra_names:
                        seg_tree_ids = np.asarray(seg_data.ff3d_tree_id, dtype=np.int32)
                    elif "tree_id" in extra_names:
                        seg_tree_ids = np.asarray(seg_data.tree_id, dtype=np.int32)
                    else:
                        seg_tree_ids = np.zeros(len(seg_data), dtype=np.int32)

                    id_mapping = edit.get("id_mapping", {})
                    mapped_itc = np.zeros(len(seg_data), dtype=np.int32)
                    for raw_id, target_id in id_mapping.items():
                        mapped_itc[seg_tree_ids == int(raw_id)] = int(target_id)

                    min_x, max_x = float(np.min(seg_xyz[:, 0])), float(np.max(seg_xyz[:, 0]))
                    min_y, max_y = float(np.min(seg_xyz[:, 1])), float(np.max(seg_xyz[:, 1]))
                    min_z, max_z = float(np.min(seg_xyz[:, 2])), float(np.max(seg_xyz[:, 2]))
                    rx0 = int(np.floor((min_x - ox) / sx))
                    rx1 = int(np.ceil((max_x - ox) / sx))
                    ry0 = int(np.floor((min_y - oy) / sy))
                    ry1 = int(np.ceil((max_y - oy) / sy))
                    rz0 = int(np.floor((min_z - oz) / sz))
                    rz1 = int(np.ceil((max_z - oz) / sz))

                    compiled_edits.append({
                        "type": "ff3d_split",
                        "edit": edit,
                        "rx0": rx0, "rx1": rx1,
                        "ry0": ry0, "ry1": ry1,
                        "rz0": rz0, "rz1": rz1,
                        "kdtree": cKDTree(seg_xyz),
                        "mapped_itc": mapped_itc,
                        "seg_tree_ids": seg_tree_ids
                    })

            # Stream chunks from reader to writer
            total_modified = 0
            processed_points = 0
            has_deletes = any(c_edit["edit"].get("action") == "delete" for c_edit in compiled_edits)

            with laspy.open(output_path, mode="w", header=out_hdr) as writer:
                for chunk in reader.chunk_iterator(chunk_size):
                    chunk_len = len(chunk)

                    if needs_add_itc:
                        work = laspy.PackedPointRecord.zeros(chunk_len, point_format=out_hdr.point_format)
                        for dname in chunk.point_format.dimension_names:
                            work[dname] = chunk[dname]
                        work["itc"] = np.zeros(chunk_len, dtype=np.int32)
                    else:
                        work = chunk

                    # 'touched' counts each point once even when several edits overlap it;
                    # 'keep_mask' accumulates deletions so one filtered write leaves the file.
                    touched = np.zeros(chunk_len, dtype=bool)
                    keep_mask = np.ones(chunk_len, dtype=bool) if has_deletes else None

                    # Apply compiled edits
                    for c_edit in compiled_edits:
                        edit = c_edit["edit"]
                        action = edit.get("action", "assign_itc")
                        val = edit.get("value", 0)

                        # Fast raw integer coordinate box filter (XY for polygons, +Z for boxes)
                        bmask = (
                            (work.X >= c_edit["rx0"]) & (work.X <= c_edit["rx1"]) &
                            (work.Y >= c_edit["ry0"]) & (work.Y <= c_edit["ry1"])
                        )
                        if c_edit["type"] == "box":
                            bmask &= (work.Z >= c_edit["rz0"]) & (work.Z <= c_edit["rz1"])
                            matched = np.where(bmask)[0]
                        elif c_edit["type"] == "ff3d_split":
                            bmask &= (work.Z >= c_edit["rz0"]) & (work.Z <= c_edit["rz1"])
                            cands = np.where(bmask)[0]
                            if len(cands) > 0:
                                cand_xyz = np.column_stack((work.X[cands] * sx + ox, work.Y[cands] * sy + oy, work.Z[cands] * sz + oz))
                                dists, indices = c_edit["kdtree"].query(cand_xyz, distance_upper_bound=0.03)
                                valid = dists < 0.03
                                if valid.any():
                                    cand_matched = cands[valid]
                                    matched_seg_idx = indices[valid]
                                    touched[cand_matched] = True

                                    new_itcs = c_edit["mapped_itc"][matched_seg_idx]
                                    for fld in itc_fields:
                                        work[fld][cand_matched] = new_itcs

                                    is_ground = (c_edit["seg_tree_ids"][matched_seg_idx] < 0) | (new_itcs == 0)
                                    if is_ground.any() and "classification" in work.point_format.dimension_names:
                                        work["classification"][cand_matched[is_ground]] = 2
                            continue
                        else:
                            cands = np.where(bmask)[0]
                            matched = cands
                            if len(cands) > 0:
                                cand_pts = np.column_stack((work.X[cands] * sx + ox, work.Y[cands] * sy + oy))
                                inside = c_edit["path"].contains_points(cand_pts)
                                matched = cands[inside]

                        if len(matched) == 0:
                            continue

                        touched[matched] = True
                        if action == "delete":
                            keep_mask[matched] = False
                        elif action == "assign_itc":
                            for fld in itc_fields:
                                work[fld][matched] = int(val)
                        elif action == "mark_noise":
                            for fld in itc_fields:
                                work[fld][matched] = int(val) if val is not None else 9999
                            work["classification"][matched] = 7
                        elif action == "set_classification":
                            work["classification"][matched] = int(val)

                    total_modified += int(touched.sum())

                    if keep_mask is not None:
                        work = work[keep_mask]

                    writer.write_points(work)
                    processed_points += chunk_len

                    if progress_cb:
                        pct = int((processed_points / total_points) * 100) if total_points > 0 else 0
                        msg = f"Writing points: {processed_points:,} / {total_points:,} ({pct}%)"
                        progress_cb(processed_points, total_points, msg)

        elapsed = time.time() - t_start
        # Edits are now part of the output file: empty the registry so a second
        # commit does not re-apply them on top of the committed results.
        self.staged_edits.clear()
        self.has_unsaved_changes = False
        print(f"[PointCloudManager] Commit finished in {elapsed:.2f}s. Points processed: {processed_points:,}. Points modified: {total_modified:,}.")

        return {
            "success": True,
            "output_path": output_path,
            "total_points": processed_points,
            "modified_points": total_modified,
            "edits_count": len(edits),
            "elapsed_seconds": elapsed
        }

    # -------------------------------------------------------------------------
    # Spatial Queries & In-Memory Fallbacks
    # -------------------------------------------------------------------------
    @_synchronized
    def select_by_polygon_2d(self, polygon_coords: List[List[float]]) -> Dict[str, Any]:
        """
        Record current selection polygon and query points inside 2D polygon (X, Y) if cloud is in memory.
        """
        self.current_selection = {
            "type": "polygon",
            "coords": polygon_coords
        }

        if self.las_data is None:
            # Streaming mode: return polygon metadata without requiring full RAM allocation
            return {
                "count": -1,
                "mode": "staged",
                "vertex_count": len(polygon_coords),
                "coords": polygon_coords,
                "classes": [],
                "itc_values": []
            }

        poly = np.array(polygon_coords)
        if len(poly) < 3:
            return {"count": 0, "indices": []}

        min_x, max_x = float(np.min(poly[:, 0])), float(np.max(poly[:, 0]))
        min_y, max_y = float(np.min(poly[:, 1])), float(np.max(poly[:, 1]))

        hdr = self.las_data.header
        sx, sy = hdr.scales[0], hdr.scales[1]
        ox, oy = hdr.offsets[0], hdr.offsets[1]

        rx0 = int(np.floor((min_x - ox) / sx))
        rx1 = int(np.ceil((max_x - ox) / sx))
        ry0 = int(np.floor((min_y - oy) / sy))
        ry1 = int(np.ceil((max_y - oy) / sy))

        # Fast pre-filtering directly on raw coordinates
        bbox_mask = (
            (self.las_data.X >= rx0) & (self.las_data.X <= rx1) &
            (self.las_data.Y >= ry0) & (self.las_data.Y <= ry1)
        )
        candidate_indices = np.where(bbox_mask)[0]

        if len(candidate_indices) == 0:
            self.selected_indices = np.array([], dtype=int)
            return {"count": 0, "indices": []}

        cand_x = self.las_data.X[candidate_indices] * sx + ox
        cand_y = self.las_data.Y[candidate_indices] * sy + oy
        candidate_pts = np.column_stack((cand_x, cand_y))

        path = MplPath(poly[:, :2])
        inside_mask = path.contains_points(candidate_pts)

        self.selected_indices = candidate_indices[inside_mask]
        count = len(self.selected_indices)

        classes_in_sel = []
        itc_in_sel = []
        if count > 0:
            if hasattr(self.las_data, "classification"):
                classes_in_sel = np.unique(self.las_data.classification[self.selected_indices]).tolist()
            if hasattr(self.las_data, "itc"):
                itc_in_sel = np.unique(self.las_data.itc[self.selected_indices]).tolist()

        return {
            "count": count,
            "classes": classes_in_sel,
            "itc_values": itc_in_sel
        }

    @_synchronized
    def select_by_box_3d(self, min_pt: List[float], max_pt: List[float]) -> Dict[str, Any]:
        """Record current 3D selection box."""
        self.current_selection = {
            "type": "box",
            "min": min_pt,
            "max": max_pt
        }

        if self.las_data is None:
            return {
                "count": -1,
                "mode": "staged",
                "min": min_pt,
                "max": max_pt
            }

        hdr = self.las_data.header
        sx, sy, sz = hdr.scales
        ox, oy, oz = hdr.offsets

        rx0 = int(np.floor((min_pt[0] - ox) / sx))
        rx1 = int(np.ceil((max_pt[0] - ox) / sx))
        ry0 = int(np.floor((min_pt[1] - oy) / sy))
        ry1 = int(np.ceil((max_pt[1] - oy) / sy))
        rz0 = int(np.floor((min_pt[2] - oz) / sz))
        rz1 = int(np.ceil((max_pt[2] - oz) / sz))

        bbox_mask = (
            (self.las_data.X >= rx0) & (self.las_data.X <= rx1) &
            (self.las_data.Y >= ry0) & (self.las_data.Y <= ry1) &
            (self.las_data.Z >= rz0) & (self.las_data.Z <= rz1)
        )
        candidates = np.where(bbox_mask)[0]

        if len(candidates) == 0:
            self.selected_indices = np.array([], dtype=int)
            return {"count": 0}

        cand_x = self.las_data.X[candidates] * sx + ox
        cand_y = self.las_data.Y[candidates] * sy + oy
        cand_z = self.las_data.Z[candidates] * sz + oz

        exact_mask = (
            (cand_x >= min_pt[0]) & (cand_x <= max_pt[0]) &
            (cand_y >= min_pt[1]) & (cand_y <= max_pt[1]) &
            (cand_z >= min_pt[2]) & (cand_z <= max_pt[2])
        )

        self.selected_indices = candidates[exact_mask]
        count = len(self.selected_indices)

        classes_in_sel = []
        itc_in_sel = []
        if count > 0:
            if hasattr(self.las_data, "classification"):
                classes_in_sel = np.unique(self.las_data.classification[self.selected_indices]).tolist()
            if hasattr(self.las_data, "itc"):
                itc_in_sel = np.unique(self.las_data.itc[self.selected_indices]).tolist()

        return {
            "count": count,
            "classes": classes_in_sel,
            "itc_values": itc_in_sel
        }

    @_synchronized
    def pick_point(self, x: float, y: float, z: float, max_dist: float = 2.0) -> Optional[Dict[str, Any]]:
        """Find the nearest point to (x, y, z) in the dataset if cloud is in memory."""
        if self.las_data is None:
            # Check if picked coordinate falls inside any staged edit
            staged_hit = None
            for edit in self.staged_edits:
                if edit.get("type") == "polygon" and "coords" in edit:
                    poly = np.array(edit["coords"])
                    mpath = MplPath(poly[:, :2])
                    if mpath.contains_point((x, y)):
                        staged_hit = edit
                        break
            return {
                "x": x, "y": y, "z": z,
                "staged_hit": staged_hit
            }

        hdr = self.las_data.header
        sx, sy, sz = hdr.scales
        ox, oy, oz = hdr.offsets

        rx0 = int(np.floor((x - max_dist - ox) / sx))
        rx1 = int(np.ceil((x + max_dist - ox) / sx))
        ry0 = int(np.floor((y - max_dist - oy) / sy))
        ry1 = int(np.ceil((y + max_dist - oy) / sy))
        rz0 = int(np.floor((z - max_dist - oz) / sz))
        rz1 = int(np.ceil((z + max_dist - oz) / sz))

        mask = (
            (self.las_data.X >= rx0) & (self.las_data.X <= rx1) &
            (self.las_data.Y >= ry0) & (self.las_data.Y <= ry1) &
            (self.las_data.Z >= rz0) & (self.las_data.Z <= rz1)
        )
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            return None

        c_x = self.las_data.X[candidates] * sx + ox
        c_y = self.las_data.Y[candidates] * sy + oy
        c_z = self.las_data.Z[candidates] * sz + oz
        dists = np.sqrt((c_x - x) ** 2 + (c_y - y) ** 2 + (c_z - z) ** 2)
        min_i = int(np.argmin(dists))
        dist = float(dists[min_i])
        if dist > max_dist:
            return None

        idx = int(candidates[min_i])
        classification = int(self.las_data.classification[idx]) if hasattr(self.las_data, "classification") else 0
        intensity = int(self.las_data.intensity[idx]) if hasattr(self.las_data, "intensity") else 0
        itc = int(self.las_data.itc[idx]) if hasattr(self.las_data, "itc") else None

        return {
            "index": idx,
            "distance": dist,
            "x": float(c_x[min_i]),
            "y": float(c_y[min_i]),
            "z": float(c_z[min_i]),
            "classification": classification,
            "intensity": intensity,
            "itc": itc
        }

    @_synchronized
    def assign_itc(self, itc_value: int) -> int:
        """Assign an ITC (tree ID) value to currently selected points in memory."""
        if self.las_data is None or len(self.selected_indices) == 0:
            return 0

        self._ensure_itc_dimension()
        self.las_data.itc[self.selected_indices] = int(itc_value)
        self.has_unsaved_changes = True
        return len(self.selected_indices)

    @_synchronized
    def set_classification(self, class_value: int) -> int:
        """Set ASPRS classification value for currently selected points in memory."""
        if self.las_data is None or len(self.selected_indices) == 0:
            return 0

        self.las_data.classification[self.selected_indices] = int(class_value)
        self.has_unsaved_changes = True
        return len(self.selected_indices)

    @_synchronized
    def mark_as_noise(self, noise_itc: int = 9999, class_val: int = 7) -> int:
        """Mark currently selected points as noise in memory."""
        if self.las_data is None or len(self.selected_indices) == 0:
            return 0

        self.set_classification(class_val)
        self.assign_itc(noise_itc)
        self.has_unsaved_changes = True
        return len(self.selected_indices)

    @_synchronized
    def delete_selected_points(self) -> int:
        """Delete currently selected points from the cloud in memory."""
        if self.las_data is None or len(self.selected_indices) == 0:
            return 0

        del_count = len(self.selected_indices)
        keep_mask = np.ones(len(self.las_data), dtype=bool)
        keep_mask[self.selected_indices] = False

        self.las_data.points = self.las_data.points[keep_mask]
        self.selected_indices = np.array([], dtype=int)
        self.has_unsaved_changes = True
        return del_count

    @_synchronized
    def get_tree_points(self, itc_value: int, x: Optional[float] = None, y: Optional[float] = None,
                        radius: float = 0.0, neighbor_mode: str = "trees_only",
                        ground_radius: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Retrieve ALL points whose ITC/tree-id equals `itc_value` (full density).
        Optionally retrieves surrounding neighbor points within `radius` meters
        and ground points (ASPRS class 2) within `ground_radius` meters.
        Works in both in-memory (small clouds) and streaming (200M+) modes.
        Uses spatial/chunk indexing and early-exit to respond in milliseconds.
        """
        itc_val = int(itc_value)
        rad_val = max(0.0, float(radius or 0.0))
        g_rad_val = max(0.0, float(ground_radius or 0.0))
        n_mode = str(neighbor_mode or "trees_only")
        cache_key = (itc_val, round(rad_val, 1), n_mode, round(g_rad_val, 1))

        if cache_key in self._tree_cache:
            return self._tree_cache[cache_key]
        if rad_val == 0.0 and g_rad_val == 0.0 and itc_val in self._tree_cache:
            return self._tree_cache[itc_val]

        if self.las_data is None and not self.las_path:
            return None

        rgb: Optional[np.ndarray] = None
        intensity: Optional[np.ndarray] = None
        classification: Optional[np.ndarray] = None
        target_count = 0
        neighbor_count = 0

        # In-memory fast path
        if self.las_data is not None:
            extra_dims = [d.name for d in self.las_data.point_format.dimensions]
            itc_name = next((n for n in ("itc", "treeID", "tree_id") if n in extra_dims), None)
            if itc_name is None:
                return None
            t_itc = np.asarray(getattr(self.las_data, itc_name))
            mask = t_itc == itc_val
            if not mask.any():
                res = {"count": 0, "itc": itc_val, "target_count": 0, "neighbor_count": 0}
                self._tree_cache[cache_key] = res
                return res

            target_count = int(mask.sum())
            t_x = np.asarray(self.las_data.x[mask], dtype=np.float64)
            t_y = np.asarray(self.las_data.y[mask], dtype=np.float64)
            t_z = np.asarray(self.las_data.z[mask], dtype=np.float64)
            t_cx, t_cy, t_cz = float(t_x.mean()), float(t_y.mean()), float(t_z.mean())

            has_rgb = "red" in extra_dims
            has_int = "intensity" in extra_dims
            has_cls = "classification" in extra_dims

            skip_dims = {"x", "y", "z", "X", "Y", "Z", "red", "green", "blue", "intensity", "classification"}
            scalar_names = [d for d in extra_dims if d not in skip_dims]
            scalars = {}

            all_x_raw = np.asarray(self.las_data.x, dtype=np.float64)
            all_y_raw = np.asarray(self.las_data.y, dtype=np.float64)
            dx = all_x_raw - t_cx
            dy = all_y_raw - t_cy

            n_mask = None
            if rad_val > 0.0:
                in_rad = (dx * dx + dy * dy) <= (rad_val * rad_val)
                not_target = (t_itc != itc_val)

                has_alt_tid = hasattr(self.las_data, "tree_id") or hasattr(self.las_data, "treeID")
                nc_tid = np.asarray(getattr(self.las_data, "tree_id", getattr(self.las_data, "treeID", t_itc)))
                is_tree = (t_itc > 0) | (nc_tid > 0)

                if n_mode == "all":
                    n_mask = in_rad & not_target
                elif n_mode == "trees_only":
                    n_mask = in_rad & not_target & is_tree
                    # Smart fallback: if no segmented trees are within radius, include surrounding canopy
                    if not n_mask.any() and has_cls:
                        n_mask = in_rad & not_target & (np.asarray(self.las_data.classification) == 5)
                    if not n_mask.any():
                        n_mask = in_rad & not_target & (np.asarray(self.las_data.z) >= t_cz - 8.0)
                else:  # "canopy" (trees + canopy)
                    is_canopy = is_tree
                    if has_cls:
                        is_canopy = is_canopy | (np.asarray(self.las_data.classification) == 5)
                    n_mask = in_rad & not_target & is_canopy
                    if not n_mask.any():
                        n_mask = in_rad & not_target & (np.asarray(self.las_data.z) >= t_cz - 8.0)

            # Ground disc under the tree: ASPRS class 2 points within ground_radius
            g_mask = None
            g_count = 0
            if g_rad_val > 0.0:
                g_in = (dx * dx + dy * dy) <= (g_rad_val * g_rad_val)
                if has_cls:
                    g_mask = g_in & (np.asarray(self.las_data.classification) == 2)
                else:
                    g_mask = g_in & (np.asarray(self.las_data.z, dtype=np.float64) <= t_cz - 2.0)
                g_mask = g_mask & ~mask
                if n_mask is not None:
                    g_mask = g_mask & ~n_mask
                g_count = int(g_mask.sum())

            neighbor_count = (int(n_mask.sum()) if n_mask is not None else 0) + g_count

            if (neighbor_count > 0) or g_count > 0:
                # Keep target points first, then neighbors, then ground
                idx_target = np.flatnonzero(mask)
                parts = [idx_target]
                if n_mask is not None and int(n_mask.sum()) > 0:
                    parts.append(np.flatnonzero(n_mask))
                if g_count > 0:
                    parts.append(np.flatnonzero(g_mask))
                combined_idx = np.concatenate(parts)
                xyz_f64 = np.column_stack((all_x_raw[combined_idx], all_y_raw[combined_idx], np.asarray(self.las_data.z[combined_idx], dtype=np.float64)))
                if has_rgb:
                    r_raw = (np.asarray(self.las_data.red[combined_idx]) >> 8).astype(np.uint8)
                    g_raw = (np.asarray(self.las_data.green[combined_idx]) >> 8).astype(np.uint8)
                    b_raw = (np.asarray(self.las_data.blue[combined_idx]) >> 8).astype(np.uint8)
                    rgb = np.column_stack((r_raw, g_raw, b_raw))
                if has_int:
                    intensity = np.asarray(self.las_data.intensity[combined_idx], dtype=np.uint16)
                if has_cls:
                    classification = np.asarray(self.las_data.classification[combined_idx], dtype=np.uint8)
                for sname in scalar_names:
                    try:
                        scalars[sname] = np.asarray(getattr(self.las_data, sname)[combined_idx], dtype=np.float32)
                    except Exception:
                        pass
            else:
                xyz_f64 = np.column_stack((t_x, t_y, t_z))
                if has_rgb:
                    r_raw = (np.asarray(self.las_data.red[mask]) >> 8).astype(np.uint8)
                    g_raw = (np.asarray(self.las_data.green[mask]) >> 8).astype(np.uint8)
                    b_raw = (np.asarray(self.las_data.blue[mask]) >> 8).astype(np.uint8)
                    rgb = np.column_stack((r_raw, g_raw, b_raw))
                if has_int:
                    intensity = np.asarray(self.las_data.intensity[mask], dtype=np.uint16)
                if has_cls:
                    classification = np.asarray(self.las_data.classification[mask], dtype=np.uint8)
                for sname in scalar_names:
                    try:
                        scalars[sname] = np.asarray(getattr(self.las_data, sname)[mask], dtype=np.float32)
                    except Exception:
                        pass
            centroid = np.array([t_cx, t_cy, t_cz], dtype=np.float64)

        # Streaming path (large clouds)
        else:
            with laspy.open(self.las_path) as reader:
                hdr = reader.header
                extra_dims = [d.name for d in hdr.point_format.dimensions]
                itc_name = next((n for n in ("itc", "treeID", "tree_id") if n in extra_dims), None)
                if itc_name is None:
                    return None
                has_rgb = "red" in extra_dims
                has_int = "intensity" in extra_dims
                has_cls = "classification" in extra_dims

                skip_dims = {"x", "y", "z", "X", "Y", "Z", "red", "green", "blue", "intensity", "classification"}
                scalar_names = [d for d in extra_dims if d not in skip_dims]
                scalar_chunks = {sname: [] for sname in scalar_names}

                xs, ys, zs, rs, ints, classes = [], [], [], [], [], []

                # Ultra-fast indexed seeking
                if self._tree_index is not None:
                    target_chunk_ids = self._tree_index.get(itc_val)
                    if not target_chunk_ids:
                        res = {"count": 0, "itc": itc_val, "target_count": 0, "neighbor_count": 0}
                        self._tree_cache[cache_key] = res
                        return res
                    if self._chunk_index:
                        candidates = [c for c in self._chunk_index if c["chunk_idx"] in target_chunk_ids]
                    else:
                        candidates = []
                elif self._chunk_index:
                    candidates = [c for c in self._chunk_index if c["min_itc"] <= itc_val <= c["max_itc"]]
                    if x is not None and y is not None:
                        spat = [c for c in candidates if (c["min_x"] - 35 <= x <= c["max_x"] + 35 and
                                                          c["min_y"] - 35 <= y <= c["max_y"] + 35)]
                        if spat:
                            candidates = spat

                if self._chunk_index or self._tree_index is not None:
                    for c in candidates:
                        reader.seek(c["point_offset"])
                        chunk = reader.read_points(c["count"])
                        c_mask = np.asarray(getattr(chunk, itc_name)) == itc_val
                        if not c_mask.any():
                            continue
                        xs.append(np.asarray(chunk.x[c_mask], dtype=np.float64))
                        ys.append(np.asarray(chunk.y[c_mask], dtype=np.float64))
                        zs.append(np.asarray(chunk.z[c_mask], dtype=np.float64))
                        if has_rgb:
                            rr = (np.asarray(chunk.red[c_mask]) >> 8).astype(np.uint8)
                            gg = (np.asarray(chunk.green[c_mask]) >> 8).astype(np.uint8)
                            bb = (np.asarray(chunk.blue[c_mask]) >> 8).astype(np.uint8)
                            rs.append(np.column_stack((rr, gg, bb)))
                        if has_int:
                            ints.append(np.asarray(chunk.intensity[c_mask], dtype=np.uint16))
                        if has_cls:
                            classes.append(np.asarray(chunk.classification[c_mask], dtype=np.uint8))
                        for sname in scalar_names:
                            try:
                                scalar_chunks[sname].append(np.asarray(getattr(chunk, sname)[c_mask], dtype=np.float32))
                            except Exception:
                                pass
                else:
                    consecutive_empty = 0
                    for chunk in reader.chunk_iterator(5_000_000):
                        c_mask = np.asarray(getattr(chunk, itc_name)) == itc_val
                        if not c_mask.any():
                            if xs:
                                consecutive_empty += 1
                                if consecutive_empty >= 2:
                                    break
                            continue
                        consecutive_empty = 0
                        xs.append(np.asarray(chunk.x[c_mask], dtype=np.float64))
                        ys.append(np.asarray(chunk.y[c_mask], dtype=np.float64))
                        zs.append(np.asarray(chunk.z[c_mask], dtype=np.float64))
                        if has_rgb:
                            rr = (np.asarray(chunk.red[c_mask]) >> 8).astype(np.uint8)
                            gg = (np.asarray(chunk.green[c_mask]) >> 8).astype(np.uint8)
                            bb = (np.asarray(chunk.blue[c_mask]) >> 8).astype(np.uint8)
                            rs.append(np.column_stack((rr, gg, bb)))
                        if has_int:
                            ints.append(np.asarray(chunk.intensity[c_mask], dtype=np.uint16))
                        if has_cls:
                            classes.append(np.asarray(chunk.classification[c_mask], dtype=np.uint8))
                        for sname in scalar_names:
                            try:
                                scalar_chunks[sname].append(np.asarray(getattr(chunk, sname)[c_mask], dtype=np.float32))
                            except Exception:
                                pass

                if not xs:
                    res = {"count": 0, "itc": itc_val, "target_count": 0, "neighbor_count": 0}
                    self._tree_cache[cache_key] = res
                    return res

                t_x_arr = np.concatenate(xs)
                t_y_arr = np.concatenate(ys)
                t_z_arr = np.concatenate(zs)
                t_cx, t_cy, t_cz = float(t_x_arr.mean()), float(t_y_arr.mean()), float(t_z_arr.mean())
                centroid = np.array([t_cx, t_cy, t_cz], dtype=np.float64)
                target_count = len(t_x_arr)

                # Extract surrounding neighbors within radius if requested
                n_xs, n_ys, n_zs, n_rs, n_ints, n_classes = [], [], [], [], [], []
                n_scalar_chunks = {sname: [] for sname in scalar_names}
                neighbor_count = 0

                scan_rad = max(rad_val, g_rad_val)
                if scan_rad > 0.0:
                    rad_sq = rad_val * rad_val
                    g_sq = g_rad_val * g_rad_val

                    def _extract_from_chunk(n_chunk):
                        nc_x = np.asarray(n_chunk.x, dtype=np.float64)
                        nc_y = np.asarray(n_chunk.y, dtype=np.float64)
                        dx = nc_x - t_cx
                        dy = nc_y - t_cy
                        d2 = dx * dx + dy * dy
                        in_scan = d2 <= (scan_rad * scan_rad)
                        if not in_scan.any():
                            return

                        nc_itc = np.asarray(getattr(n_chunk, itc_name))
                        not_target = (nc_itc != itc_val)

                        nc_tid = np.asarray(getattr(n_chunk, "tree_id", getattr(n_chunk, "treeID", nc_itc)))
                        is_tree = (nc_itc > 0) | (nc_tid > 0)

                        n_mask = np.zeros(nc_itc.shape, dtype=bool)
                        if rad_val > 0.0:
                            in_rad = d2 <= rad_sq
                            if n_mode == "all":
                                n_mask = in_rad & not_target
                            elif n_mode == "trees_only":
                                n_mask = in_rad & not_target & is_tree
                                if not n_mask.any() and has_cls:
                                    n_mask = in_rad & not_target & (np.asarray(n_chunk.classification) == 5)
                                if not n_mask.any():
                                    n_mask = in_rad & not_target & (np.asarray(n_chunk.z) >= t_cz - 8.0)
                            else:  # "canopy"
                                is_canopy = is_tree
                                if has_cls:
                                    is_canopy = is_canopy | (np.asarray(n_chunk.classification) == 5)
                                n_mask = in_rad & not_target & is_canopy
                                if not n_mask.any():
                                    n_mask = in_rad & not_target & (np.asarray(n_chunk.z) >= t_cz - 8.0)

                        # Ground disc under the tree: ASPRS class 2 within ground_radius
                        if g_rad_val > 0.0:
                            g_m = d2 <= g_sq
                            if has_cls:
                                g_m = g_m & (np.asarray(n_chunk.classification) == 2)
                            else:
                                g_m = g_m & (np.asarray(n_chunk.z, dtype=np.float64) <= t_cz - 2.0)
                            n_mask = n_mask | (g_m & not_target & ~n_mask)

                        if not n_mask.any():
                            return

                        n_xs.append(nc_x[n_mask])
                        n_ys.append(nc_y[n_mask])
                        n_zs.append(np.asarray(n_chunk.z[n_mask], dtype=np.float64))
                        if has_rgb:
                            rr = (np.asarray(n_chunk.red[n_mask]) >> 8).astype(np.uint8)
                            gg = (np.asarray(n_chunk.green[n_mask]) >> 8).astype(np.uint8)
                            bb = (np.asarray(n_chunk.blue[n_mask]) >> 8).astype(np.uint8)
                            n_rs.append(np.column_stack((rr, gg, bb)))
                        if has_int:
                            n_ints.append(np.asarray(n_chunk.intensity[n_mask], dtype=np.uint16))
                        if has_cls:
                            n_classes.append(np.asarray(n_chunk.classification[n_mask], dtype=np.uint8))
                        for sname in scalar_names:
                            try:
                                n_scalar_chunks[sname].append(np.asarray(getattr(n_chunk, sname)[n_mask], dtype=np.float32))
                            except Exception:
                                pass

                    if self._chunk_index:
                        neigh_chunks = [
                            c for c in self._chunk_index
                            if not (c['max_x'] < t_cx - scan_rad or c['min_x'] > t_cx + scan_rad or
                                    c['max_y'] < t_cy - scan_rad or c['min_y'] > t_cy + scan_rad)
                        ]
                        for c in neigh_chunks:
                            reader.seek(c["point_offset"])
                            n_chunk = reader.read_points(c["count"])
                            _extract_from_chunk(n_chunk)
                    else:
                        reader.seek(0)
                        for n_chunk in reader.chunk_iterator(5_000_000):
                            nc_x_check = np.asarray(n_chunk.x, dtype=np.float64)
                            nc_y_check = np.asarray(n_chunk.y, dtype=np.float64)
                            if (nc_x_check.max() < t_cx - scan_rad or nc_x_check.min() > t_cx + scan_rad or
                                nc_y_check.max() < t_cy - scan_rad or nc_y_check.min() > t_cy + scan_rad):
                                continue
                            _extract_from_chunk(n_chunk)

                    if n_xs:
                        neighbor_count = sum(len(nx) for nx in n_xs)

                if neighbor_count > 0:
                    all_x = np.concatenate([t_x_arr, np.concatenate(n_xs)])
                    all_y = np.concatenate([t_y_arr, np.concatenate(n_ys)])
                    all_z = np.concatenate([t_z_arr, np.concatenate(n_zs)])
                    xyz_f64 = np.column_stack((all_x, all_y, all_z))
                    if has_rgb:
                        rgb = np.vstack([np.concatenate(rs), np.concatenate(n_rs)])
                    if has_int:
                        intensity = np.concatenate([np.concatenate(ints), np.concatenate(n_ints)])
                    if has_cls and classes:
                        classification = np.concatenate([np.concatenate(classes), np.concatenate(n_classes)])
                    scalars = {}
                    for sname in scalar_names:
                        t_sc = np.concatenate(scalar_chunks[sname]) if scalar_chunks[sname] else np.zeros(target_count, dtype=np.float32)
                        n_sc = np.concatenate(n_scalar_chunks[sname]) if n_scalar_chunks[sname] else np.zeros(neighbor_count, dtype=np.float32)
                        scalars[sname] = np.concatenate([t_sc, n_sc]).astype(np.float32)
                else:
                    xyz_f64 = np.column_stack((t_x_arr, t_y_arr, t_z_arr))
                    if has_rgb:
                        rgb = np.concatenate(rs)
                    if has_int:
                        intensity = np.concatenate(ints)
                    if has_cls and classes:
                        classification = np.concatenate(classes)
                    scalars = {}
                    for sname, chs in scalar_chunks.items():
                        if chs:
                            scalars[sname] = np.concatenate(chs).astype(np.float32)

        count = int(xyz_f64.shape[0])
        # Coordinates are centered relative to the target tree centroid so it remains dead-center in Three.js
        centered = (xyz_f64 - centroid).astype(np.float32)
        lo = centered.min(axis=0)
        hi = centered.max(axis=0)

        res = {
            "itc": itc_val,
            "count": count,
            "target_count": target_count if target_count > 0 else count,
            "neighbor_count": neighbor_count,
            "radius": rad_val,
            "neighbor_mode": n_mode,
            "indexing_active": getattr(self, "_is_indexing", False),
            "centroid": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
            "xyz": np.ascontiguousarray(centered, dtype=np.float32),
            "rgb": np.ascontiguousarray(rgb, dtype=np.uint8) if rgb is not None else None,
            "intensity": np.ascontiguousarray(intensity, dtype=np.uint16) if intensity is not None else None,
            "classification": np.ascontiguousarray(classification, dtype=np.uint8) if classification is not None else None,
            "scalars": scalars,
            "bbox_min": [float(lo[0]), float(lo[1]), float(lo[2])],
            "bbox_max": [float(hi[0]), float(hi[1]), float(hi[2])]
        }
        self._tree_cache[cache_key] = res
        if rad_val == 0.0 and g_rad_val == 0.0:
            self._tree_cache[itc_val] = res
        return res

    @_synchronized
    def export_tree_points(self, output_path: str, itc_value: int,
                           x: Optional[float] = None, y: Optional[float] = None,
                           radius: float = 0.0, neighbor_mode: str = "canopy",
                           keep_tree_ids: Optional[List[int]] = None,
                           ground_radius: float = 0.0) -> int:
        """Extract the full-density tree/stand currently displayed in the
        Forestry tab from the LAS master and write it to a LAS/LAZ file.

        Reuses `get_tree_points` (same query the WebGL viewer hit), so the
        export carries every attribute at full density. When `keep_tree_ids`
        is provided (an active Neighbor Stand Filter), only points whose ITC
        is in that set are written. Returns the number of points exported.
        """
        res = self.get_tree_points(itc_value, x=x, y=y, radius=radius,
                                   neighbor_mode=neighbor_mode,
                                   ground_radius=ground_radius)
        if not res or res.get("count", 0) == 0:
            return 0

        centroid = np.asarray(res["centroid"], dtype=np.float64)
        xyz = np.asarray(res["xyz"], dtype=np.float64) + centroid  # centered -> master CRS
        rgb = res.get("rgb")
        intensity = res.get("intensity")
        classification = res.get("classification")
        scalars = res.get("scalars") or {}

        # Apply the active stand filter: keep only the checked trees
        if keep_tree_ids:
            keep = np.asarray(sorted({int(t) for t in keep_tree_ids}))
            itc_name = next((n for n in ("itc", "treeID", "tree_id") if n in scalars), None)
            if itc_name is not None:
                sel = np.isin(np.asarray(scalars[itc_name]), keep)
                if not sel.any():
                    return 0
                xyz = xyz[sel]
                if rgb is not None:
                    rgb = np.asarray(rgb)[sel]
                if intensity is not None:
                    intensity = np.asarray(intensity)[sel]
                if classification is not None:
                    classification = np.asarray(classification)[sel]
                scalars = {k: np.asarray(v)[sel] for k, v in scalars.items()}

        # Rebuild on the master header to preserve point format, scale & offset
        if self.las_data is not None:
            header = self.las_data.header
        elif self.las_path:
            with laspy.open(self.las_path) as reader:
                header = reader.header
        else:
            return 0

        # The exported subset has its own point order, so the master's COPC
        # octree VLRs (offsets/counts of the full cloud) no longer apply
        out = laspy.LasData(copy.deepcopy(header))
        out.header.vlrs = [v for v in out.header.vlrs if getattr(v, "user_id", "") != "copc"]
        if hasattr(out.header, "evlrs") and out.header.evlrs:
            out.header.evlrs = [v for v in out.header.evlrs if getattr(v, "user_id", "") != "copc"]
        n_out = int(xyz.shape[0])
        out.points.resize(n_out)
        out.x = xyz[:, 0]
        out.y = xyz[:, 1]
        out.z = xyz[:, 2]
        if rgb is not None:
            rgb16 = (np.asarray(rgb, dtype=np.uint16) << 8)  # 8-bit back to 16-bit LAS range
            out.red = rgb16[:, 0]
            out.green = rgb16[:, 1]
            out.blue = rgb16[:, 2]
        if intensity is not None:
            out.intensity = np.asarray(intensity, dtype=np.uint16)
        if classification is not None:
            out.classification = np.asarray(classification, dtype=np.uint8)
        for name, values in scalars.items():
            try:
                dim = header.point_format.dimension_by_name(name)
                setattr(out, name, np.asarray(values, dtype=dim.dtype))
            except Exception:
                pass

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        print(f"[PointCloudManager] Extracting tree #{res.get('itc')} stand -> {output_path}...")
        if output_path.lower().endswith(".copc.laz"):
            # COPC = LAZ 1.4 + octree index VLRs; attributes are untouched, so
            # this is the same lossless point data, ready for streaming viewers
            import tempfile
            from copc_writer import write_copc
            fd, tmp_laz = tempfile.mkstemp(suffix=".laz", prefix="tree_extract_")
            os.close(fd)
            try:
                out.write(tmp_laz)
                write_copc(Path(tmp_laz), Path(output_path))
            finally:
                try:
                    os.remove(tmp_laz)
                except OSError:
                    pass
        else:
            out.write(output_path)
        return int(xyz.shape[0])

    @_synchronized
    def save_las(self, output_path: str):
        """Save the in-memory point cloud to disk."""
        if self.las_data is None:
            raise ValueError("No point cloud loaded to save.")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        print(f"[PointCloudManager] Writing updated LAS/LAZ to: {output_path}...")

        if hasattr(self.las_data.header, "vlrs") and self.las_data.header.vlrs:
            self.las_data.header.vlrs = [v for v in self.las_data.header.vlrs if getattr(v, "user_id", "") != "copc"]
        if hasattr(self.las_data.header, "evlrs") and self.las_data.header.evlrs:
            self.las_data.header.evlrs = [v for v in self.las_data.header.evlrs if getattr(v, "user_id", "") != "copc"]
        if hasattr(self.las_data, "evlrs") and self.las_data.evlrs:
            self.las_data.evlrs = [v for v in self.las_data.evlrs if getattr(v, "user_id", "") != "copc"]

        self.las_data.write(output_path)
        self.has_unsaved_changes = False
        print("[PointCloudManager] Save complete.")
