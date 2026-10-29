"""
Point Cloud Data Manager for the Hybrid Viewer.
Handles loading LAS/LAZ files, spatial queries, optimistic visual staging,
attribute/classification/ITC modifications, and high-performance streaming commit.
"""

import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable
import numpy as np
import laspy
from matplotlib.path import Path as MplPath


class PointCloudManager:
    def __init__(self):
        self.las_path: Optional[str] = None
        self.header_info: Optional[Dict[str, Any]] = None
        self.las_data: Optional[laspy.LasData] = None
        self.selected_indices: np.ndarray = np.array([], dtype=int)
        self.has_unsaved_changes: bool = False
        self.staged_edits: List[Dict[str, Any]] = []
        self.current_selection: Optional[Dict[str, Any]] = None

    @property
    def xyz(self) -> Optional[np.ndarray]:
        """Backward-compatible on-demand coordinate array. Not stored permanently in RAM."""
        if self.las_data is None:
            return None
        return np.vstack((self.las_data.x, self.las_data.y, self.las_data.z)).T

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
                "is_header_only": True
            }

            self.las_path = path
            self.header_info = info
            self.las_data = None
            self.selected_indices = np.array([], dtype=int)
            print(f"[PointCloudManager] Header read complete: {total_points:,} points. Has ITC: {has_itc}.")
            return info

    def load_las(self, path: str) -> Dict[str, Any]:
        """Load a LAS/LAZ file fully into memory (optional for small clouds)."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        print(f"[PointCloudManager] Reading LAS file into RAM: {path}...")
        self.las_data = laspy.read(path)
        self.las_path = path
        self.selected_indices = np.array([], dtype=int)
        self.has_unsaved_changes = False

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
            "classes": unique_classes,
            "unique_itc_count": len(unique_itc),
            "is_header_only": False
        }
        self.header_info = info
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

    # -------------------------------------------------------------------------
    # Optimistic Visual Staging System
    # -------------------------------------------------------------------------
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
            elif action == "set_classification":
                edit["color"] = "#3b82f6"
            elif action == "delete":
                edit["color"] = "#f59e0b"
            else:
                edit["color"] = "#10b981"

        if "label" not in edit or not edit["label"]:
            if action == "assign_itc":
                edit["label"] = f"Tree #{edit.get('value', 1)}"
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

    def remove_staged_edit(self, edit_id: str) -> bool:
        """Remove a staged edit by ID."""
        orig_len = len(self.staged_edits)
        self.staged_edits = [e for e in self.staged_edits if e.get("id") != edit_id]
        found = len(self.staged_edits) < orig_len
        self.has_unsaved_changes = bool(self.staged_edits)
        if found:
            print(f"[PointCloudManager] Staged edit removed: {edit_id}")
        return found

    def clear_staged_edits(self):
        """Clear all active staged edits."""
        self.staged_edits.clear()
        self.has_unsaved_changes = False
        print("[PointCloudManager] All staged edits cleared.")

    def get_staged_edits(self) -> List[Dict[str, Any]]:
        """Return a copy of all current staged edits."""
        return list(self.staged_edits)

    # -------------------------------------------------------------------------
    # High-Performance Deferred Commit (Streaming Chunks)
    # -------------------------------------------------------------------------
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

        if not self.staged_edits:
            raise ValueError("No staged edits to commit.")

        t_start = time.time()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        print(f"[PointCloudManager] Committing {len(self.staged_edits)} staged edits to: {output_path}...")

        with laspy.open(input_path) as reader:
            src_hdr = reader.header
            total_points = src_hdr.point_count

            # Determine if target header needs 'itc' added
            orig_extra_dims = [dim.name for dim in src_hdr.point_format.extra_dimensions]
            itc_fields = [f for f in ["itc", "treeID", "tree_id"] if f in orig_extra_dims]
            needs_add_itc = any(e.get("action") in ("assign_itc", "mark_noise") for e in self.staged_edits) and (not itc_fields)

            out_hdr = src_hdr.copy()

            # Strip COPC VLRs to allow standard LAS/LAZ serialization
            if hasattr(out_hdr, "vlrs") and out_hdr.vlrs:
                out_hdr.vlrs = [v for v in out_hdr.vlrs if getattr(v, "user_id", "") != "copc"]

            if needs_add_itc:
                out_hdr.add_extra_dim(laspy.ExtraBytesParams(name="itc", type=np.int32))
                itc_fields = ["itc"]
                out_hdr.vlrs = [v for v in out_hdr.vlrs if getattr(v, "user_id", "") != "copc"]

            if needs_add_itc:
                out_hdr.add_extra_dim(laspy.ExtraBytesParams(name="itc", type=np.int32))
                target_itc_field = "itc"

            sx, sy = float(out_hdr.scales[0]), float(out_hdr.scales[1])
            ox, oy = float(out_hdr.offsets[0]), float(out_hdr.offsets[1])
            sz, oz = float(out_hdr.scales[2]), float(out_hdr.offsets[2])

            # Pre-compile spatial test structures for each staged edit
            compiled_edits = []
            for edit in self.staged_edits:
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

            # Stream chunks from reader to writer
            total_modified = 0
            processed_points = 0

            with laspy.open(output_path, mode="w", header=out_hdr) as writer:
                for chunk_idx, chunk in enumerate(reader.chunk_iterator(chunk_size)):
                    chunk_len = len(chunk)

                    if needs_add_itc:
                        work = laspy.PackedPointRecord.zeros(chunk_len, point_format=out_hdr.point_format)
                        for dname in chunk.point_format.dimension_names:
                            work[dname] = chunk[dname]
                        work["itc"] = np.zeros(chunk_len, dtype=np.int32)
                    else:
                        work = chunk

                    # Apply compiled edits
                    for c_edit in compiled_edits:
                        edit = c_edit["edit"]
                        action = edit.get("action", "assign_itc")
                        val = edit.get("value", 0)

                        if c_edit["type"] == "polygon":
                            # Fast raw integer coordinate box filter
                            bmask = (
                                (work.X >= c_edit["rx0"]) & (work.X <= c_edit["rx1"]) &
                                (work.Y >= c_edit["ry0"]) & (work.Y <= c_edit["ry1"])
                            )
                            cands = np.where(bmask)[0]
                            if len(cands) > 0:
                                cand_pts = np.column_stack((work.X[cands] * sx + ox, work.Y[cands] * sy + oy))
                                inside = c_edit["path"].contains_points(cand_pts)
                                matched = cands[inside]
                                if len(matched) > 0:
                                    total_modified += len(matched)
                                    if action == "assign_itc":
                                        for fld in itc_fields:
                                            work[fld][matched] = int(val)
                                    elif action == "mark_noise":
                                        for fld in itc_fields:
                                            work[fld][matched] = int(val) if val else 9999
                                        work["classification"][matched] = 7
                                    elif action == "set_classification":
                                        work["classification"][matched] = int(val)

                        elif c_edit["type"] == "box":
                            bmask = (
                                (work.X >= c_edit["rx0"]) & (work.X <= c_edit["rx1"]) &
                                (work.Y >= c_edit["ry0"]) & (work.Y <= c_edit["ry1"]) &
                                (work.Z >= c_edit["rz0"]) & (work.Z <= c_edit["rz1"])
                            )
                            matched = np.where(bmask)[0]
                            if len(matched) > 0:
                                total_modified += len(matched)
                                if action == "assign_itc":
                                    for fld in itc_fields:
                                        work[fld][matched] = int(val)
                                elif action == "mark_noise":
                                    for fld in itc_fields:
                                        work[fld][matched] = int(val) if val else 9999
                                    work["classification"][matched] = 7
                                elif action == "set_classification":
                                    work["classification"][matched] = int(val)

                    writer.write_points(work)
                    processed_points += chunk_len

                    if progress_cb:
                        pct = int((processed_points / total_points) * 100) if total_points > 0 else 0
                        msg = f"Writing points: {processed_points:,} / {total_points:,} ({pct}%)"
                        progress_cb(processed_points, total_points, msg)

        elapsed = time.time() - t_start
        self.has_unsaved_changes = False
        print(f"[PointCloudManager] Commit finished in {elapsed:.2f}s. Points processed: {processed_points:,}. Points modified: {total_modified:,}.")

        return {
            "success": True,
            "output_path": output_path,
            "total_points": processed_points,
            "modified_points": total_modified,
            "edits_count": len(self.staged_edits),
            "elapsed_seconds": elapsed
        }

    # -------------------------------------------------------------------------
    # Spatial Queries & In-Memory Fallbacks
    # -------------------------------------------------------------------------
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

    def assign_itc(self, itc_value: int) -> int:
        """Assign an ITC (tree ID) value to currently selected points in memory."""
        if self.las_data is None or len(self.selected_indices) == 0:
            return 0

        self._ensure_itc_dimension()
        self.las_data.itc[self.selected_indices] = int(itc_value)
        self.has_unsaved_changes = True
        return len(self.selected_indices)

    def set_classification(self, class_value: int) -> int:
        """Set ASPRS classification value for currently selected points in memory."""
        if self.las_data is None or len(self.selected_indices) == 0:
            return 0

        self.las_data.classification[self.selected_indices] = int(class_value)
        self.has_unsaved_changes = True
        return len(self.selected_indices)

    def mark_as_noise(self, noise_itc: int = 9999, class_val: int = 7) -> int:
        """Mark currently selected points as noise in memory."""
        if self.las_data is None or len(self.selected_indices) == 0:
            return 0

        self.set_classification(class_val)
        self.assign_itc(noise_itc)
        self.has_unsaved_changes = True
        return len(self.selected_indices)

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
