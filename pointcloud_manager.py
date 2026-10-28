"""
Point Cloud Data Manager for the Hybrid Viewer.
Handles loading LAS/LAZ files, spatial queries (polygon, box, point pick),
attribute/classification/ITC modifications, and writing updated files.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import laspy
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree


class PointCloudManager:
    def __init__(self):
        self.las_path: Optional[str] = None
        self.las_data: Optional[laspy.LasData] = None
        self.xyz: Optional[np.ndarray] = None
        self.kdtree: Optional[cKDTree] = None
        self.selected_indices: np.ndarray = np.array([], dtype=int)
        self.has_unsaved_changes: bool = False

    def load_las(self, path: str) -> Dict[str, Any]:
        """Load a LAS/LAZ file into memory for editing."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        print(f"[PointCloudManager] Reading LAS file: {path}...")
        self.las_data = laspy.read(path)
        self.las_path = path

        # Cache coordinate arrays for fast spatial queries
        self.xyz = np.vstack((self.las_data.x, self.las_data.y, self.las_data.z)).T

        # Build 2D/3D spatial index lazily or on demand (build kdtree on XY for top-down queries)
        self.kdtree = None
        self.selected_indices = np.array([], dtype=int)
        self.has_unsaved_changes = False

        # Gather summary
        total_points = len(self.xyz)
        bounds = {
            "min": [float(np.min(self.xyz[:, 0])), float(np.min(self.xyz[:, 1])), float(np.min(self.xyz[:, 2]))],
            "max": [float(np.max(self.xyz[:, 0])), float(np.max(self.xyz[:, 1])), float(np.max(self.xyz[:, 2]))]
        }

        # Check existing dimensions
        extra_dims = [dim.name for dim in self.las_data.point_format.extra_dimensions]
        has_itc = "itc" in extra_dims or "treeID" in extra_dims or "tree_id" in extra_dims

        unique_classes = np.unique(self.las_data.classification).tolist() if hasattr(self.las_data, "classification") else []

        unique_itc = []
        if has_itc:
            itc_attr = "itc" if "itc" in extra_dims else ("treeID" if "treeID" in extra_dims else "tree_id")
            itc_vals = np.array(getattr(self.las_data, itc_attr))
            unique_itc = np.unique(itc_vals).tolist()

        print(f"[PointCloudManager] Loaded {total_points:,} points. Has ITC: {has_itc}. Classes: {unique_classes}")

        return {
            "path": path,
            "filename": os.path.basename(path),
            "total_points": total_points,
            "bounds": bounds,
            "has_itc": has_itc,
            "classes": unique_classes,
            "unique_itc_count": len(unique_itc)
        }

    def _ensure_itc_dimension(self, default_val: int = 0):
        """Ensure the 'itc' extra dimension exists on the LasData object."""
        if self.las_data is None:
            return

        extra_dims = [dim.name for dim in self.las_data.point_format.extra_dimensions]
        if "itc" in extra_dims:
            return

        print("[PointCloudManager] Adding 'itc' extra dimension to LAS file...")
        self.las_data.add_extra_dim(laspy.ExtraBytesParams(name="itc", type=np.int32))
        self.las_data.itc[:] = default_val

    def select_by_polygon_2d(self, polygon_coords: List[List[float]]) -> Dict[str, Any]:
        """
        Select points inside a 2D polygon (X, Y) in world coordinates.
        polygon_coords: [[x0, y0], [x1, y1], ...]
        """
        if self.las_data is None or self.xyz is None:
            return {"count": 0, "indices": []}

        poly = np.array(polygon_coords)
        if len(poly) < 3:
            return {"count": 0, "indices": []}

        min_x, max_x = np.min(poly[:, 0]), np.max(poly[:, 0])
        min_y, max_y = np.min(poly[:, 1]), np.max(poly[:, 1])

        # Step 1: Fast bounding box pre-filtering in NumPy
        xs = self.xyz[:, 0]
        ys = self.xyz[:, 1]
        bbox_mask = (xs >= min_x) & (xs <= max_x) & (ys >= min_y) & (ys <= max_y)
        candidate_indices = np.where(bbox_mask)[0]

        if len(candidate_indices) == 0:
            self.selected_indices = np.array([], dtype=int)
            return {"count": 0, "indices": []}

        # Step 2: Exact polygon inclusion test using Matplotlib Path
        candidate_pts = self.xyz[candidate_indices, :2]
        path = MplPath(poly[:, :2])
        inside_mask = path.contains_points(candidate_pts)

        self.selected_indices = candidate_indices[inside_mask]
        count = len(self.selected_indices)

        # Inspect classes & ITC in selection
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
        """Select points inside a 3D bounding box."""
        if self.las_data is None or self.xyz is None:
            return {"count": 0}

        mask = (
            (self.xyz[:, 0] >= min_pt[0]) & (self.xyz[:, 0] <= max_pt[0]) &
            (self.xyz[:, 1] >= min_pt[1]) & (self.xyz[:, 1] <= max_pt[1]) &
            (self.xyz[:, 2] >= min_pt[2]) & (self.xyz[:, 2] <= max_pt[2])
        )

        self.selected_indices = np.where(mask)[0]
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
        """Find the nearest point to (x, y, z) in the dataset."""
        if self.las_data is None or self.xyz is None:
            return None

        if self.kdtree is None:
            print("[PointCloudManager] Building spatial index for picking...")
            self.kdtree = cKDTree(self.xyz)

        dist, idx = self.kdtree.query([x, y, z])
        if dist > max_dist:
            return None

        pt = self.xyz[idx]
        classification = int(self.las_data.classification[idx]) if hasattr(self.las_data, "classification") else 0
        intensity = int(self.las_data.intensity[idx]) if hasattr(self.las_data, "intensity") else 0
        itc = int(self.las_data.itc[idx]) if hasattr(self.las_data, "itc") else None

        return {
            "index": int(idx),
            "distance": float(dist),
            "x": float(pt[0]),
            "y": float(pt[1]),
            "z": float(pt[2]),
            "classification": classification,
            "intensity": intensity,
            "itc": itc
        }

    def assign_itc(self, itc_value: int) -> int:
        """Assign an ITC (tree ID) value to currently selected points."""
        if self.las_data is None or len(self.selected_indices) == 0:
            return 0

        self._ensure_itc_dimension()
        self.las_data.itc[self.selected_indices] = int(itc_value)
        self.has_unsaved_changes = True
        return len(self.selected_indices)

    def set_classification(self, class_value: int) -> int:
        """Set ASPRS classification value for currently selected points."""
        if self.las_data is None or len(self.selected_indices) == 0:
            return 0

        self.las_data.classification[self.selected_indices] = int(class_value)
        self.has_unsaved_changes = True
        return len(self.selected_indices)

    def mark_as_noise(self, noise_itc: int = 9999, class_val: int = 7) -> int:
        """Mark currently selected points as noise (Class 7 and ITC 9999)."""
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
        self.xyz = self.xyz[keep_mask]
        self.kdtree = None
        self.selected_indices = np.array([], dtype=int)
        self.has_unsaved_changes = True
        return del_count

    def save_las(self, output_path: str):
        """Save the modified point cloud to disk."""
        if self.las_data is None:
            raise ValueError("No point cloud loaded to save.")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        print(f"[PointCloudManager] Writing updated LAS/LAZ to: {output_path}...")
        self.las_data.write(output_path)
        self.has_unsaved_changes = False
        print("[PointCloudManager] Save complete.")
