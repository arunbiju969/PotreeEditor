"""Crown polygon extraction from segmented ForestFormer3D LAS/LAZ point clouds.

Uses ``semantic_pred`` (0=ground, 1=wood/trunk/twigs, 2=leaf/crown) together
with tree instance IDs (``itc`` / ``tree_id`` / ``treeID``) to delineate 2D/3D
crown polygons per tree, compute crown metrics, and export GeoJSON or CSV/WKT.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import laspy
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError

SEMANTIC_GROUND = 0
SEMANTIC_WOOD = 1
SEMANTIC_LEAF = 2

TREE_ID_CANDIDATES = ("itc", "tree_id", "treeID", "ff3d_tree_id")
SEMANTIC_CANDIDATES = ("semantic_pred", "ff3d_semantic")


class CrownExtractionCancelled(Exception):
    """Raised when crown extraction is cancelled by the user."""


def detect_crown_fields(dimensions: Iterable[str]) -> Tuple[Optional[str], Optional[str]]:
    """Detect the tree instance ID field and semantic prediction field names."""
    dims = list(dimensions)
    tree_field = next((name for name in TREE_ID_CANDIDATES if name in dims), None)
    sem_field = next((name for name in SEMANTIC_CANDIDATES if name in dims), None)
    return tree_field, sem_field


def _proj4_hint_from_wkt(wkt: Optional[str]) -> Optional[str]:
    """Derive a proj4 projection string from a LAS OGC WKT string when available."""
    if not wkt:
        return None
    m = re.search(r"UTM\s+zone\s+(\d+)\s*([NS])", wkt, re.IGNORECASE)
    if m:
        zone = int(m.group(1))
        hemi = m.group(2).upper()
        upper = wkt.upper()
        datum = "+datum=NAD83" if "NAD83" in upper else "+datum=WGS84"
        south = " +south" if hemi == "S" else ""
        return f"+proj=utm +zone={zone}{south} {datum} +units=m +no_defs"
    return None


def _extract_crs_info(header: Any) -> Tuple[Optional[str], Optional[str]]:
    """Extract (crs_wkt, proj4_hint) safely even without pyproj installed."""
    crs_wkt: Optional[str] = None
    try:
        crs = header.parse_crs()
        if crs is not None:
            crs_wkt = crs.to_wkt()
    except Exception:
        crs_wkt = None

    if not crs_wkt:
        try:
            for vlr in getattr(header, "vlrs", []) or []:
                if getattr(vlr, "record_id", None) == 2112:
                    raw = getattr(vlr, "string", None)
                    if raw is None:
                        raw_bytes = getattr(vlr, "record_data", b"")
                        if isinstance(raw_bytes, (bytes, bytearray)):
                            raw = bytes(raw_bytes).decode("utf-8", errors="ignore")
                    if raw:
                        crs_wkt = str(raw).rstrip("\x00").strip() or None
                        break
        except Exception:
            crs_wkt = None

    return crs_wkt, _proj4_hint_from_wkt(crs_wkt)


def _signed_area_2d(ring: np.ndarray) -> float:
    if len(ring) < 3:
        return 0.0
    # Shift relative to vertex 0 to prevent catastrophic float64 cancellation on UTM coords
    rx = ring[:, 0] - ring[0, 0]
    ry = ring[:, 1] - ring[0, 1]
    return 0.5 * float(np.sum(rx * np.roll(ry, -1) - np.roll(rx, -1) * ry))


def _ensure_ccw(ring: np.ndarray) -> np.ndarray:
    if len(ring) >= 3 and _signed_area_2d(ring) < 0.0:
        return ring[::-1].copy()
    return ring


def compute_convex_hull_2d(pts: np.ndarray) -> Optional[np.ndarray]:
    """Compute a counter-clockwise 2D convex hull ring ``(K, 2)``."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return None
    try:
        hull = ConvexHull(pts)
    except (QhullError, ValueError):
        return None
    ring = pts[hull.vertices]
    if len(ring) < 3:
        return None
    return _ensure_ccw(ring)


def _trace_boundary_loops(directed_edges: np.ndarray) -> List[List[int]]:
    """Trace closed loops from directed boundary edges ``(M, 2)``."""
    adj: Dict[int, List[int]] = {}
    for u, v in directed_edges.tolist():
        adj.setdefault(u, []).append(v)

    loops: List[List[int]] = []
    while adj:
        start_u = next(iter(adj))
        nxt = adj[start_u].pop(0)
        if not adj[start_u]:
            del adj[start_u]

        chain = [start_u]
        pos_in_chain = {start_u: 0}
        curr = nxt

        while curr is not None:
            if curr == start_u:
                if len(chain) >= 3:
                    loops.append(chain)
                break
            if curr in pos_in_chain:
                # Closed internal sub-loop caused by a pinch vertex; record sub-loop
                # and continue tracing from `curr` if outgoing edges remain.
                idx = pos_in_chain[curr]
                sub = chain[idx:]
                if len(sub) >= 3:
                    loops.append(sub)
                chain = chain[:idx]
                pos_in_chain = {node: i for i, node in enumerate(chain)}
                # Walk forward from curr
                if not chain:
                    start_u = curr
                    chain = [curr]
                    pos_in_chain = {curr: 0}
            else:
                pos_in_chain[curr] = len(chain)
                chain.append(curr)

            if curr not in adj or not adj[curr]:
                break
            nxt = adj[curr].pop(0)
            if not adj[curr]:
                del adj[curr]
            curr = nxt

    return loops


def compute_concave_hull_2d(pts: np.ndarray, alpha: float = 1.5) -> Optional[np.ndarray]:
    """Compute a counter-clockwise 2D concave alpha-shape outer ring ``(K, 2)``.

    Falls back to the convex hull if the point set is too small, collinear,
    or if ``alpha`` disconnects/breaks closed outer boundary loops.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 4:
        return compute_convex_hull_2d(pts)

    try:
        tri = Delaunay(pts)
    except (QhullError, ValueError):
        return compute_convex_hull_2d(pts)

    simplices = tri.simplices
    if len(simplices) == 0:
        return compute_convex_hull_2d(pts)

    p0 = pts[simplices[:, 0]]
    p1 = pts[simplices[:, 1]]
    p2 = pts[simplices[:, 2]]

    a = np.hypot(p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1])
    b = np.hypot(p2[:, 0] - p1[:, 0], p2[:, 1] - p1[:, 1])
    c = np.hypot(p0[:, 0] - p2[:, 0], p0[:, 1] - p2[:, 1])
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    two_area = np.abs(cross)

    with np.errstate(divide="ignore", invalid="ignore"):
        circum_r = np.where(two_area > 1e-12, (a * b * c) / (2.0 * two_area), np.inf)

    valid_mask = circum_r <= float(alpha)
    if not np.any(valid_mask):
        return compute_convex_hull_2d(pts)

    valid_tris = simplices[valid_mask].copy()
    valid_cross = cross[valid_mask]
    neg = valid_cross < 0.0
    if np.any(neg):
        v1 = valid_tris[neg, 1].copy()
        valid_tris[neg, 1] = valid_tris[neg, 2]
        valid_tris[neg, 2] = v1

    dir_edges = np.vstack([
        valid_tris[:, [0, 1]],
        valid_tris[:, [1, 2]],
        valid_tris[:, [2, 0]],
    ])
    undir = np.sort(dir_edges, axis=1)
    edge_keys = (undir[:, 0].astype(np.uint64) << np.uint64(32)) | undir[:, 1].astype(np.uint64)
    _, inv, counts = np.unique(edge_keys, return_inverse=True, return_counts=True)
    boundary_edges = dir_edges[counts[inv] == 1]

    if len(boundary_edges) < 3:
        return compute_convex_hull_2d(pts)

    loops = _trace_boundary_loops(boundary_edges)
    if not loops:
        return compute_convex_hull_2d(pts)

    best_ring: Optional[np.ndarray] = None
    best_area = 0.0
    for loop in loops:
        if len(loop) < 3:
            continue
        candidate = pts[np.asarray(loop, dtype=np.int64)]
        signed = _signed_area_2d(candidate)
        if signed > best_area:
            best_area = signed
            best_ring = candidate

    if best_ring is None or best_area <= 1e-6:
        return compute_convex_hull_2d(pts)
    return best_ring


def _rdp_open(points: np.ndarray, tol: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification for an open 2D polyline."""
    n = len(points)
    if n <= 2:
        return points

    start = points[0]
    end = points[-1]
    seg = end - start
    seg_len_sq = float(seg[0] * seg[0] + seg[1] * seg[1])

    mid = points[1:-1]
    if seg_len_sq <= 1e-18:
        dists = np.hypot(mid[:, 0] - start[0], mid[:, 1] - start[1])
    else:
        t = np.clip(((mid[:, 0] - start[0]) * seg[0] + (mid[:, 1] - start[1]) * seg[1]) / seg_len_sq, 0.0, 1.0)
        proj_x = start[0] + t * seg[0]
        proj_y = start[1] + t * seg[1]
        dists = np.hypot(mid[:, 0] - proj_x, mid[:, 1] - proj_y)

    max_idx = int(np.argmax(dists))
    max_dist = float(dists[max_idx])
    if max_dist <= tol:
        return np.vstack([start, end])

    split = max_idx + 1
    left = _rdp_open(points[: split + 1], tol)
    right = _rdp_open(points[split:], tol)
    return np.vstack([left[:-1], right])


def simplify_closed_ring(ring: np.ndarray, tolerance: float) -> np.ndarray:
    """Simplify a closed 2D polygon ring while keeping at least 3 vertices."""
    ring = np.asarray(ring, dtype=np.float64)
    if tolerance <= 0.0 or len(ring) <= 4:
        return ring

    d2 = np.sum((ring - ring[0]) ** 2, axis=1)
    far_idx = int(np.argmax(d2))
    if far_idx == 0:
        return ring

    chain1 = _rdp_open(ring[: far_idx + 1], float(tolerance))
    chain2 = _rdp_open(np.vstack([ring[far_idx:], ring[:1]]), float(tolerance))
    stitched = np.vstack([chain1[:-1], chain2[:-1]])
    if len(stitched) < 3 or abs(_signed_area_2d(stitched)) <= 1e-9:
        return ring
    return _ensure_ccw(stitched)


def polygon_metrics(ring: np.ndarray) -> Dict[str, float]:
    """Compute area, perimeter, equivalent crown diameter, and centroid of a 2D ring."""
    ring = np.asarray(ring, dtype=np.float64)
    if len(ring) < 3:
        return {"area_m2": 0.0, "perimeter_m": 0.0, "crown_diameter_m": 0.0, "centroid_x": 0.0, "centroid_y": 0.0}

    x0 = float(ring[0, 0])
    y0 = float(ring[0, 1])
    rx = ring[:, 0] - x0
    ry = ring[:, 1] - y0
    rx_next = np.roll(rx, -1)
    ry_next = np.roll(ry, -1)

    cross = rx * ry_next - rx_next * ry
    signed_area = 0.5 * float(np.sum(cross))
    area_m2 = abs(signed_area)
    perimeter_m = float(np.sum(np.hypot(rx_next - rx, ry_next - ry)))
    crown_diameter_m = 2.0 * math.sqrt(area_m2 / math.pi) if area_m2 > 0.0 else 0.0

    if abs(signed_area) > 1e-9:
        cx = x0 + float(np.sum((rx + rx_next) * cross) / (6.0 * signed_area))
        cy = y0 + float(np.sum((ry + ry_next) * cross) / (6.0 * signed_area))
    else:
        cx = x0 + float(rx.mean())
        cy = y0 + float(ry.mean())

    return {
        "area_m2": area_m2,
        "perimeter_m": perimeter_m,
        "crown_diameter_m": crown_diameter_m,
        "centroid_x": cx,
        "centroid_y": cy,
    }


def _dedup_sorted_tree_cells(
    tid: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    tree_cells: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
):
    """Deduplicate quantized ``(tid, gx, gy)`` cells in one chunk and append per tree."""
    if len(tid) == 0:
        return
    order = np.lexsort((gy, gx, tid))
    tid_s = tid[order]
    gx_s = gx[order]
    gy_s = gy[order]

    uniq = np.empty(len(tid_s), dtype=bool)
    uniq[0] = True
    if len(tid_s) > 1:
        uniq[1:] = (tid_s[1:] != tid_s[:-1]) | (gx_s[1:] != gx_s[:-1]) | (gy_s[1:] != gy_s[:-1])

    tid_u = tid_s[uniq]
    gx_u = gx_s[uniq].astype(np.int32)
    gy_u = gy_s[uniq].astype(np.int32)

    if len(tid_u) == 0:
        return
    splits = np.flatnonzero(tid_u[1:] != tid_u[:-1]) + 1
    starts = np.concatenate(([0], splits))
    ends = np.concatenate((splits, [len(tid_u)]))
    for s_idx, e_idx in zip(starts.tolist(), ends.tolist()):
        t_val = int(tid_u[s_idx])
        tree_cells.setdefault(t_val, []).append((gx_u[s_idx:e_idx], gy_u[s_idx:e_idx]))


def _accumulate_chunk(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    tree_ids: np.ndarray,
    semantic: Optional[np.ndarray],
    semantic_mode: str,
    grid_res: float,
    ox_ref: float,
    oy_ref: float,
    tree_stats: Dict[int, List[float]],
    tree_cells: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
):
    """Filter chunk points, accumulate exact Z statistics, and store quantized XY cells."""
    mask = tree_ids > 0
    if semantic_mode == "leaf":
        if semantic is None:
            raise ValueError("Semantic prediction field not found for Leaf/Crown filtering.")
        mask &= (semantic == SEMANTIC_LEAF)
    elif semantic_mode == "wood_and_leaf":
        if semantic is None:
            raise ValueError("Semantic prediction field not found for Wood+Crown filtering.")
        mask &= ((semantic == SEMANTIC_WOOD) | (semantic == SEMANTIC_LEAF))
    elif semantic_mode != "all":
        raise ValueError(f"Unsupported semantic_mode: {semantic_mode!r}")

    if not np.any(mask):
        return

    tid = np.asarray(tree_ids[mask], dtype=np.int64)
    xf = np.asarray(x[mask], dtype=np.float64)
    yf = np.asarray(y[mask], dtype=np.float64)
    zf = np.asarray(z[mask], dtype=np.float64)

    # Exact per-tree point count and Z stats before XY grid deduplication
    u_tids, inv = np.unique(tid, return_inverse=True)
    counts = np.bincount(inv)
    z_sums = np.bincount(inv, weights=zf)
    z_mins = np.full(len(u_tids), np.inf, dtype=np.float64)
    z_maxs = np.full(len(u_tids), -np.inf, dtype=np.float64)
    np.minimum.at(z_mins, inv, zf)
    np.maximum.at(z_maxs, inv, zf)

    for i, t_raw in enumerate(u_tids.tolist()):
        t_val = int(t_raw)
        c_i = int(counts[i])
        zmin_i = float(z_mins[i])
        zmax_i = float(z_maxs[i])
        zsum_i = float(z_sums[i])
        cur = tree_stats.get(t_val)
        if cur is None:
            tree_stats[t_val] = [c_i, zmin_i, zmax_i, zsum_i]
        else:
            cur[0] += c_i
            if zmin_i < cur[1]:
                cur[1] = zmin_i
            if zmax_i > cur[2]:
                cur[2] = zmax_i
            cur[3] += zsum_i

    gx = np.floor((xf - ox_ref) / grid_res).astype(np.int64)
    gy = np.floor((yf - oy_ref) / grid_res).astype(np.int64)
    _dedup_sorted_tree_cells(tid, gx, gy, tree_cells)


def _finalize_crowns(
    tree_stats: Dict[int, List[float]],
    tree_cells: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
    *,
    ox_ref: float,
    oy_ref: float,
    grid_res: float,
    semantic_mode: str,
    method: str,
    alpha: float,
    min_points: int,
    simplify_tol: float,
    source: str,
    tree_field: Optional[str],
    semantic_field: Optional[str],
    crs_wkt: Optional[str],
    proj4_hint: Optional[str],
    t0: float,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    sorted_tids = sorted(tree_stats.keys())
    total_trees = len(sorted_tids)
    crowns: List[Dict[str, Any]] = []
    skipped = 0
    total_crown_points = 0

    for idx, tid in enumerate(sorted_tids):
        if cancel_check is not None and (idx & 63) == 0 and cancel_check():
            raise CrownExtractionCancelled("Crown extraction cancelled.")

        if progress_cb is not None and total_trees > 0 and ((idx % max(1, total_trees // 35)) == 0 or idx == total_trees - 1):
            pct = 65 + int(round(((idx + 1) / total_trees) * 35.0))
            progress_cb(min(99, pct), f"Building crown polygons: {idx + 1}/{total_trees} trees...")

        pt_count, z_min, z_max, z_sum = tree_stats[tid]
        pt_count_i = int(pt_count)
        if pt_count_i < int(min_points):
            skipped += 1
            continue

        cell_chunks = tree_cells.get(tid)
        if not cell_chunks:
            skipped += 1
            continue

        if len(cell_chunks) == 1:
            gx_all, gy_all = cell_chunks[0]
        else:
            gx_cat = np.concatenate([c[0] for c in cell_chunks])
            gy_cat = np.concatenate([c[1] for c in cell_chunks])
            order = np.lexsort((gy_cat, gx_cat))
            gx_s = gx_cat[order]
            gy_s = gy_cat[order]
            uniq = np.empty(len(gx_s), dtype=bool)
            uniq[0] = True
            if len(gx_s) > 1:
                uniq[1:] = (gx_s[1:] != gx_s[:-1]) | (gy_s[1:] != gy_s[:-1])
            gx_all = gx_s[uniq]
            gy_all = gy_s[uniq]

        if len(gx_all) < 3:
            skipped += 1
            continue

        # Work in local coordinates during triangulation for maximum numeric stability
        pts_local = np.column_stack((
            (gx_all.astype(np.float64) + 0.5) * grid_res,
            (gy_all.astype(np.float64) + 0.5) * grid_res,
        ))

        if method == "convex":
            ring_local = compute_convex_hull_2d(pts_local)
        elif method == "concave":
            ring_local = compute_concave_hull_2d(pts_local, alpha=float(alpha))
        else:
            raise ValueError(f"Unsupported crown extraction method: {method!r}")

        if ring_local is None or len(ring_local) < 3:
            skipped += 1
            continue

        if simplify_tol > 0.0:
            ring_local = simplify_closed_ring(ring_local, float(simplify_tol))

        ring_world = np.column_stack((ring_local[:, 0] + ox_ref, ring_local[:, 1] + oy_ref))
        metrics = polygon_metrics(ring_world)
        if metrics["area_m2"] <= 1e-6:
            skipped += 1
            continue

        z_mean = float(z_sum) / max(1, pt_count_i)
        crowns.append({
            "tree_id": int(tid),
            "point_count": pt_count_i,
            "coords": [[round(float(px), 3), round(float(py), 3)] for px, py in ring_world],
            "centroid": [round(metrics["centroid_x"], 3), round(metrics["centroid_y"], 3)],
            "area_m2": round(metrics["area_m2"], 3),
            "perimeter_m": round(metrics["perimeter_m"], 3),
            "crown_diameter_m": round(metrics["crown_diameter_m"], 3),
            "z_top": round(float(z_max), 3),
            "z_mean": round(z_mean, 3),
            "z_base": round(float(z_min), 3),
            "crown_depth_m": round(max(0.0, float(z_max - z_min)), 3),
        })
        total_crown_points += pt_count_i

    if crowns:
        mean_cx = float(np.mean([c["centroid"][0] for c in crowns]))
        mean_cy = float(np.mean([c["centroid"][1] for c in crowns]))
        mean_zt = float(np.mean([c["z_top"] for c in crowns]))
        total_area = float(sum(c["area_m2"] for c in crowns))
        mean_area = total_area / len(crowns)
        mean_diam = float(np.mean([c["crown_diameter_m"] for c in crowns]))
    else:
        mean_cx, mean_cy, mean_zt = 0.0, 0.0, 0.0
        total_area = 0.0
        mean_area = 0.0
        mean_diam = 0.0

    elapsed_s = time.time() - t0
    if progress_cb is not None:
        progress_cb(100, f"Extracted {len(crowns):,} crown polygons in {elapsed_s:.2f}s.")

    return {
        "source": source,
        "tree_field": tree_field,
        "semantic_field": semantic_field,
        "semantic_mode": semantic_mode,
        "method": method,
        "alpha": float(alpha),
        "min_points": int(min_points),
        "simplify_tol": float(simplify_tol),
        "crs_wkt": crs_wkt,
        "proj4_hint": proj4_hint,
        "origin": [mean_cx, mean_cy, mean_zt],
        "crowns": crowns,
        "summary": {
            "count": len(crowns),
            "skipped_trees": skipped,
            "total_crown_points": total_crown_points,
            "total_area_m2": round(total_area, 3),
            "mean_area_m2": round(mean_area, 3),
            "mean_diameter_m": round(mean_diam, 3),
            "mean_z_top": round(mean_zt, 3),
            "elapsed_s": round(elapsed_s, 3),
        },
    }


def extract_crowns_from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    tree_ids: np.ndarray,
    semantic: Optional[np.ndarray] = None,
    *,
    semantic_mode: str = "leaf",
    method: str = "concave",
    alpha: float = 1.5,
    min_points: int = 15,
    simplify_tol: float = 0.15,
    grid_res: float = 0.10,
    source: str = "",
    tree_field: Optional[str] = "itc",
    semantic_field: Optional[str] = "semantic_pred",
    crs_wkt: Optional[str] = None,
    proj4_hint: Optional[str] = None,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Extract crown polygons from in-memory coordinate and attribute arrays."""
    t0 = time.time()
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)
    tid_arr = np.asarray(tree_ids, dtype=np.int64)
    sem_arr = np.asarray(semantic, dtype=np.int64) if semantic is not None else None

    if tree_field is None:
        raise ValueError(
            "No tree ID field ('itc', 'tree_id', 'treeID', or 'ff3d_tree_id') found in point cloud."
        )
    if semantic_mode in ("leaf", "wood_and_leaf") and sem_arr is None:
        raise ValueError(
            "No 'semantic_pred' / 'ff3d_semantic' field found. Run ForestFormer3D segmentation "
            "or switch Points mode to 'All Tree Points'."
        )

    if len(x_arr) == 0:
        ox_ref, oy_ref = 0.0, 0.0
    else:
        ox_ref = float(np.min(x_arr))
        oy_ref = float(np.min(y_arr))

    tree_stats: Dict[int, List[float]] = {}
    tree_cells: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}

    if cancel_check is not None and cancel_check():
        raise CrownExtractionCancelled("Crown extraction cancelled.")
    if progress_cb is not None:
        progress_cb(15, "Filtering canopy points and building tree grid...")

    _accumulate_chunk(
        x_arr, y_arr, z_arr, tid_arr, sem_arr,
        semantic_mode=semantic_mode,
        grid_res=float(grid_res),
        ox_ref=ox_ref,
        oy_ref=oy_ref,
        tree_stats=tree_stats,
        tree_cells=tree_cells,
    )

    return _finalize_crowns(
        tree_stats, tree_cells,
        ox_ref=ox_ref,
        oy_ref=oy_ref,
        grid_res=float(grid_res),
        semantic_mode=semantic_mode,
        method=method,
        alpha=alpha,
        min_points=min_points,
        simplify_tol=simplify_tol,
        source=source,
        tree_field=tree_field,
        semantic_field=semantic_field,
        crs_wkt=crs_wkt,
        proj4_hint=proj4_hint or _proj4_hint_from_wkt(crs_wkt),
        t0=t0,
        progress_cb=progress_cb,
        cancel_check=cancel_check,
    )


def extract_crowns_from_las(
    las_path: Optional[str],
    *,
    las_data: Any = None,
    semantic_mode: str = "leaf",
    method: str = "concave",
    alpha: float = 1.5,
    min_points: int = 15,
    simplify_tol: float = 0.15,
    grid_res: float = 0.10,
    chunk_size: int = 5_000_000,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Extract crown polygons from either an in-memory ``las_data`` or a streaming LAS/LAZ file."""
    t0 = time.time()

    if las_data is not None:
        dims = [d.name for d in las_data.point_format.dimensions]
        tree_field, sem_field = detect_crown_fields(dims)
        if tree_field is None:
            raise ValueError(
                "This point cloud has no tree segmentation field ('itc', 'tree_id', 'treeID'). "
                "Run ForestFormer3D segmentation first."
            )
        if semantic_mode in ("leaf", "wood_and_leaf") and sem_field is None:
            raise ValueError(
                "This cloud does not have 'semantic_pred' / 'ff3d_semantic'. "
                "Run ForestFormer3D segmentation or switch Points to 'All Tree Points'."
            )
        crs_wkt, proj4_hint = _extract_crs_info(las_data.header)

        return extract_crowns_from_arrays(
            las_data.x,
            las_data.y,
            las_data.z,
            getattr(las_data, tree_field),
            getattr(las_data, sem_field) if sem_field is not None else None,
            semantic_mode=semantic_mode,
            method=method,
            alpha=alpha,
            min_points=min_points,
            simplify_tol=simplify_tol,
            grid_res=grid_res,
            source=las_path or "",
            tree_field=tree_field,
            semantic_field=sem_field,
            crs_wkt=crs_wkt,
            proj4_hint=proj4_hint,
            progress_cb=progress_cb,
            cancel_check=cancel_check,
        )

    if not las_path or not os.path.exists(las_path):
        raise FileNotFoundError(f"Point cloud file not found: {las_path}")

    with laspy.open(las_path) as reader:
        hdr = reader.header
        dims = [d.name for d in hdr.point_format.dimensions]
        tree_field, sem_field = detect_crown_fields(dims)
        if tree_field is None:
            raise ValueError(
                "This point cloud has no tree segmentation field ('itc', 'tree_id', 'treeID'). "
                "Run ForestFormer3D segmentation first."
            )
        if semantic_mode in ("leaf", "wood_and_leaf") and sem_field is None:
            raise ValueError(
                "This cloud does not have 'semantic_pred' / 'ff3d_semantic'. "
                "Run ForestFormer3D segmentation or switch Points to 'All Tree Points'."
            )

        crs_wkt, proj4_hint = _extract_crs_info(hdr)

        ox_ref = float(hdr.x_min)
        oy_ref = float(hdr.y_min)
        total_points = max(1, int(hdr.point_count))
        total_chunks = max(1, math.ceil(total_points / int(chunk_size)))

        tree_stats: Dict[int, List[float]] = {}
        tree_cells: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}

        for c_idx, chunk in enumerate(reader.chunk_iterator(int(chunk_size))):
            if cancel_check is not None and cancel_check():
                raise CrownExtractionCancelled("Crown extraction cancelled.")

            if progress_cb is not None:
                pct = int(round((c_idx / total_chunks) * 65.0))
                progress_cb(pct, f"Scanning semantic/tree points: chunk {c_idx + 1}/{total_chunks}...")

            sem_vals = getattr(chunk, sem_field) if sem_field is not None else None
            _accumulate_chunk(
                chunk.x,
                chunk.y,
                chunk.z,
                getattr(chunk, tree_field),
                sem_vals,
                semantic_mode=semantic_mode,
                grid_res=float(grid_res),
                ox_ref=ox_ref,
                oy_ref=oy_ref,
                tree_stats=tree_stats,
                tree_cells=tree_cells,
            )

        if progress_cb is not None:
            progress_cb(65, f"Computing boundaries for {len(tree_stats):,} trees...")

    return _finalize_crowns(
        tree_stats, tree_cells,
        ox_ref=ox_ref,
        oy_ref=oy_ref,
        grid_res=float(grid_res),
        semantic_mode=semantic_mode,
        method=method,
        alpha=alpha,
        min_points=min_points,
        simplify_tol=simplify_tol,
        source=las_path,
        tree_field=tree_field,
        semantic_field=sem_field,
        crs_wkt=crs_wkt,
        proj4_hint=proj4_hint,
        t0=t0,
        progress_cb=progress_cb,
        cancel_check=cancel_check,
    )


def export_crowns_geojson(
    result: Dict[str, Any],
    out_path: str,
    *,
    include_z: bool = True,
    height_mode: str = "top",
) -> int:
    """Export extracted crown polygons as a GeoJSON FeatureCollection."""
    crowns = result.get("crowns") or []
    features: List[Dict[str, Any]] = []

    for c in crowns:
        coords = c.get("coords") or []
        if len(coords) < 3:
            continue
        if height_mode == "base":
            z_val = float(c.get("z_base", 0.0))
        elif height_mode == "mean":
            z_val = float(c.get("z_mean", 0.0))
        else:
            z_val = float(c.get("z_top", 0.0))

        if include_z:
            ring = [[float(px), float(py), z_val] for px, py in coords]
        else:
            ring = [[float(px), float(py)] for px, py in coords]
        ring.append(list(ring[0]))

        features.append({
            "type": "Feature",
            "id": int(c["tree_id"]),
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring],
            },
            "properties": {
                "tree_id": int(c["tree_id"]),
                "point_count": int(c["point_count"]),
                "area_m2": float(c["area_m2"]),
                "perimeter_m": float(c["perimeter_m"]),
                "crown_diameter_m": float(c["crown_diameter_m"]),
                "z_top": float(c["z_top"]),
                "z_mean": float(c["z_mean"]),
                "z_base": float(c["z_base"]),
                "crown_depth_m": float(c["crown_depth_m"]),
                "centroid_x": float(c["centroid"][0]),
                "centroid_y": float(c["centroid"][1]),
            },
        })

    payload = {
        "type": "FeatureCollection",
        "metadata": {
            "source": result.get("source", ""),
            "semantic_mode": result.get("semantic_mode", ""),
            "method": result.get("method", ""),
            "alpha": result.get("alpha"),
            "crs_wkt": result.get("crs_wkt"),
            "summary": result.get("summary", {}),
        },
        "features": features,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return len(features)


def export_crowns_csv(result: Dict[str, Any], out_path: str) -> int:
    """Export extracted crown metrics and 2D WKT polygons as CSV."""
    crowns = result.get("crowns") or []
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fieldnames = [
        "tree_id", "point_count", "centroid_x", "centroid_y",
        "area_m2", "perimeter_m", "crown_diameter_m",
        "z_top", "z_mean", "z_base", "crown_depth_m", "wkt_polygon",
    ]
    count = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in crowns:
            coords = c.get("coords") or []
            if len(coords) < 3:
                continue
            closed = list(coords) + [coords[0]]
            wkt = "POLYGON ((" + ", ".join(f"{float(px):.3f} {float(py):.3f}" for px, py in closed) + "))"
            writer.writerow({
                "tree_id": int(c["tree_id"]),
                "point_count": int(c["point_count"]),
                "centroid_x": float(c["centroid"][0]),
                "centroid_y": float(c["centroid"][1]),
                "area_m2": float(c["area_m2"]),
                "perimeter_m": float(c["perimeter_m"]),
                "crown_diameter_m": float(c["crown_diameter_m"]),
                "z_top": float(c["z_top"]),
                "z_mean": float(c["z_mean"]),
                "z_base": float(c["z_base"]),
                "crown_depth_m": float(c["crown_depth_m"]),
                "wkt_polygon": wkt,
            })
            count += 1
    return count
