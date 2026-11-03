"""
Local HTTP Server for Potree Hybrid Viewer.
Supports:
- HTTP 206 Partial Content (Range requests) for Potree 2.0 octree chunks.
- Serving Potree core assets from the main Potree project directory.
- Serving converted point cloud directories from arbitrary user paths.
- REST API for polygon selection, point picking, optimistic visual staging, and LAS data mutations.
"""

import os
import sys
import json
import re
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import threading
from typing import Optional, Dict, Any, Callable
from urllib.parse import urlparse, unquote

from pointcloud_manager import PointCloudManager


class HybridRequestHandler(BaseHTTPRequestHandler):
    potree_root: str = ""
    web_root: str = ""
    pointcloud_routes: Dict[str, str] = {}  # alias -> directory path
    manager: Optional[PointCloudManager] = None
    on_event_callback: Optional[Callable[[str, Any], None]] = None

    def log_message(self, format, *args):
        # Suppress routine GET logging to keep terminal output clean
        pass

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type, Accept")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        # 1. API Status
        if path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_cors_headers()
            self.end_headers()
            status = {
                "active_las": self.manager.las_path if self.manager else None,
                "has_unsaved": self.manager.has_unsaved_changes if self.manager else False,
                "selected_count": len(self.manager.selected_indices) if (self.manager and self.manager.selected_indices is not None) else 0,
                "staged_count": len(self.manager.staged_edits) if self.manager else 0,
                "staged_edits": self.manager.staged_edits if self.manager else [],
                "indexing": self.manager.get_indexing_status() if self.manager else None
            }
            self.wfile.write(json.dumps(status).encode("utf-8"))
            return

        # 2. Web UI files (e.g. /, /index.html, /app.js, /style.css)
        if path == "/" or path == "/index.html":
            file_path = os.path.join(self.web_root, "index.html")
            return self._serve_file(file_path)
        elif path.startswith("/web/"):
            rel = path[len("/web/"):]
            file_path = os.path.join(self.web_root, rel)
            return self._serve_file(file_path)

        # 3. Potree core assets (/build/..., /libs/..., /resources/...)
        for prefix in ["/build/", "/libs/", "/resources/", "/src/"]:
            if path.startswith(prefix):
                rel = path.lstrip("/")
                file_path = os.path.join(self.potree_root, rel)
                return self._serve_file(file_path)

        # 4. Point cloud routes (/clouds/<alias>/...)
        if path.startswith("/clouds/"):
            parts = path[len("/clouds/"):].split("/", 1)
            alias = parts[0]
            rel_file = parts[1] if len(parts) > 1 else ""
            if alias in self.pointcloud_routes:
                base_dir = self.pointcloud_routes[alias]
                file_path = os.path.join(base_dir, rel_file)
                return self._serve_file(file_path)

        # Not found
        self.send_response(404)
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(b"404 Not Found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            data = json.loads(body)
        except Exception:
            data = {}

        result = {}

        if path == "/api/select_polygon":
            coords = data.get("coordinates", [])
            points_3d = data.get("points_3d", [])
            if self.manager:
                self.manager.current_selection = {
                    "type": "polygon",
                    "coords": coords,
                    "points_3d": points_3d
                }
                result = self.manager.select_by_polygon_2d(coords)
                if self.on_event_callback:
                    self.on_event_callback("selection_changed", result)

        elif path == "/api/select_box":
            min_pt = data.get("min", [0, 0, 0])
            max_pt = data.get("max", [0, 0, 0])
            if self.manager:
                self.manager.current_selection = {
                    "type": "box",
                    "min": min_pt,
                    "max": max_pt
                }
                result = self.manager.select_by_box_3d(min_pt, max_pt)
                if self.on_event_callback:
                    self.on_event_callback("selection_changed", result)

        elif path == "/api/pick_point":
            x = data.get("x", 0.0)
            y = data.get("y", 0.0)
            z = data.get("z", 0.0)
            if self.manager:
                picked = self.manager.pick_point(x, y, z)
                if picked is None:
                    picked = {"x": x, "y": y, "z": z}
                # Streaming mode: the manager has no per-point access; the ITC
                # travels back from the GPU octree attribute the browser picked.
                if picked.get("itc") is None and data.get("itc") is not None:
                    try:
                        picked["itc"] = int(data["itc"])
                    except (TypeError, ValueError):
                        pass
                if data.get("attributes"):
                    picked["attributes"] = data["attributes"]
                result = {"picked": picked}
                if self.on_event_callback and picked:
                    self.on_event_callback("point_picked", picked)
                # Background pre-warm tree points for full-density viewer
                if picked.get("itc") is not None and hasattr(self.manager, "prefetch_tree"):
                    try:
                        itc_num = int(picked["itc"])
                        if itc_num > 0:
                            self.manager.prefetch_tree(itc_num, x, y)
                    except (ValueError, TypeError):
                        pass

        elif path == "/api/stage_edit":
            if self.manager:
                staged = self.manager.add_staged_edit(data)
                result = {
                    "success": True,
                    "edit": staged,
                    "staged_count": len(self.manager.staged_edits)
                }
                if self.on_event_callback:
                    self.on_event_callback("edit_staged", result)

        elif path == "/api/remove_staged_edit":
            edit_id = data.get("id")
            if self.manager and edit_id:
                removed = self.manager.remove_staged_edit(edit_id)
                result = {
                    "success": removed,
                    "id": edit_id,
                    "staged_count": len(self.manager.staged_edits)
                }
                if self.on_event_callback:
                    self.on_event_callback("edit_removed", result)

        elif path == "/api/clear_staged_edits":
            if self.manager:
                self.manager.clear_staged_edits()
                result = {
                    "success": True,
                    "staged_count": 0
                }
                if self.on_event_callback:
                    self.on_event_callback("edits_cleared", result)

        elif path == "/api/edit":
            action = data.get("action")
            val = data.get("value")
            count = 0
            if self.manager:
                if action == "assign_itc":
                    count = self.manager.assign_itc(int(val))
                elif action == "set_classification":
                    count = self.manager.set_classification(int(val))
                elif action == "mark_noise":
                    count = self.manager.mark_as_noise(noise_itc=int(val) if val else 9999)
                elif action == "delete":
                    count = self.manager.delete_selected_points()

                result = {"success": True, "action": action, "modified_count": count}
                if self.on_event_callback:
                    self.on_event_callback("edit_applied", result)

        elif path == "/api/save_las":
            save_path = data.get("path")
            if self.manager and save_path:
                try:
                    self.manager.save_las(save_path)
                    result = {"success": True, "path": save_path}
                    if self.on_event_callback:
                        self.on_event_callback("saved", result)
                except Exception as e:
                    result = {"success": False, "error": str(e)}

        elif path == "/api/get_tree":
            if self.manager and data.get("itc") is not None:
                itc = int(data["itc"])
                x = float(data["x"]) if data.get("x") is not None else None
                y = float(data["y"]) if data.get("y") is not None else None
                radius = float(data.get("radius", 0.0))
                neighbor_mode = str(data.get("neighbor_mode", "trees_only"))
                ground_radius = float(data.get("ground_radius", 0.0))
                payload = self.manager.get_tree_points(itc, x=x, y=y, radius=radius,
                                                       neighbor_mode=neighbor_mode,
                                                       ground_radius=ground_radius)
                self._send_tree_binary(payload)
                return

        elif path == "/api/segment_tree":
            # Same tree/stand point set as /api/get_tree, labelled by
            # ForestFormer3D: tree_id column carries the FF3D instance
            # (0 = ground, 1..n = each detected tree) instead of the ITC id.
            if self.manager and data.get("itc") is not None:
                itc = int(data["itc"])
                x = float(data["x"]) if data.get("x") is not None else None
                y = float(data["y"]) if data.get("y") is not None else None
                radius = float(data.get("radius", 0.0))
                neighbor_mode = str(data.get("neighbor_mode", "trees_only"))
                ground_radius = float(data.get("ground_radius", 0.0))
                payload = self.manager.get_tree_points(itc, x=x, y=y, radius=radius,
                                                       neighbor_mode=neighbor_mode,
                                                       ground_radius=ground_radius)
                if payload.get("count", 0) > 0:
                    import numpy as np
                    from ff3d_module import get_module
                    xyz = np.asarray(payload["xyz"], dtype=np.float32)
                    _sem, inst, _scores = get_module().predict(xyz)
                    ff_ids = np.where(inst < 0, 0, inst + 1).astype(np.float32)
                    payload = dict(payload)
                    scalars = dict(payload.get("scalars") or {})
                    scalars["itc"] = ff_ids          # JS colours from the itc scalar
                    payload["scalars"] = scalars
                    payload["ff3d_trees"] = int(np.unique(inst[inst >= 0]).size)
                    las_out = self._write_ff3d_las(itc, xyz, _sem, inst, _scores, payload)
                    if las_out:
                        payload["ff3d_las"] = las_out
                self._send_tree_binary(payload)
                return

        elif path == "/api/event":
            event_name = data.get("event")
            if self.on_event_callback and event_name:
                self.on_event_callback(event_name, data)
            result = {"success": True}

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(result).encode("utf-8"))

    def _send_tree_binary(self, payload):
        """
        Binary tree-response for /api/get_tree:
          u32 magic 'TREE' | u32 version | u32 count | u32 flags (1=rgb, 2=intensity, 4=classification, 8=has_scalars)
          f64 centroid[3] | f32 bbox_min[3] | f32 bbox_max[3]
          [if flags & 8: u32 meta_len | utf-8 json bytes (meta_len) | padding to 4 bytes]
          xyz float32[n*3] | rgb uint8[n*3] | padding | intensity uint16[n] | classification uint8[n] | padding to 4 bytes
          [if flags & 8: for each sname in meta.scalars: float32[n]]
        """
        import numpy as np
        import struct

        if payload is None:
            payload = {"count": 0}
        count = payload.get("count", 0)
        flags = 0
        if payload.get("rgb") is not None:
            flags |= 1
        if payload.get("intensity") is not None:
            flags |= 2
        if payload.get("classification") is not None:
            flags |= 4

        scalars = payload.get("scalars", {})
        has_scalars = bool(scalars and count > 0)
        has_neighbors = bool(payload.get("neighbor_count", 0) > 0)
        if has_scalars or has_neighbors:
            flags |= 8

        header = struct.pack(
            "<4sIII3d6f",
            b"TREE", 2, count, flags,
            *payload.get("centroid", [0.0, 0.0, 0.0]),
            *payload.get("bbox_min", [0.0, 0.0, 0.0]),
            *payload.get("bbox_max", [0.0, 0.0, 0.0])
        )

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(header)
        if count == 0:
            return

        if flags & 8:
            scalar_names = list(scalars.keys()) if has_scalars else []
            meta_dict = {
                "scalars": scalar_names,
                "target_count": payload.get("target_count", count),
                "neighbor_count": payload.get("neighbor_count", 0),
                "radius": payload.get("radius", 0.0),
                "neighbor_mode": payload.get("neighbor_mode", "trees_only")
            }
            if payload.get("ff3d_trees") is not None:
                meta_dict["ff3d_trees"] = int(payload["ff3d_trees"])
            if payload.get("ff3d_las"):
                meta_dict["ff3d_las"] = str(payload["ff3d_las"])
            meta_json = json.dumps(meta_dict).encode("utf-8")
            meta_len = len(meta_json)
            self.wfile.write(struct.pack("<I", meta_len))
            self.wfile.write(meta_json)
            pad_meta = (4 - ((4 + meta_len) % 4)) % 4
            if pad_meta > 0:
                self.wfile.write(b"\x00" * pad_meta)

        self.wfile.write(np.asarray(payload["xyz"], dtype=np.float32).tobytes())
        total_bytes = 0
        if flags & 1:
            rgb_bytes = np.asarray(payload["rgb"], dtype=np.uint8).tobytes()
            self.wfile.write(rgb_bytes)
            if len(rgb_bytes) % 2 != 0:
                self.wfile.write(b"\x00")  # 2-byte alignment padding for uint16 intensity
        if flags & 2:
            self.wfile.write(np.asarray(payload["intensity"], dtype=np.uint16).tobytes())
        if flags & 4:
            cls_bytes = np.asarray(payload["classification"], dtype=np.uint8).tobytes()
            self.wfile.write(cls_bytes)
            total_bytes += len(cls_bytes)

        if has_scalars:
            pad_scalars = (4 - (total_bytes % 4)) % 4
            if pad_scalars > 0:
                self.wfile.write(b"\x00" * pad_scalars)
            for sname in scalar_names:
                arr = np.asarray(scalars[sname], dtype=np.float32)
                self.wfile.write(arr.tobytes())

    def _write_ff3d_las(self, itc, xyz, sem, inst, scores, payload):
        """Persist the segmentation result: the exact points that were shown,
        with the master's header (CRS/scales) plus ff3d_tree_id / ff3d_semantic
        / ff3d_score columns. Saved next to the master file."""
        import os
        import numpy as np
        try:
            import laspy
        except ImportError:
            return None
        src = getattr(self.manager, "las_path", None)
        hdr = None
        master_hdr = None
        if src and os.path.exists(src):
            try:
                with laspy.open(src) as r:
                    master_hdr = r.header
            except Exception:
                master_hdr = None
        # Format-3 point format with the ff3d columns declared up front (the
        # record stride must include them before arrays are assigned).
        pf = laspy.PointFormat(3)
        try:
            for name, dtype in (("ff3d_tree_id", "uint16"),
                                ("ff3d_semantic", "uint8"),
                                ("ff3d_score", "float32")):
                pf.add_extra_dimension(laspy.ExtraBytesParams(name=name, type=np.dtype(dtype)))
        except Exception:
            pass   # older laspy: standard dims only
        hdr = laspy.LasHeader(point_format=pf, version="1.4")
        if master_hdr is not None:
            # Inherit the master's scales, offsets and CRS VLRs (drop LASzip;
            # output is uncompressed .las).
            hdr.scales = master_hdr.scales
            hdr.offsets = master_hdr.offsets
            src_vlrs = list(getattr(master_hdr, "vlrs",
                                    getattr(master_hdr, "vlr_list", [])))
            keep = [v for v in src_vlrs
                    if getattr(v, "srcl_id", "") in
                    ("LASF_Projection", "OGC MR", "GeoTIFF", "WKT1.4.2-")
                    or getattr(v, "srcl_id", "").startswith("epsg")]
            try:
                hdr.vlrs = keep
            except AttributeError:
                hdr.vlr_list = keep
        out = laspy.LasData(hdr)
        # Tree payload xyz is centroid-relative; restore master CRS coords.
        cen = np.asarray(payload.get("centroid", [0.0, 0.0, 0.0]), dtype=np.float64)
        xyz = np.asarray(xyz, dtype=np.float64) + cen
        out.x = np.asarray(xyz[:, 0], dtype=np.float64)
        out.y = np.asarray(xyz[:, 1], dtype=np.float64)
        out.z = np.asarray(xyz[:, 2], dtype=np.float64)
        if payload.get("intensity") is not None:
            out.intensity = np.asarray(payload["intensity"], dtype=np.uint16)
        if payload.get("classification") is not None:
            out.classification = np.asarray(payload["classification"], dtype=np.uint8)
        rgb = payload.get("rgb")
        if rgb is not None:
            rgb = np.asarray(rgb, dtype=np.uint8)
            out.red = rgb[:, 0].astype(np.uint16) * 256
            out.green = rgb[:, 1].astype(np.uint16) * 256
            out.blue = rgb[:, 2].astype(np.uint16) * 256
        try:
            out["ff3d_tree_id"] = np.asarray(inst).astype(np.uint16)
            out["ff3d_semantic"] = np.asarray(sem).astype(np.uint8)
            out["ff3d_score"] = np.asarray(scores).astype(np.float32)
        except Exception:
            pass
        out_dir = os.path.dirname(src) if src and os.path.exists(src) else os.getcwd()
        out_path = os.path.join(out_dir, f"tree_{itc}_ff3d_segmented.las")
        out.write(out_path)
        return out_path

    def _serve_file(self, file_path: str):
        if not os.path.exists(file_path) or os.path.isdir(file_path):
            self.send_response(404)
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(f"File not found: {file_path}".encode("utf-8"))
            return

        file_size = os.path.getsize(file_path)
        content_type, _ = mimetypes.guess_type(file_path)
        if not content_type:
            if file_path.endswith(".bin"):
                content_type = "application/octet-stream"
            elif file_path.endswith(".json"):
                content_type = "application/json"
            elif file_path.endswith(".js"):
                content_type = "application/javascript"
            elif file_path.endswith(".css"):
                content_type = "text/css"
            else:
                content_type = "application/octet-stream"

        # Handle HTTP 206 Partial Content (Range requests)
        range_header = self.headers.get("Range")
        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                start_str, end_str = match.groups()
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1
                if end >= file_size:
                    end = file_size - 1

                length = end - start + 1

                self.send_response(206)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.send_cors_headers()
                self.end_headers()

                with open(file_path, "rb") as f:
                    f.seek(start)
                    self.wfile.write(f.read(length))
                return

        # Regular 200 OK
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        # App assets (web/app.js, style.css, index.html) are iterated on constantly;
        # without validators QWebEngine serves them from cache and edits need a
        # full restart. Localhost re-fetches are free.
        if file_path.startswith(self.web_root):
            self.send_header("Cache-Control", "no-cache")
        self.send_cors_headers()
        self.end_headers()

        with open(file_path, "rb") as f:
            chunk = f.read(64 * 1024)
            while chunk:
                self.wfile.write(chunk)
                chunk = f.read(64 * 1024)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


class HybridServer:
    def __init__(self, potree_root: str, web_root: str, manager: PointCloudManager, port: int = 8990):
        self.potree_root = os.path.abspath(potree_root)
        self.web_root = os.path.abspath(web_root)
        self.manager = manager
        self.port = port
        self.httpd: Optional[ThreadedHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.event_callbacks = []

    def register_pointcloud_dir(self, alias: str, dir_path: str):
        """Register a point cloud folder to be accessible via /clouds/<alias>/."""
        HybridRequestHandler.pointcloud_routes[alias] = os.path.abspath(dir_path)

    def on_event(self, callback: Callable[[str, Any], None]):
        self.event_callbacks.append(callback)

    def _dispatch_event(self, event_name: str, data: Any):
        for cb in self.event_callbacks:
            try:
                cb(event_name, data)
            except Exception as e:
                print(f"[HybridServer] Error in event callback: {e}")

    def start(self):
        HybridRequestHandler.potree_root = self.potree_root
        HybridRequestHandler.web_root = self.web_root
        HybridRequestHandler.manager = self.manager
        HybridRequestHandler.on_event_callback = self._dispatch_event

        # Find available port if 8990 is taken
        for p in range(self.port, self.port + 20):
            try:
                self.httpd = ThreadedHTTPServer(("127.0.0.1", p), HybridRequestHandler)
                self.port = p
                break
            except OSError:
                continue

        if not self.httpd:
            raise RuntimeError(f"Could not bind HTTP server on ports {self.port}-{self.port + 20}")

        print(f"[HybridServer] Started server at http://127.0.0.1:{self.port}/")
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        if self.httpd:
            print("[HybridServer] Stopping server...")
            self.httpd.shutdown()
            self.httpd.server_close()
