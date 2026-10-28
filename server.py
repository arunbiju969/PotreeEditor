"""
Local HTTP Server for Potree Hybrid Viewer.
Supports:
- HTTP 206 Partial Content (Range requests) for Potree 2.0 octree chunks.
- Serving Potree core assets from the main Potree project directory.
- Serving converted point cloud directories from arbitrary user paths.
- REST API for polygon selection, point picking, and LAS data mutations.
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
                "selected_count": len(self.manager.selected_indices) if (self.manager and self.manager.selected_indices is not None) else 0
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
            if self.manager:
                result = self.manager.select_by_polygon_2d(coords)
                if self.on_event_callback:
                    self.on_event_callback("selection_changed", result)

        elif path == "/api/select_box":
            min_pt = data.get("min", [0, 0, 0])
            max_pt = data.get("max", [0, 0, 0])
            if self.manager:
                result = self.manager.select_by_box_3d(min_pt, max_pt)
                if self.on_event_callback:
                    self.on_event_callback("selection_changed", result)

        elif path == "/api/pick_point":
            x = data.get("x", 0.0)
            y = data.get("y", 0.0)
            z = data.get("z", 0.0)
            if self.manager:
                picked = self.manager.pick_point(x, y, z)
                result = {"picked": picked}
                if self.on_event_callback and picked:
                    self.on_event_callback("point_picked", picked)

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

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(result).encode("utf-8"))

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
