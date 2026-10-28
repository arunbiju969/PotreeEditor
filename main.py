"""
Potree Hybrid Viewer - Application Entry Point.
Launches the hybrid architecture:
- Python HTTP Server (Range request support for Potree 2.0 octree binary streaming)
- PyQt6 GUI with embedded Chromium (QWebEngineView)
- Direct LAS/LAZ point cloud spatial analysis and editing bridge
"""

import os
import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from pointcloud_manager import PointCloudManager
from server import HybridServer
from app_window import MainWindow


def main():
    # Configure UTF-8 on Windows
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    # Path discovery
    app_dir = Path(__file__).resolve().parent
    web_dir = app_dir / "web"

    # Reference Potree project directory (contains build/, libs/, resources/, PotreeConverter2/)
    potree_dir = app_dir.parent / "potree"
    if not potree_dir.exists():
        # Fallback to current directory or sibling
        potree_dir = Path(r"C:\Users\w0491597\Documents\Projects\potree")

    print("=================================================================")
    print("      Potree Hybrid Viewer (Experimental Architecture)          ")
    print("=================================================================")
    print(f"[Init] App Directory   : {app_dir}")
    print(f"[Init] Web Assets      : {web_dir}")
    print(f"[Init] Potree Core Lib : {potree_dir}")

    # Initialize Point Cloud Manager (Python / laspy)
    manager = PointCloudManager()

    # Initialize & Start Local Server
    server = HybridServer(
        potree_root=str(potree_dir),
        web_root=str(web_dir),
        manager=manager,
        port=8990
    )
    server.start()

    # Launch PyQt6 Application
    app = QApplication(sys.argv)
    app.setApplicationName("Potree Hybrid Editor")

    window = MainWindow(server=server, manager=manager, potree_dir=str(potree_dir))
    window.show()

    exit_code = app.exec()

    # Clean shutdown
    server.stop()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
