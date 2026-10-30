"""
Potree Hybrid Viewer - Application Entry Point.
Launches the hybrid architecture:
- Python HTTP Server (Range request support for Potree 2.0 octree binary streaming)
- PyQt6 GUI with embedded Chromium (QWebEngineView)
- Direct LAS/LAZ point cloud spatial analysis and editing bridge
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from pointcloud_manager import PointCloudManager
from server import HybridServer
from app_window import MainWindow


# Files the Potree viewer cannot boot without. index.html loads /build/potree/potree.js
# and /build/potree/potree.css, and app.js imports /libs/three.js/build/three.module.js.
REQUIRED_POTREE_FILES = (
    "build/potree/potree.js",
    "build/potree/potree.css",
    "libs/three.js/build/three.module.js",
    "libs/jquery/jquery-3.1.1.min.js",
)


def potree_is_usable(candidate: Path):
    """Return (ok, reason). build/ is git-ignored in the potree repo, so a fresh
    clone can be present yet unbuilt, which shows up as a blank viewport."""
    if not candidate.is_dir():
        return False, "directory missing"
    missing = [rel for rel in REQUIRED_POTREE_FILES if not (candidate / rel).is_file()]
    if missing:
        return False, "missing " + ", ".join(missing) + " (run: cd potree && npm install)"
    return True, "assets present"


def resolve_potree_dir(app_dir: Path):
    """Return the vendored renderer as (path, source_label), or (None, reason).

    This project is self-contained: the renderer must live at ./potree, at the
    revision pinned in potree.version. No environment variable and no
    sibling-project fallback is consulted, so the app can never silently render
    with a Potree copy that lives outside the project folder.
    """
    vendored = app_dir / "potree"
    ok, reason = potree_is_usable(vendored)
    if ok:
        return vendored, "pinned ./potree clone"
    return None, f"vendored ./potree unusable: {reason}"


def main():
    # Configure UTF-8 on Windows
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    # Path discovery
    app_dir = Path(__file__).resolve().parent
    web_dir = app_dir / "web"

    potree_dir, why = resolve_potree_dir(app_dir)
    if potree_dir is None:
        print("=================================================================")
        print("      Potree Hybrid Viewer (Experimental Architecture)          ")
        print("=================================================================")
        print(f"[Init] App Directory   : {app_dir}")
        print(f"[Init] Web Assets      : {web_dir}")
        print(f"[Init] ERROR: no usable Potree renderer. {why}")
        print("")
        print("This project ships its own renderer copy under ./potree, but the")
        print("file the viewer renders from (potree/build/potree/potree.js) is not")
        print("tracked by git in the potree repository - it must be built locally.")
        print("")
        print("Run the one-time setup (clones the pinned revision + builds assets):")
        print("    setup_potree_assets.bat")
        print("or, with an existing ./potree checkout:")
        print("    cd potree && npm install")
        print("")
        print("Pinned revision is recorded in potree.version.")
        sys.exit(1)

    print("=================================================================")
    print("      Potree Hybrid Viewer (Experimental Architecture)          ")
    print("=================================================================")
    print(f"[Init] App Directory   : {app_dir}")
    print(f"[Init] Web Assets      : {web_dir}")
    print(f"[Init] Potree Core Lib : {potree_dir}  ({why})")

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
