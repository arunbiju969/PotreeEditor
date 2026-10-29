# PotreeEditor

A desktop point cloud viewer and editing environment combining:
1. **Frontend:** **Potree (Three.js + WebGL)** running inside an embedded Chromium viewport (`QWebEngineView`) for smooth **Level of Detail (LOD) out-of-core streaming** of massive point clouds (millions to billions of points).
2. **Backend:** **Python (`laspy` + NumPy + SciPy + Matplotlib)** running in the native desktop process for direct point cloud attribute editing (ITC / Tree ID assignment, ASPRS classification, noise conversion, spatial queries, and streaming LAS/LAZ file export).
3. **Local Bridge:** A threaded Python HTTP server with full **HTTP 206 Partial Content (Byte Range)** support for Potree 2.0 octree streaming, plus REST endpoints for bi-directional communication between the 3D viewport and Python.
4. **Optimistic Visual Staging + Deferred Commit:** Point clouds load instantly (<0.1s, ~200MB RAM) via header inspection. Delineated crowns and noise classifications are staged visually in the 3D viewport with colored boundaries and floating 3D labels. Dense point modification is streamed in a single deferred pass upon export.

---

## Architecture Overview

```
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                              PyQt6 Application Window                                  │
 │                                                                                        │
 │  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
 │  │       Top Ribbon Bar: [File]  [Point]  [Settings] (Categorized Command Tabs)     │  │
 │  └──────────────────────────────────────────────────────────────────────────────────┘  │
 │                                                                                        │
 │  ┌───────────────────────────────┐ ┌────────────────────────────────────────────────┐  │
 │  │        Native Sidebar         │ │                 QWebEngineView                 │  │
 │  │ - Fast Header Inspection      │ │           (Potree Streaming Engine)            │  │
 │  │ - Active Polygon Selection    │ │                                                │  │
 │  │ - Assign Tree ID (ITC)        │ │ - Smooth LOD Octree Rendering (HTTP 206)       │  │
 │  │ - Convert to Noise (Class 7)  │ │ - 3D Orbit & Pan Navigation                    │  │
 │  │ - ASPRS Classification        │ │ - Interactive Polygon Lasso Delineation        │  │
 │  │ - Staged Edits Table & [X]    │ │ - 3D Colored Visual Staged Boundaries & Labels │  │
 │  │ - Commit & Save to LAS/LAZ    │ │ - Floating HUD Controls & Quick Staging        │  │
 │  └───────────────┬───────────────┘ └───────────────────────┬────────────────────────┘  │
 └──────────────────┼─────────────────────────────────────────┼───────────────────────────┘
                    │                                         │
                    ▼                                         ▼
      ┌───────────────────────────┐             ┌───────────────────────────┐
      │    PointCloudManager      │             │       HybridServer        │
      │   (Python / laspy)        │◄───────────►│  (127.0.0.1 HTTP Server)  │
      │                           │             │                           │
      │ - Fast Header Metadata    │             │ - HTTP 206 Range Requests │
      │ - Visual Staging Registry │             │ - REST API (/api/stage)   │
      │ - Streaming Chunk Commit  │             │ - Serves Potree Assets    │
      │ - Raw Coord Integer Box   │             │                           │
      │ - laspy Streaming Output  │             │                           │
      └───────────────────────────┘             └───────────────────────────┘
```

---

## Key Features

* **Instant Startup (<0.1s) & Low RAM (~200MB):** Never blocks on multi-gigabyte raw LAS files upon opening `metadata.json`. Potree streams LOD octrees on demand.
* **Optimistic Visual Staging:** Delineated tree crowns and noise areas are highlighted immediately in the 3D viewport with closed colored line loops, glowing vertices, and floating 3D text billboards (`Tree #101`, `Noise (7)`).
* **Categorized Top Ribbon Bar:** Clean tabbed ribbon bar with `File`, `Point`, and `Settings` categories for uncluttered, modern desktop workflows.
* **Single-Pass Deferred Commit:** When ready, a background streaming worker updates millions of points chunk-by-chunk in a single pass without memory overhead.
* **Auto-Incrementing Tree IDs:** Automatically increments the Tree ID for rapid sequential crown drafting.
* **Individual Staged Edit Management:** Active staged edits are tracked in a sidebar table with individual deletion buttons and a master clear button.
* **Eye-Dome Lighting (EDL) & Splats:** Full control over EDL radius, strength, opacity, splat quality (HQ paraboloids), and point budgets.
* **Zero Emojis:** Clean, distraction-free technical UI adhering to GIS and remote sensing standards.

---

## How to Run

Double-click or run from command line:
```bat
E:\Projects\PotreeEditor\run_potree_editor.bat
```

Or activate the local virtual environment and launch directly:
```powershell
cd E:\Projects\PotreeEditor
.\.venv\Scripts\activate
python main.py
```

The launcher automatically activates the dedicated virtual environment in `.venv` (configured with Python 3.12, PyQt6, PyQt6-WebEngine, laspy, lazrs, NumPy, SciPy, and Matplotlib).

---

## Directory Structure

```
PotreeEditor/
├── .venv/                      # Dedicated Python 3.12 virtual environment
├── run_potree_editor.bat       # One-click Windows launcher
├── main.py                     # Application entry point & lifecycle
├── server.py                   # Local HTTP server with Range request & REST support
├── app_window.py               # PyQt6 MainWindow with Ribbon, WebEngine & Staged Edits Table
├── pointcloud_manager.py       # Header inspector, staging manager & streaming commit engine
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation
└── web/
    ├── index.html              # Potree viewer page with floating HUD & staging controls
    ├── app.js                  # Frontend JS connecting Potree tools to Python & Three.js visuals
    └── style.css               # Professional dark-theme styling
```
