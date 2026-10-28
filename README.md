# PotreeEditor

A desktop point cloud viewer and editing environment combining:
1. **Frontend:** **Potree (Three.js + WebGL)** running inside an embedded Chromium viewport (`QWebEngineView`) for smooth **Level of Detail (LOD) out-of-core streaming** of massive point clouds (millions to billions of points).
2. **Backend:** **Python (`laspy` + NumPy + SciPy + Matplotlib)** running in the native desktop process for direct point cloud attribute editing (ITC / Tree ID assignment, ASPRS classification, noise conversion, spatial queries, and LAS/LAZ file export).
3. **Local Bridge:** A threaded Python HTTP server with full **HTTP 206 Partial Content (Byte Range)** support for Potree 2.0 octree streaming, plus REST endpoints for bi-directional communication between the 3D viewport and Python.

---

## Architecture Overview

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      PyQt6 Application Window                          │
 │                                                                        │
 │  ┌───────────────────────────────┐ ┌────────────────────────────────┐  │
 │  │        Native Sidebar         │ │        QWebEngineView          │  │
 │  │ - Open Potree (metadata.json) │ │    (Potree Streaming Engine)   │  │
 │  │ - Open LAS/LAZ & Auto-Convert │ │                                │  │
 │  │ - Selection Statistics        │ │ - Smooth LOD Octree Rendering  │  │
 │  │ - Assign Tree ID (ITC)        │ │ - 3D Navigation & Orbit        │  │
 │  │ - Mark Noise (Class 7)        │ │ - Polygon Lasso Selection      │  │
 │  │ - ASPRS Classification        │ │ - 3D Box Selection             │  │
 │  │ - Point Budget & EDL Controls │ │ - Real-Time Point Inspection   │  │
 │  │ - Save Modified LAS File      │ │                                │  │
 │  └───────────────┬───────────────┘ └────────────────┬───────────────┘  │
 └──────────────────┼──────────────────────────────────┼──────────────────┘
                    │                                  │
                    ▼                                  ▼
      ┌───────────────────────────┐      ┌───────────────────────────┐
      │    PointCloudManager      │      │       HybridServer        │
      │   (Python / laspy)        │◄────►│  (127.0.0.1 HTTP Server)  │
      │                           │      │                           │
      │ - 2D/3D Point-in-Polygon  │      │ - HTTP 206 Range Requests │
      │ - KD-Tree Point Picking   │      │ - REST API (/api/select)  │
      │ - ITC & Class Mutations   │      │ - Serves Potree Assets    │
      │ - laspy File Read & Write │      │                           │
      └───────────────────────────┘      └───────────────────────────┘
```

---

## Key Features

* **Massive Point Cloud Performance:** Streams multi-gigabyte Potree octrees without consuming excessive system RAM or choking the GPU.
* **Interactive Polygon / Lasso Selection:** Click points to outline a tree crown or forest region directly on the 3D canvas (double-click or right-click to close).
* **Tree Crown (ITC) Assignment:** Instantly assign Individual Tree Crown (ITC) IDs to selected point clusters.
* **Noise Conversion:** Convert unwanted branches, birds, or multipath reflections to ASPRS Class 7 (Noise) and ITC 9999.
* **Point Attribute Inspector:** Click any point to inspect its 3D world coordinates, ASPRS classification, intensity, and Tree ID.
* **Auto-Conversion:** Opening a raw `.las` / `.laz` file without a converted octree offers automatic conversion via `PotreeConverter2`.
* **Export to LAS:** Save all modifications cleanly to standard `.las` / `.laz` files.
* **Completely Isolated:** Built in a dedicated directory without altering existing files in `potree` or `TreeAIBox`.

---

## How to Run

Double-click or run from command line:
```bat
C:\Users\w0491597\Documents\Projects\PotreeEditor\run_potree_editor.bat
```

The launcher automatically detects and uses the existing Python virtual environment in `TreeAIBox\.venv` (which already includes PyQt6, PyQt6-WebEngine, laspy, NumPy, SciPy, and Matplotlib).

---

## Directory Structure

```
PotreeEditor/
├── run_potree_editor.bat       # One-click Windows launcher
├── main.py                     # Application entry point & lifecycle
├── server.py                   # Local HTTP server with Range request & REST support
├── app_window.py               # PyQt6 MainWindow with QWebEngineView & controls
├── pointcloud_manager.py       # laspy-based spatial querying and editing engine
├── README.md                   # System documentation
└── web/
    ├── index.html              # Potree viewer page with floating HUD
    ├── app.js                  # Frontend JS connecting Potree tools to Python
    └── style.css               # Styling for HUD, cards, and banners
```
