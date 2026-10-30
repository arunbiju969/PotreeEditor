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

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | The bundled `.venv` is built on **3.14.6**; `requirements.txt` bounds `PyQt6`/`PyQt6-WebEngine` to `>=6.5.0,<6.12.0` (installed: 6.11.0). |
| **Git** | Clones the vendored Potree renderer (see below). |
| **Node.js + npm** | Required **once**, to build the renderer assets. Verified on Node v24.22.0 / npm 11.19.1. |

---

## First-Time Setup

The editor renders inside `QWebEngineView` from Potree. That renderer now lives **inside this project** at `./potree`, cloned from **`https://github.com/arunbiju969/potree.git` (branch `develop`)** — a fork of upstream `potree/potree`. The exact revision is pinned in [`potree.version`](potree.version) and is **not** tracked by this repo's git.

> **Important:** `potree/build/` is git-ignored by the potree repository itself (`potree/.gitignore` line 1: `/build`). A clone alone therefore gives you `libs/` and `resources/` but **no `build/potree/potree.js`**, and the app starts to a blank black canvas with no error. You must build it once.

One command does clone + build + verify:

```bat
setup_potree_assets.bat
```

It clones the pinned revision, runs `npm install` (whose `postinstall` hook runs `npm run build` → `gulp build pack`), and checks that `build/potree/potree.js`, `build/potree/potree.css`, `three.module.js`, `jquery-3.1.1.min.js` and `PotreeConverter2/PotreeConverter.exe` are all present. To re-render the assets after editing potree sources:

```powershell
cd potree
npm run build
```

**No outside references.** `main.py` resolves the renderer from **`./potree` only** — it consults no environment variable (there is no `POTREE_DIR`) and no sibling folder. If those assets are missing it exits with instructions instead of opening an empty viewport, and it prints which renderer it resolved. Everything the app needs — the Python environment (`.venv`) and the renderer (`potree`) — lives inside this one folder.

---

## How to Run

Double-click or run from command line:

```bat
run_potree_editor.bat
```

Or activate the local virtual environment and launch directly:

```powershell
cd PotreeEditor
.\.venv\Scripts\activate
python main.py
```

The launcher activates the virtual environment in `.venv` (PyQt6, PyQt6-WebEngine, laspy, lazrs, NumPy, SciPy, Matplotlib). It refuses to start — with instructions, not a crash — if `.venv` or the built renderer is missing, and it no longer borrows an environment from a sibling project.

---

## Directory Structure

```
PotreeEditor/
├── setup_potree_assets.bat     # One-click renderer clone + npm build + verify
├── potree.version              # Pinned renderer revision (repo, branch, commit)
├── run_potree_editor.bat       # One-click Windows launcher
├── main.py                     # Entry point, renderer resolution, lifecycle
├── server.py                   # Local HTTP server with Range request & REST support
├── app_window.py               # PyQt6 MainWindow with Ribbon, WebEngine & Staged Edits Table
├── pointcloud_manager.py       # Header inspector, staging manager & streaming commit engine
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation
├── potree/                     # Vendored renderer (own git clone; untracked)
│   ├── build/potree/potree.js  # Built by gulp — git-ignored, must be built locally
│   ├── libs/                   # three.js, jquery, proj4, PotreeConverter2/…
│   ├── resources/              # icons, shaders, localisation
│   └── src/                    # renderer sources (edit + `npm run build` to change rendering)
└── web/
    ├── index.html              # Potree viewer page with floating HUD & staging controls
    ├── app.js                  # Frontend JS connecting Potree tools to Python & Three.js visuals
    └── style.css               # Professional dark-theme styling
```
