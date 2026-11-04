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
* **Categorized Top Ribbon Bar:** Clean tabbed ribbon bar with `File`, `Point`, `Layers`, `Settings`, `Segmentation` (plus `Filter` and `AI` while the Forestry tab is open) for uncluttered, modern desktop workflows.
* **Full ForestFormer3D Pipeline from the UI:** Pick any LAS/LAZ on the `Segmentation` tab and run tiled-tract or single-cloud instance segmentation with editable radius, voxel step, score threshold, tiling and ROI, then load the stitched result straight into the viewport.
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
| **Git** | Clones the vendored Potree renderer and the ForestFormer3D submodule (see below). |
| **Node.js + npm** | Required **once**, to build the renderer assets. Verified on Node v24.22.0 / npm 11.19.1. |
| **CUDA GPU + torch env** | Only for the AI tools (`Segment (ForestFormer3D)`). Needs a CUDA-enabled interpreter with torch — verified on an RTX 5090 Laptop (sm_120) with torch 2.7.1+cu128 in `~/.venvs/ff3d` (Windows: conda env `ff3d`). |

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

### AI segmentation engine (ForestFormer3D)

The `Segment (ForestFormer3D)` tool splits merged crowns into individual trees with a OneFormer3D-style transformer (ICCV 2025 Oral, [arXiv:2506.16991](https://arxiv.org/abs/2506.16991)). Its sources live **inside this project** as a git submodule at `./vendor/ForestFormer3D`, cloned from **`https://github.com/arunbiju969/ForestFormer3D.git` (branch `main`)**. Unlike `potree/`, a submodule is tracked: the superproject stores the exact commit, recorded in prose in [`forestformer3d.version`](forestformer3d.version). The model runs in its own CUDA Python interpreter, so the editor's 3.14 `.venv` stays torch-free.

Two things are **not** in git and must be fetched:

| Artifact | Where it comes from | Lands at |
|---|---|---|
| Model weights (220 MB, gitignored) | Zenodo record [16742708](https://zenodo.org/records/16742708) → `clean_forestformer.zip` | `vendor/ForestFormer3D/work_dirs/clean_forestformer/epoch_3000_fix.pth` |
| torch env (~5 GB) | `vendor/ForestFormer3D/setup_linux.sh` (Linux) · conda env `ff3d` (Windows, see `SETUP_NOTES.md`) | `~/.venvs/ff3d` |

One command does submodule init + weights + verification:

```bash
./setup_ff3d.sh              # Windows: setup_ff3d_assets.bat
./setup_ff3d.sh --venv        # also build ~/.venvs/ff3d (~15 min)
./setup_ff3d.sh --smoke       # prove the stack with a 5 000-point inference
```

A fresh clone is one step behind; the submodule needs one init:

```bash
git clone --recurse-submodules https://github.com/arunbiju969/PotreeEditor.git
# existing clone:  git submodule update --init --recursive
```

`ff3d_module.py` resolves the engine in this order, so a checkout can move without a code edit: `$FF3D_REPO` → `./vendor/ForestFormer3D` → the legacy external checkout at `/run/media/neo/Omen/Projects/ForestFormer3D` (last, as a fallback). The interpreter resolves `$FF3D_PYTHON` → `$FF3D_VENV/bin/python` → `~/.venvs/ff3d/bin/python`. If either is missing, segmentation fails with the fix in the message, not a stack trace.

### Segmentation tab (full pipeline)

The top ribbon has a permanent **Segmentation** category that runs the complete ForestFormer3D pipeline on **any** point cloud — not only the one you have open.

| Control | What it does |
|---|---|
| **Browse… / Use Open Cloud** | Pick the LAS/LAZ to segment; defaults to the cloud currently open |
| **Pipeline** | *Tiled Tract* — cuts the cloud into tiles, segments each, merges and seam-stitches (any size). *Single Cloud* — one cylinder pass over the whole file |
| **Radius / Voxel step / Score ≥** | Model-level: neighbourhood radius (16 m), voxel step (6 m), instance score threshold (0.40) |
| **Tile / Overlap** | Tiling for the tract pipeline (70 m tiles, 10 m overlap) |
| **Clip to ROI** | Restrict a Single Cloud run to an X/Y rectangle |
| **Model / Advanced…** | Config, checkpoint, CUDA device and chunk size; blank = engine defaults |
| **Run Segmentation** | Starts the run; the bar animates and the label streams tiles completed and trees found |
| **Load Result** | Streams the finished cloud into the Potree viewport (converts on demand) |

Runs land in the workspace at `<master>.potreeedit/segmented/pipeline/` as `<stem>_entire_tract_stitched.las` (tract) or `<stem>_ff3d_segmented.las` (single cloud). Parameters are stored per dataset in `edits.db`, so a tract reopens with the settings that produced the last run. Changing a model-level parameter rebuilds the warm model on the next run (~10 s); the tiled tract run reports progress from ForestFormer3D's own `tract_tiling_progress.json`, so the counter is real, not a guess.

---

## Per-Dataset Workspace

Every master you open gets its own sidecar folder, so a dataset keeps its editing state in one place instead of scattering files next to the LAS:

```
tract.laz
tract.potreeedit/
├── workspace.json     manifest: master, created, last commit, commit count
├── edits.db           edit history, split tree-ID routes, staged-edit snapshot
├── segmented/         ForestFormer3D split clouds (tree_<itc>_ff3d_segmented.las)
└── committed.laz      cumulative committed cloud — all edits, stacked
```

* **Commits stack.** Each commit reads `committed.laz` when it exists (the master on the very first commit) and writes back to it atomically, so the workspace holds *the* edited cloud instead of a chain of `_edited_edited.laz` files. The master is never rewritten.
* **Nothing is lost on close.** Staged-but-uncommitted edits are snapshotted into `edits.db` and reappear in the sidebar table the next time you open the same master.
* **History and routes are per dataset.** `edits.db` carries the `edit_history` log, the `tree_edits_index` routes that map a split tree's new ID to its segmented cloud, and the rendering settings saved for that dataset. Opening a converted cloud folder with no master still falls back to the folder-level `settings.db`.

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
├── setup_ff3d.sh               # One-click AI engine: submodule + weights + verify
├── setup_ff3d_assets.bat       # Windows twin of the above
├── forestformer3d.version      # Pinned AI engine revision + fetch/override notes
├── run_potree_editor.bat       # One-click Windows launcher
├── main.py                     # Entry point, renderer resolution, lifecycle
├── server.py                   # Local HTTP server with Range request & REST support
├── app_window.py               # PyQt6 MainWindow with Ribbon, WebEngine & Staged Edits Table
├── pointcloud_manager.py       # Header inspector, staging manager & streaming commit engine
├── ff3d_module.py              # Persistent ForestFormer3D worker (JSON-line stdio)
├── workspace.py                # Per-master sidecar workspace (<stem>.potreeedit/)
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation
├── potree/                     # Vendored renderer (own git clone; untracked)
│   ├── build/potree/potree.js  # Built by gulp — git-ignored, must be built locally
│   ├── libs/                   # three.js, jquery, proj4, PotreeConverter2/…
│   ├── resources/              # icons, shaders, localisation
│   └── src/                    # renderer sources (edit + `npm run build` to change rendering)
├── vendor/
│   └── ForestFormer3D/         # AI segmentation engine (git submodule, own history)
│       ├── ff3d_api.py         # Python API the editor calls: predict_points()
│       ├── configs/            # oneformer3d_qs_radius16_qp300_2many.py
│       └── work_dirs/          # epoch_3000_fix.pth — gitignored, from Zenodo
└── web/
    ├── index.html              # Potree viewer page with floating HUD & staging controls
    ├── app.js                  # Frontend JS connecting Potree tools to Python & Three.js visuals
    └── style.css               # Professional dark-theme styling
```
