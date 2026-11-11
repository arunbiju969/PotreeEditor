# ForestFormer3D Whole-Tract Parameter Quality Sweep — Agent Handoff Guide

This document explains how to pull the updated `PotreeEditor` + `vendor/ForestFormer3D` pipeline on another machine, set up the CUDA inference environment, and complete the remaining whole-tract parameter quality sweep runs (**Runs 3–6**) for direct comparison against the already-completed baselines (**Runs 1–2**).

---

## 1. What Was Updated & Pushed Upstream

Both repositories (`PotreeEditor` and the `vendor/ForestFormer3D` submodule on `origin/main`) include the latest high-throughput low-memory pipeline optimizations:

- **Streaming On-Disk Pre-Tiler (`vendor/ForestFormer3D/stream_tiler.py`)**
  - Streams large `.laz`/`.las` files in 3M-point chunks (`extract_tiles_streaming`) into per-tile buffered files under `<output-dir>/raw_tiles/` so host RAM during tiling stays under ~200 MB (instead of loading a ~15–28 GB master cloud per worker).
  - Cleans up each raw tile immediately after inference finishes and removes `raw_tiles/` on completion (override via `FF3D_KEEP_RAW_TILES=1` or disable streaming via `FF3D_DISABLE_STREAMING_TILES=1`).
- **Asynchronous Pipelined Inference (`vendor/ForestFormer3D/predict_pipelined.py` + `run_inference_fast.py`)**
  - Overlaps CPU kNN prep, GPU batch execution, and CPU mask/score decode/deduplication (`predict_pipelined`) so tiles do not stall sequentially on CPU post-processing between regions.
- **LPT Load-Balanced Multi-Worker Orchestrator (`vendor/ForestFormer3D/run_parallel.py` & `finalize_parallel.py`)**
  - `run_parallel.py` pre-tiles once up front, forwards `--radius`, `--step-size`, and `--score-th` to workers, and assigns tiles using greedy LPT load balancing by buffered point count (`n_eval`) so workers finish evenly.
  - `finalize_parallel.py` remaps per-tile global `tree_id` / `itc` offsets cleanly, merges `<name>_entire_tract_segmented.las`, runs boundary seam stitching into `<name>_entire_tract_stitched.las`, and removes `raw_tiles/`.
- **Portable Sweep & Quality Evaluator (`vendor/ForestFormer3D/benchmark_sweep.py`)**
  - Runs the parameter matrix sequentially (each run parallelized across `--workers`), verifies free disk space (>= 40 GB default), computes streaming LAS quality metrics, and updates `comparison_report.md` and `sweep_results.json` after every run.
- **PotreeEditor UI & Subprocess Bridge (`app_window.py`, `ff3d_module.py`)**
  - Forwards UI `Radius`, `Voxel step`, `Score >=`, and `Workers` to `run_parallel.py`, handles clean process-group cancellation (`stop()`), and preserves existing COPC bridge & forestry viewer improvements.

---

## 2. Machine Setup Instructions

### 2.1 Clone or Update Repositories

```bash
# Fresh clone (with submodule):
git clone --recurse-submodules https://github.com/arunbiju969/PotreeEditor.git
cd PotreeEditor

# OR update an existing clone:
git pull origin main
git submodule update --init --recursive
```

### 2.2 Install Inference Environment & Model Weights

**Linux (recommended native CUDA venv at `~/.venvs/ff3d`):**
```bash
./setup_ff3d.sh --venv --smoke
```
This initializes the pinned submodule SHA (`forestformer3d.version`), downloads `clean_forestformer.zip` (Zenodo record `16742708`) to `vendor/ForestFormer3D/work_dirs/clean_forestformer/epoch_3000_fix.pth`, builds `~/.venvs/ff3d` via `vendor/ForestFormer3D/setup_linux.sh`, and runs a 5,000-point smoke test.

**Windows / Custom Python env:**
- See `vendor/ForestFormer3D/SETUP_NOTES.md` and `setup_ff3d_assets.bat` (or pass `--python <path-to-python>` to `benchmark_sweep.py`).

### 2.3 Point Cloud Dataset (Not in Git)

`.laz` and `.las` point clouds are gitignored and must be transferred separately to the target machine:

- **Dataset file:** `20260424_LLR_L3_entire_tract_stitched_normalized.laz`
- **Point count:** `213,325,437` points
- **Tiling geometry used across all baselines:** `--tile-size 200.0 --tile-overlap 10.0` (yields **6 active tiles** for this tract)

### 2.4 Hardware, VRAM, and Disk Requirements

- **Disk space (critical):** Each completed run produces **~31 GB** of output (`tiles/` + `_entire_tract_segmented.las` + `_entire_tract_stitched.las`), plus up to ~10–15 GB temporary `raw_tiles/` peak before cleanup. Ensure **at least 165 GB free** on `--output-root` before running all 4 remaining runs (**>= 40 GB free** per single run) on a fast SSD/NVMe volume.
- **GPU VRAM & `--workers`:** On a 24 GB GPU, 2 workers used ~3.9 GB peak VRAM per worker.
  - **>= 16 GB VRAM:** `--workers 2` (default preset). If CPU cores >= 16 and host RAM >= 32 GB, you may test `--workers 3` (since 6 active tiles divide evenly into 2 tiles per worker for `--workers 3` or 3 tiles per worker for `--workers 2`).
  - **<= 12 GB VRAM:** pass `--workers 1` to avoid CUDA OOM.

---

## 3. Baseline Results Already Completed (Reference — Do Not Re-Run)

Two full-tract baselines on `20260424_LLR_L3_entire_tract_stitched_normalized.laz` (`200m` tiles / `10m` overlap) have already been completed and verified:

| Run | Params (`r`, `step`, `score`, `w`) | Wall Time | Stitched Trees (Pre -> Reconnected) | Veg Captured % (`tree_pts`) | Mean Score | Median Pts/Tree (P25 / P75) | Small (<500 pts) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run 1 (Dense Baseline)** | `r=16.0m, s=3.0m, th=0.40, w=1` | `258.5m` | **11,424** (`11,880` - `456`) | `94.07%` (`119,360,599`) | `0.7687` | `3,325` (`862` / `10,298`) | `2,069` |
| **Run 2 (Standard Fast)** | `r=16.0m, s=6.0m, th=0.40, w=2` | `66.0m` | **10,933** (`11,389` - `456`) | `92.26%` (`117,033,601`) | `0.7438` | `3,550` (`919` / `10,700`) | `1,905` |

Baseline observation: Increasing stride from `3.0m` to `6.0m` with 2 workers cut runtime nearly `4x` (`258.5m` -> `66.0m`) while retaining `95.7%` of detected trees (`10,933` vs `11,424`) and missing ~`1.8%` of vegetation points (mostly smaller understory crowns).

---

## 4. Parameter Quality Sweep To Run Next (Runs 3–6)

Run **Runs 3–6** on the target machine to map the speed/recall/precision Pareto frontier:

| Run ID | Label | `radius` | `step-size` | `score-th` | Purpose / Hypothesis |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `run3_step8_th40_w2` | **Run 3 (Ultra-Fast Stride)** | `16.0m` | `8.0m` | `0.40` | Minimal overlap `r/2` stride; tests speed floor vs dropped crowns |
| `run4_step4_th50_w2` | **Run 4 (High Precision)** | `16.0m` | `4.0m` | `0.50` | Native `r/4 = 4.0m` stride + stricter confidence filter (`>=0.50`) |
| `run5_step4_th30_w2` | **Run 5 (High Recall Understory)** | `16.0m` | `4.0m` | `0.30` | Native `4.0m` stride + permissive threshold (`>=0.30`) for suppressed understory |
| `run6_r12_step4_th40_w2` | **Run 6 (Tight 12m Radius)** | `12.0m` | `4.0m` | `0.40` | Tighter 12m cylinder for higher transformer query density per m^2 |

> **Important:** Run 3 was previously started and cancelled immediately at tile 0 on the source machine; its output directory does not exist yet on a fresh machine, so all of **Runs 3–6** start cleanly from scratch.

### Option A: Automated Full Sweep (Recommended)

1. Dry-run first to inspect commands and verify paths:
   ```bash
   ~/.venvs/ff3d/bin/python -u vendor/ForestFormer3D/benchmark_sweep.py \
     --input /path/to/20260424_LLR_L3_entire_tract_stitched_normalized.laz \
     --output-root /path/to/ff3d_comparisons \
     --runs run3 run4 run5 run6 \
     --dry-run
   ```

2. Execute Runs 3–6 sequentially (resumable per run if interrupted; updates `comparison_report.md` and `sweep_results.json` after each run):
   ```bash
   ~/.venvs/ff3d/bin/python -u vendor/ForestFormer3D/benchmark_sweep.py \
     --input /path/to/20260424_LLR_L3_entire_tract_stitched_normalized.laz \
     --output-root /path/to/ff3d_comparisons \
     --runs run3 run4 run5 run6 \
     --workers 2
   ```
   *(Pass `--workers 1` if GPU VRAM <= 12 GB, or `--workers 3` if testing 3-way parallelism on high-VRAM GPUs.)*

### Option B: Manual Per-Run Execution (Example: Run 3)

If you want to launch or monitor a single run manually:

```bash
export PYTHONPATH="$PWD/vendor/ForestFormer3D"
export PYTHONUNBUFFERED=1

# Step 1: Parallel tile inference
~/.venvs/ff3d/bin/python -u vendor/ForestFormer3D/run_parallel.py \
  --input /path/to/20260424_LLR_L3_entire_tract_stitched_normalized.laz \
  --output-dir /path/to/ff3d_comparisons/run3_step8_th40_w2 \
  --workers 2 \
  --tile-size 200.0 \
  --tile-overlap 10.0 \
  --radius 16.0 \
  --step-size 8.0 \
  --score-th 0.40

# Step 2: Offset compaction, tract merge, seam stitch, and raw_tiles cleanup
~/.venvs/ff3d/bin/python -u vendor/ForestFormer3D/finalize_parallel.py \
  --input /path/to/20260424_LLR_L3_entire_tract_stitched_normalized.laz \
  --output-dir /path/to/ff3d_comparisons/run3_step8_th40_w2 \
  --tile-size 200.0
```

For Runs 4, 5, and 6, substitute `--output-dir`, `--radius`, `--step-size`, and `--score-th` using the table in Section 4, then run:
```bash
~/.venvs/ff3d/bin/python vendor/ForestFormer3D/benchmark_sweep.py \
  --input /path/to/20260424_LLR_L3_entire_tract_stitched_normalized.laz \
  --output-root /path/to/ff3d_comparisons \
  --eval-only
```
to scan all completed `_entire_tract_stitched.las` files and regenerate `comparison_report.md` and `sweep_results.json`.

---

## 5. How to Evaluate & Compare Results

`benchmark_sweep.py` records the following metrics per run in `sweep_results.json` and `comparison_report.md`:

1. **`stitched_trees`**: Final connected tree count after boundary stitching. Compare against Run 1 (`11,424`) and Run 2 (`10,933`) to check under-segmentation vs over-splitting.
2. **`veg_assigned_pct`**: Percentage of non-ground points (`semantic_pred >= 1`) assigned to a valid `tree_id >= 0` (Run 1: `94.07%`, Run 2: `92.26%`). Higher = fewer orphan canopy/understory vegetation points.
3. **`mean_tree_score`**: Point-weighted mean instance confidence score (Run 1: `0.7687`, Run 2: `0.7438`). Watch whether `th=0.30` (Run 5) captures real understory or drags down confidence with noisy fragments.
4. **`median_pts_per_tree`, `p25_pts_per_tree`, `p75_pts_per_tree`, and `tiny_trees_lt500pts`**: Crown size distribution. A sharp spike in `tiny_trees_lt500pts` alongside a drop in `median_pts_per_tree` indicates crown fragmentation or noisy small detections (watch Run 5 `th=0.30` and Run 6 `r=12.0m`).
5. **`wall_min`**: Total end-to-end runtime (workers + finalize/stitch). Compare runtime speedup vs quality loss relative to Run 1 (`258.5m`) and Run 2 (`66.0m`).

### Visual Verification Checklist

For the top candidate parameter sets (e.g., comparing Run 4 `s=4.0, th=0.50` vs Run 6 `r=12.0, s=4.0`), inspect:
- Dense canopy interlock zones (do adjacent mature crowns merge or stay cleanly separated?)
- Sub-canopy / suppressed saplings (how many understory stems drop between `s=4.0` and `s=8.0` or between `th=0.30` and `th=0.50`?)
- Tall dominant crowns in Run 6 (`r=12.0m` cylinder width vs `24m` diameter — check whether wide overstory crowns get clipped vertically/laterally near seams).

You can visually inspect any `_entire_tract_stitched.las` in the desktop editor:
```bash
./run_potree_editor.sh   # Open resulting LAS/LAZ and inspect ITC colors + seam continuity
```
