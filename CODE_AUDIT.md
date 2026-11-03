# PotreeEditor — Code Audit (dead code & improvements)

Date: 2026-10-30 · Scope: all 5,926 project lines (`app_window.py`, `pointcloud_manager.py`,
`copc_writer.py`, `server.py`, `potree2_patch.py`, `main.py`, `web/`), cross-checked between
layers (Qt ↔ REST ↔ JS). Method: four parallel full-file audits, findings verified with grep.

Note: there is **no QWebChannel bridge** — the bridge is one-way `page().runJavaScript("window.x(...)")`
(40 call sites) out, REST POSTs in, and `pyqtSignal`s back to Qt. "Bridge" below means that pipeline.

---

## A. Real bugs worth fixing (ranked)

| # | Severity | Finding | Where |
|---|----------|---------|-------|
| 1 | **High** | Duplicate `itc` extra-dim: `out_hdr.add_extra_dim` runs twice (`needs_add_itc` block duplicated) → `ValueError: field 'itc' occurs more than once` on the default commit path for every source lacking ITC ("Will be added on commit" flow) | [pointcloud_manager.py:232-239](app_window.py) |
| 2 | **High** | Staged `delete` edits are never applied during streaming commit (dispatch handles only assign_itc / mark_noise / set_classification) — verified 5,000 in → 5,000 out with a delete staged | [pointcloud_manager.py:316-343](pointcloud_manager.py#L316-L343) |
| 3 | **High** | No lock around shared `PointCloudManager` state: HTTP handler threads, the Qt GUI thread, and `CommitWorker(QThread)` all mutate `staged_edits`/`current_selection`/`selected_indices` concurrently | [server.py:278-279](server.py#L278-L279), app_window.py:1004, :1712 |
| 4 | **High** | Box selection tool is visual-only: `volume.addEventListener('insertion_finished')` — the shipped 1.8.2-based renderer dispatches `drop`, so the `/api/select_box` POST never fires | [web/app.js:520-525](web/app.js#L520-L525) |
| 5 | Med | `runJavaScript` fire-and-forget with no `loadFinished` gate (~30 setters): early ribbon interaction after open/reload silently drops settings; `window.set… is not a function` | app_window.py:428, :922-955, :1786-1824 |
| 6 | Med | Registry lifecycle: commit leaves `staged_edits` populated → a second commit re-applies everything; `remove_staged_edit` re-derives dirtiness from list non-emptiness | [pointcloud_manager.py:354](pointcloud_manager.py#L354), :176 |
| 7 | Med | Range-request edge cases: suffix range `bytes=-500` serves the *first* 501 bytes; no 416 (start past EOF → negative `Content-Length`); multi-range silently truncated; handler stays HTTP/1.0 → one TCP connect per octree.bin chunk | [server.py:238-261](server.py#L238-L261) |
| 8 | Med | Polygon lasso geometry: dblclick fires click first → duplicate closing vertex; `contextmenu` also fires at end of right-drag zoom → lasso finishes mid-zoom; fallback plane Z taken from the orbit pivot → vertices land on different Z planes → non-planar polygon selects full vertical columns | [web/app.js:583-586](web/app.js#L583-L586), :971-1002 |
| 9 | Med | `save_las` never calls `header.update_header()` → after deletions the bbox is the pre-delete box (copc center/halfsize derive from it) | [pointcloud_manager.py:610-627](pointcloud_manager.py#L610-L627) |
| 10 | Med | Path safety: `..` segments in URLs escape web_root/potree_root (naive `join` after one `unquote`); layer alias `abs(hash(folder)) % 100000` is collision-prone, salted per run, and unencoded into URLs | [server.py:48-90](server.py#L48-L90), app_window.py:1032, :1081, :1150 |
| 11 | Med | `CommitWorker`/QThread leak: replaced per commit, never `deleteLater()`/`wait()`; `finished` signal shadows `QThread.finished`; `run()` touches `progress_bar`/`status_bar`/`runJavaScript` off-thread | app_window.py:44-69, :1146-1159, :1712 |
| 12 | Low | `mark_noise` ITC fallback `int(val) if val else 9999` maps an explicit `itc = 0` to 9999; classification wraps > 255 into uint8 | [pointcloud_manager.py:321](pointcloud_manager.py#L321), :340, :324 |
| 13 | Low | `total_modified` double-counts overlapping edits (last edit wins, count inflated: 6,106 "modified" for 5,000 pts / 3 edits) | [pointcloud_manager.py:315](pointcloud_manager.py#L315), :334 |
| 14 | Low | Reader stays open while writer truncates output; `input_path == output_path` → partial overwrite read mid-stream; no temp+rename atomicity | [pointcloud_manager.py:217](pointcloud_manager.py#L217), :285 |
| 15 | Low | REST JSON: malformed body → `data={}` → `/api/edit` `int(None)` TypeError in a worker thread, socket closed with no status; unknown POST path → 200 `{}` | [server.py:104-107](server.py#L104-L107), :186-190, :209-213 |
| 16 | Low | Event `"saved"` emitted by server.py:205 has no branch in the Qt dispatcher → silently dropped | app_window.py:96-108 |
| 17 | Low | Patch durability: `patch()` flushes octree.bin but refreshed statistics reach disk only with `--save-metadata` → viewers show stale min/max + histogram; `--expect` compares against `changed`, so an idempotent re-run "fails"; `np.bincount` raises on signed attributes | [potree2_patch.py:207-239](potree2_patch.py#L207-L239), :370-382 |
| 18 | Low | JS hygiene: no `res.ok` checks on 4 fetch chains; `showToast` timer leak (4 toasts → earliest timer hides newest); BoxVolumes accumulate (clear removes only the last); per-frame `new THREE.Vector3` in update loop; `viewerSettings.colorMode` leaks into next loaded cloud | [web/app.js:494](web/app.js#L494), :525, :1031, :1070, :180-187, :518-539, :799, :95 |

---

## B. Dead / inactive code inventory

### Whole modules
| Item | Lines | Evidence |
|------|-------|----------|
| [copc_writer.py](copc_writer.py) — entire module unimported (`write_copc` has zero callers; conversion goes through PotreeConverter2) | 513 | grep: only its own definition |
| [potree2_patch.py](potree2_patch.py) — CLI-only; nothing imports it, so REST/UI can never patch `octree.bin`. Dead inside its own CLI: `all_records`:138, `world_xyz`:171, `field_names`:161 | 389 | zero import sites |

### Python (app_window.py / server.py / pointcloud_manager.py / main.py)
- Unused methods: `apply_delete_selection` (app_window.py:1528 — no button/shortcut ever wired),
  `load_las_for_editing` (:1117 — only the header-only path is used, so `manager.las_data` is ~always None),
  `toggle_in_viewport_settings` (:1814 — calls `window.toggleSettingsMenu()`, which doesn't exist in JS → double-dead).
- Menu bar built then hidden (`init_menubar` app_window.py:448-481, `menuBar().hide()` :116) → 5 QActions dead.
- `self.current_potree_url` (:79, written :1324) never read.
- Unused imports: `sys`,`Path`,`Dict`,`Any`,`List`,`QFont`,`QActionGroup` (app_window.py:11-27);
  `sys` (server.py:11); `Qt` (main.py:12); `Path`,`Tuple` (pointcloud_manager.py:9-10).
- MIME fallbacks for `.bin/.json/.js/.css` (server.py:226-233) unreachable on py3.14 — `mimetypes` covers all four.
- `do_OPTIONS`+CORS (server.py:35-44) — single same-origin QWebEngine client, never cross-origin.
- URL prefixes `/src/` + `/resources/` (server.py:76) — only a sourcemap comment / hidden sidebar consume them.
- pointcloud_manager: `xyz` property (:26-31, zero callers); unused params `default_val` (:572), `max_dist` (:505);
  unused locals `chunk_idx` (:286), `target_itc_field` (:239); duplicated `needs_add_itc` block (:237-239) and
  duplicated `out_hdr.vlrs` filter (:230/:235); producer-only info keys `scales`,`offsets`,`extra_dimensions`,
  `is_header_only` (:60-64, :110-111) and `indices` in select returns (:391, :414) — no consumer;
  `select_by_box_3d` staged return omits `classes`/`itc_values` (:449-454).
- REST handlers never fetched from JS (Qt mutates the manager directly): `/api/status` (server.py:51),
  `/api/remove_staged_edit` (:158), `/api/clear_staged_edits` (:170), `/api/edit` (:180), `/api/save_las` (:198).
  Keep them as public API surface, or prune — they're the only server-side consumers of several manager paths.

### Web frontend (web/app.js / index.html / style.css)
- JS never invoked: `updateAllPointClouds` (web/app.js:257), `loadStagedVisuals` (:445 — staged visuals are
  never restored on page reload, a functional gap disguised as dead code).
- JS exports with no Python caller: `getRenderMode` (:1140), `getIntersectionAt` (:1242),
  `setLayerColorAttribute` (:1246), `setLayerPointSize` (:1247), `activeLayerName` (:1249),
  `highlightPickedPoint` (:1251).
- `window.toggleSettingsMenu()` — called from Python (app_window.py:1815), implemented nowhere → silent TypeError.
- CSS dead (~170 L): `.settings-card`..`.setting-checkbox-label` block style.css:184-335 (the settings-menu
  feature it styles is dead); `.top-hud` :35, `.hud-btn.active` :72, `.hud-btn.btn-danger` :79/:85.
- `#potree_sidebar_container` (web/index.html:32) — `display:none`, never touched.
- Config triple-declaration: `viewerSettings` (web/app.js:130-143) → applied :161-168 → re-declared in
  `RENDER_MODES.performance` :1104-1105 → re-copied :1116-1121; several values are also Potree's own
  constructor defaults (edlRadius 1.4, edlStrength 1.0, edlOpacity 1.0, FOV 60).

---

## C. Improvement backlog (prioritized)

### P0 — correctness, low effort
1. **Delete the duplicate `itc` block** — keep pointcloud_manager.py:232-235, drop :237-239. Kills the
   guaranteed `ValueError` on the default commit path.
2. **Apply `delete` in streaming commit** — build one per-chunk `keep` mask in the edit loop
   (pointcloud_manager.py:298-343) and `writer.write_points(work[keep])`; accumulate `total_modified`
   into a shared `touched` mask so overlaps count once (fixes A2 + A13).
3. **Serialize manager access** — `threading.RLock` in `PointCloudManager.__init__` guarding
   add/remove/clear + selection setters; snapshot `edits = list(self.staged_edits)` at commit start
   (pointcloud_manager.py:247); clear (or version) the registry after commit so re-commits don't double-apply.
4. **`'insertion_finished'` → `'drop'`** (web/app.js:520) — one line; restores box selection end-to-end.
5. **`postJSON()` helper** in web/app.js — await + `res.ok` + `.catch` → toast; replaces 4 divergent fetch chains.
6. **Gate the ribbon on `loadFinished`** (app_window.py:428) — enable ribbon / flush queued JS calls;
   mirrors the guard pattern already used at app_window.py:1237.

### P1 — robustness & performance
7. **Range parser rewrite** (server.py:238-261): suffix ranges, 416 with `bytes */size`, anchored multi-range,
   `protocol_version = "HTTP/1.1"` for keep-alive. Highest impact on octree streaming smoothness.
8. **Confine file paths** — `os.path.normpath` + prefix check (server.py:70-90); URL-encode layer aliases and
   evict stale routes (app_window.py:1032-1035).
9. **Structured POST errors** — 400 on bad JSON/missing `value`, 404 on unknown path, per-route try/except
   (server.py:98-213).
10. **Atomic output + Z-aware polygons** — temp+rename for LAS output; store min/max-Z per polygon (the
    `points_3d` captured at server.py:113-118 is stored and never used) so crowns stop selecting whole columns;
    `header.update_header()` in `save_las`.
11. **Lasso polish** — coalesce the click preceding dblclick; suppress `contextmenu` after right-drag;
    anchor the lasso plane once per polygon (web/app.js:583-586, :971-1002).
12. **COPC memory fix (if copc_writer is kept)** — `np.bincount(flat)` allocates `(2**level)³` counters:
    measured 512 GiB at level 12 (copc_writer.py:114-115). `np.unique(flat, return_counts=True)` +
    `return_inverse` makes it linear.
13. **QThread hygiene** — `deleteLater()`/`wait()` on commit-worker replacement; rename the shadowing
    `finished` signal; marshal `start_conversion.run()` widget touches through signals.
14. **Patch as a pair** — have `potree2_patch.py patch()` (214-221) invoke `write_metadata` (250-261)
    so octree.bin + metadata.json land together; guard `bincount` for signed attributes (:238-239).

### P2 — structure (the God class)
15. **Split `MainWindow` (~70 methods, 2,558 L)** into `RibbonBar`, `LayerPanel`, `StagedEditsPanel`,
    `ViewerBridge` (all `runJavaScript` sites, app_window.py:1786-1824, :2558); MainWindow becomes wiring.
16. **Ribbon-group factory** — the `QFrame` + `QVBoxLayout(8,4,8,4)` + button-row + `ribbonTitle` block is
    copy-pasted 10× (app_window.py:1908-2391) → one `add_group(title, controls)` saves ~350 L.
17. **Collapse duplicated sync code** — slider/spin trio (:990-1002, :2463-2487) + the 4 `toggle_*_layer`
    methods (:913-955) → one `_set_layer_visibility(kind, visible)`; parameterize the 3 staged-edit actions
    (`apply_assign_itc`/`mark_noise`/`set_classification`, :1423-1526) into one function over
    (action, value, label, color).
18. **Single config source in JS** — `RENDER_MODES` as the only table, `viewerSettings` derived
    (web/app.js:130-143, :1103-1121); budget/EDL/node-size can no longer drift.
19. **Name magic numbers + type the public API** — ribbon `98/68/30`, sidebar `340/450`, slider `1..40`÷10,
    point-size clamp `0.10..4.00`, budgets (app_window.py:1832, :415, :2042, :1184, :2131); 0 of 12 bridge
    setters are annotated.
20. **Decide the dead surface** — delete `copc_writer.py` (or wire a COPC export button), prune the 170 L of
    settings-menu CSS + hidden menu bar, and pick a home for the 5 REST-only endpoints (document as public API
    in README, or remove).

---

## D. Suggested order of attack

Weekend-sized sequence: **P0 #1 → #2 → #3** (commit-path correctness, ~1 h), **#4 → #5 → #6** (bridge
robustness, ~1 h), then **#7 → #9** (server hardening, ~2 h). The P2 split last, when tests exist to
catch regressions.
