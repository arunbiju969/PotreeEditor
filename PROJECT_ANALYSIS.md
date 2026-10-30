# PotreeEditor — Project Analysis

**Date:** 2026-10-30 · **HEAD:** `04bf61e` (main, tracking `origin/main`) · **Licence:** MIT © 2026 Arun Biju
**Codebase:** 4,845 lines across 6 source files (2,290 Python UI · 627 Python manager · 332 Python server · 1,091 JS · 67 HTML · 335 CSS)

---

## 1. Verdict

A genuinely ambitious, well-conceived hybrid app — PyQt6 desktop shell + embedded Chromium running Potree, backed by a Python HTTP bridge for LAS/LAZ streaming and editing. The **core architecture is sound and it runs**. The problem is maturity: the flagship write path (**commit staged edits → export LAS/LAZ**) **crashes on the most common input**, there are **no tests, no CI, and no vendored renderer**, and the two large files have outgrown their structure.

Ranked honestly:

| Dimension | Assessment |
|---|---|
| Architecture / idea | **Strong.** Header-only load + optimistic visual staging + deferred single-pass streaming commit is the right design for 100M+ point clouds. |
| Correctness of the hot path | **Weak.** The export path raises `ValueError` for any cloud that lacks an existing ITC column — i.e. every raw LiDAR flight. Reproduced. |
| Robustness / error handling | **Weak.** 4 `try`/`except` pairs in 2,290 lines of UI code; HTTP handler dies mid-response on unvalidated input. Reproduced. |
| Structure / maintainability | **Weak.** `MainWindow` is 2,221 of 2,290 lines (97%); `init_ui` alone is 338 lines; heavy duplication measured below. |
| Reproducibility | **Weak.** The WebGL renderer lives in an untracked sibling checkout with no version pin; docs and venv disagree about the Python version. |
| Security | **Acceptable for localhost, sloppy.** Unauthenticated `POST /api/save_las` writes arbitrary paths; HTTP path traversal escapes the web root. Reproduced. |

---

## 2. What it actually is

```
PyQt6 MainWindow (app_window.py, 2,290 L)
 ├── Ribbon bar (File / Point / Layers / Settings) + 3-tab sidebar
 └── QWebEngineView ──► http://127.0.0.1:<port>/index.html   (server.py)
                              │
             JS → Python: fetch POST /api/*            (4 endpoints used)
             Python → JS: runJavaScript("window.*")    (34 call sites, 26 entry points)
                              │
                    PointCloudManager (pointcloud_manager.py, 627 L)
                       laspy + NumPy + matplotlib.path
```

Two deliberate, good decisions worth calling out:

- **No QWebChannel.** The bridge is HTTP-up / `runJavaScript`-down, which keeps the JS side a normal web app and lets the same page be opened in a real browser. Cross-thread delivery is done correctly: HTTP handler threads emit `BridgeSignaler` signals ([app_window.py:34-41](app_window.py#L34-L41), [:91](app_window.py#L91), [:95-107](app_window.py#L95-L107)), so slots run on the GUI thread.
- **Header-only loading + deferred commit.** [pointcloud_manager.py:33-72](pointcloud_manager.py#L33-L72) reads only the LAS header; edits are held as lightweight polygons ([`add_staged_edit`](pointcloud_manager.py#L133-L169)) and applied in one streaming `chunk_iterator` pass ([`commit_staged_edits`](pointcloud_manager.py#L194-L364)) with a raw-integer bounding-box prefilter before the `matplotlib.path` inside-test. That is the correct way to touch a 200M-point file.

**Verified: it boots.** With `QT_QPA_PLATFORM=offscreen`, `main.py`'s import graph and `MainWindow(server, manager, potree_dir)` construct and `show()` cleanly on the project venv (Python 3.14.6, PyQt6 6.11.0 / WebEngine 6.11.0, laspy 2.7.0). `HybridServer` binds and answers `GET /api/status` with 200.

---

## 3. Defects proven by execution (not just reading)

I ran the project's own venv against synthetic LAS files and against the live server.

### P0 — Export crashes: `ValueError: field 'itc' occurs more than once` ★ the headline bug

[pointcloud_manager.py:232-239](pointcloud_manager.py#L232-L239) adds the `itc` extra dimension **twice**:

```python
232  if needs_add_itc:
233      out_hdr.add_extra_dim(laspy.ExtraBytesParams(name="itc", type=np.int32))
234      itc_fields = ["itc"]
235      out_hdr.vlrs = [v for v in out_hdr.vlrs if getattr(v, "user_id", "") != "copc"]
236
237  if needs_add_itc:                      # ← second, unconditional repeat
238      out_hdr.add_extra_dim(laspy.ExtraBytesParams(name="itc", type=np.int32))
239      target_itc_field = "itc"           # ← assigned, never read anywhere
```

`needs_add_itc` is true exactly when a staged edit is `assign_itc`/`mark_noise` **and** the source has no `itc`/`treeID`/`tree_id` field ([L224](pointcloud_manager.py#L224)) — i.e. raw LiDAR, the app's normal case. `np.dtype` then rejects the duplicated field when `PackedPointRecord.zeros` builds the output record ([L290](pointcloud_manager.py#L290)).

```
A (source WITHOUT itc)  → ValueError: field 'itc' occurs more than once
C (source WITH itc)     → OK, dims ['itc'], value 42 written, 1000/1000 modified
```

So **the advertised workflow — open raw LAZ, draw crowns, stage Tree IDs, export — fails**, and the user sees only `"Commit Failed"` with the numpy message ([app_window.py:1557-1561](app_window.py#L1557-L1561)). Deleting L237-239 fixes it. The repro is kept at [`.dsh-acl-repair/repro_commit.py`](.dsh-acl-repair/repro_commit.py).

### P1 — Staged "delete" edits are silently no-ops

`delete` is a first-class staged action (labelled "Deleted", [L161-162](pointcloud_manager.py#L161-L162)) but the commit loop only handles `assign_itc`, `mark_noise`, `set_classification` ([L316-343](pointcloud_manager.py#L316-L343)). Repro B:

```
B (staged delete) → success, modified_points: 1000 | output point count: 1000 (source was 1000)
```

The user is told 1,000 points were modified; zero were deleted. Also note `total_modified` double-counts a point matched by two overlapping edits.

### P1 — Range-request handling is wrong in two ways

[server.py:240-246](server.py#L240-L246) against the live server serving `potree.js` (2,392,363 bytes):

| Request | Got | Should be |
|---|---|---|
| `bytes=-1023` | `206`, `Content-Range: bytes 0-1023/2392363` | last 1023 bytes → `2391340-2392362` |
| `bytes=999999999-1000001000` | `206`, `Content-Length: -997607636`, then `ValueError: read length must be non-negative or -1` at [server.py:260](server.py#L260) | `416 Range Not Satisfiable` |

A suffix range (`bytes=-N`, RFC 7233 §14.1.1) is silently served as a *prefix*, and an unsatisfiable range kills the handler thread after headers are already flushed. `end < start` is never checked. Potree mostly asks for open-ended ranges so this is latent, not yet loud — but it is a data-corruption-class bug for anything else that hits the server.

### P1 — Unvalidated POST kills the connection instead of returning an error

[server.py:180-196](server.py#L180-L196) does `int(val)` with no guard and no `try` (only `/api/save_las` is wrapped, [L201-207](server.py#L201-L207)). Live result:

```
POST /api/edit {"action":"assign_itc"}   → RemoteDisconnected ("no response")
POST /api/unknown_endpoint {"x":1}       → 200 {}          ← not 404, not 400
```

The `TypeError` escapes `do_POST`, so `send_response` never runs. Combined with the JS side never checking `res.ok` ([web/app.js:352](web/app.js#L352), [:383](web/app.js#L383), [:728](web/app.js#L728), [:889](web/app.js#L889), [:928](web/app.js#L928)) and no `.catch` on [web/app.js:817-821](web/app.js#L817-L821), every backend failure surfaces as a confusing JSON-parse error or nothing at all.

### P1 — The HTTP server escapes its roots and leaks absolute paths

No containment check: [server.py:70-73](server.py#L70-L73), [:76-80](server.py#L76-L80), [:87-90](server.py#L87-L90) feed unquote'd, dot-segment-bearing paths straight into `os.path.join`. Proven:

```
GET /web/../requirements.txt                 → 200 (real file, outside web_root)
GET /build/../README.md                      → 200 (Potree's README)
GET /libs/../package.json                    → 200
GET /web/../../../Windows/win.ini            → 404 "File not found: D:\Projects\PotreeEditor\web\..\..\..\Windows/win.ini"
```

Two consequences: an arbitrary-read surface limited to the drive layout, and **absolute server-side path disclosure in 404 bodies** ([server.py:220](server.py#L220)). Since `Access-Control-Allow-Origin: *` is sent on everything ([server.py:36-39](server.py#L36-L39)) and `POST /api/save_las` accepts a client-supplied path ([server.py:198-207](server.py#L198-L207)) with no allow-list, **any web page this machine visits can ask the server to write files**. Fine for a private desktop tool; a real risk if it ever ships.

### P2 — `mimetypes` decides whether the app loads at all

[server.py:224-235](server.py#L224-L235) trusts `mimetypes.guess_type` and only fixes up `.js` when it returns `None`. On Windows the registry wins; if it reports `text/plain`, `<script type="module" src="/web/app.js">` ([web/index.html:65](web/index.html#L65)) is rejected by the strict module MIME check and the whole app is a black canvas. Here it happened to resolve to `text/javascript`, so this is a latent Windows-portability trap rather than a live failure.

---

## 4. Defects proven by inspection (cross-checked with grep)

**Bridge is unguarded.** `loadFinished` appears **0** times in the project ([app_window.py:426-427](app_window.py#L426-L427) just calls `load()`). Every `runJavaScript` before the document is ready is silently dropped — e.g. opening a LAS immediately after launch loses `loadPointCloudUrl` ([app_window.py:1129-1132](app_window.py#L1129-L1132)). There is also no `javaScriptConsoleMessage` override, so all JS-side failures are invisible.

**Escaping is inconsistent.** Object payloads correctly use `json.dumps` ([L1243](app_window.py#L1243), [:1277](app_window.py#L1277), [:1313](app_window.py#L1313)), but these splice raw text into a JS string literal:

- [L1131](app_window.py#L1131) `f"window.loadPointCloudUrl('{url}');"` — `url` embeds a user-picked filename, so `farmer's stand/metadata.json` produces a `SyntaxError` and the cloud never loads (while [L1009-1012](app_window.py#L1009-L1012) have already stamped "loaded" into the labels).
- [L953](app_window.py#L953), [:984](app_window.py#L984), [:1459](app_window.py#L1459) splice `edit_id`, and the manager trusts a client-supplied id ([pointcloud_manager.py:138-139](pointcloud_manager.py#L138-L139)), so a local POST can inject JS into the page.

**One bridge function does not exist.** `window.toggleSettingsMenu` is called at [app_window.py:1618](app_window.py#L1618) and defined nowhere in `app.js` (grep: 2 hits project-wide — the call, and `.settings-card` in CSS). That single missing function is the tip of a dead chain: [style.css:184-335](web/style.css#L184-L335) — **≈45% of the stylesheet** — styles a settings card that `index.html` never creates.

**Threading: one real violation.** `CommitWorker` is correct (signals, created on the GUI thread → queued delivery). But `start_conversion`'s nested `run()` on a raw `threading.Thread` ([app_window.py:1112-1127](app_window.py#L1112-L1127)) touches widgets directly from the worker — `progress_bar.setVisible(False)` [L1124](app_window.py#L1124), `status_bar.showMessage` [L1125](app_window.py#L1125) — and calls `runJavaScript` via [L1120](app_window.py#L1120), which QtWebEngine requires on the GUI thread. It also only `print`s the converter's stderr on failure ([L1122](app_window.py#L1122)) while still reporting "Conversion finished." There is no `try` around `subprocess.run`, so a launch error leaves the indeterminate progress bar stuck forever.

**Commit lifecycle gaps** (all grep-confirmed 0 hits: `cancelRequested`, `closeEvent`, `deleteLater`, `isRunning()`, `.wait()`):

- The **Cancel button does nothing** — the dialog is built with `"Cancel"` ([L1507](app_window.py#L1507)) with `setAutoReset(False)` ([L1511](app_window.py#L1511)) and no `canceled`/`cancelRequested` handler, so the file keeps being written.
- Progress scale is wrong when only a converted cloud is open: `total_pts` defaults to **100** ([L1506](app_window.py#L1506)) while the worker reports millions → bar pegs at 100% immediately.
- A second commit reassigns `self.commit_worker` ([L1515](app_window.py#L1515)) with no `isRunning()` guard → possible "QThread: Destroyed while thread is still running".
- `CommitWorker.finished` ([L45](app_window.py#L45)) **shadows the inherited `QThread.finished`**, removing the natural cleanup hook.
- **No `closeEvent`** → closing the window discards staged edits and kills a mid-flight export with no prompt.
- Commit never clears `staged_edits`, so a second export silently re-applies the same polygons to a different output file ([manager L354-364](pointcloud_manager.py#L354-L364), [app_window.py:1527-1552](app_window.py#L1527-L1552)).

**Shared state is unguarded.** `PointCloudManager` is mutated from three directions — GUI thread, per-request HTTP threads, and `CommitWorker` — with no lock. Concretely: `commit_staged_edits` iterates `self.staged_edits` ([manager L247](pointcloud_manager.py#L247)) while `remove_staged_edit` **rebinds** the list ([L174](pointcloud_manager.py#L174)) or `clear` empties it ([L183](pointcloud_manager.py#L183)), and the result reports `edits_count` from the *new* list ([L362](pointcloud_manager.py#L362)) → the table and the written file disagree. Also `load_las_header` resets `las_path`/`las_data`/`selected_indices` but **not** `staged_edits` or `has_unsaved_changes` ([L67-70](pointcloud_manager.py#L67-L70)) → polygons traced on file A get committed into file B.

**`hash()`-derived route aliases.** `alias = "cloud_" + str(abs(hash(folder)) % 100000)` at [L1003](app_window.py#L1003), [:1047](app_window.py#L1047), [:1116](app_window.py#L1116): `str.__hash__` is salted per process (non-deterministic URLs), and with 10⁵ buckets a collision silently re-points an already-loaded cloud at the wrong directory. The route dict is a **class attribute** ([server.py:27](server.py#L27)) that is never pruned.

**Layer state desyncs.** Qt checkboxes flip JS visibility flags ([app_window.py:883-939](app_window.py#L883-L939)) but nothing stores the intent JS-side, and `loadPointCloudUrl` ([web/app.js:128-165](web/app.js#L128-L165)) never re-applies it — so after opening a second dataset the sidebar lies. `window.loadStagedVisuals` ([web/app.js:303-310](web/app.js#L303-L310)) exists for exactly this and is called 0 times from Python.

**Two sources of truth for the Tree ID counter:** JS auto-increment ([web/app.js:899](web/app.js#L899)) vs Qt `spin_itc.setValue(itc_val + 1)` ([app_window.py:1245](app_window.py#L1245)) — and the whole staging flow is implemented twice (JS [871-946](web/app.js#L871-L946) vs Python [1226-1330](app_window.py#L1226-L1330)).

**3D staging loses Z on one of its two paths.** JS sends `points_3d` ([web/app.js:355](web/app.js#L355)); the server stores it ([server.py:115-119](server.py#L115-L119)); `select_by_polygon_2d` immediately overwrites `current_selection` **without** it ([pointcloud_manager.py:373-376](pointcloud_manager.py#L373-L376)), so ribbon-staged outlines always fall back to camera-Z placement ([web/app.js:226-228](web/app.js#L226-L228)) while HUD-staged outlines use real Z. Same button, two behaviours.

**Colour-mode combos disagree.** The ribbon offers `itc` ([app_window.py:2186](app_window.py#L2186)); the sidebar does not ([L567-570](app_window.py#L567-L570)), so `findData` returns −1 and the sync is skipped ([L2275-2276](app_window.py#L2275-L2276)) — the viewport colours by ITC while the Layer Manager still reads "RGBA".

**Silent error swallowing.** `load_las_header_only` catches everything into a `print` ([L1080-1081](app_window.py#L1080-L1081)) after partially writing the labels, whereas its sibling `load_las_for_editing` shows `QMessageBox.critical` for the same failure class ([L1099-1100](app_window.py#L1099-L1100)).

**Dead code (grep-verified: exactly one occurrence = the definition):** `load_las_for_editing` [L1083](app_window.py#L1083), `apply_delete_selection` [L1331](app_window.py#L1331), `toggle_in_viewport_settings` [L1617](app_window.py#L1617); `current_potree_url` written twice, read never; `has_unsaved_changes` is maintained by the manager but **read 0 times** in `app_window.py` (only by `/api/status`); `target_itc_field`; `currentSelectionStats`, `viewerSettings.splatQuality`, `window.getIntersectionAt`, `window.highlightPickedPoint`; unused imports `sys`, `Path`, `QFont`, `QActionGroup`; four server endpoints (`/api/remove_staged_edit`, `/api/clear_staged_edits`, `/api/edit`, `/api/save_las`) have no JS caller; 11 of 12 Potree-pulled libraries (jQuery, d3, proj4, i18next, jstree, spectrum, TWEEN, laslaz, …) are touched by nothing in `app.js`.

**GPU leaks in JS:** cloud removal uses `viewer.scene.scenePointCloud.remove(...)` instead of `viewer.scene.removePointCloud` ([web/app.js:138](web/app.js#L138)), leaving the old cloud in `viewer.scene.pointclouds` (which drives picking and bulk updates) with no material dispose; `clearSelection` drops `polygonLine` without `dispose()` ([web/app.js:796-799](web/app.js#L796-L799)). The toast timer is never cleared ([web/app.js:63-70](web/app.js#L63-L70)), so the 3 s "loading" toast hides the success and even the 5 s error toast. `handleSelectionResult` prints "Polygon Delineated" for **box** results because `select_by_box_3d` also returns `mode: "staged"` without `vertex_count` ([pointcloud_manager.py:448-454](pointcloud_manager.py#L448-L454), [web/app.js:772-773](web/app.js#L772-L773)).

---

## 5. Structural assessment

**Every meaningful decision lives in one class.** 3 classes, 77 `def`s; `MainWindow` spans [L70-2290](app_window.py#L70) = **2,221 lines (97% of the file)**. Five largest functions:

| Function | Span | Code lines |
|---|---|---|
| `init_ui` ([L109](app_window.py#L109)) | 338 | 315 (≈290 of them are one inline stylesheet string, [L118-398](app_window.py#L118-L398)) |
| `create_layer_manager_tab` ([L505](app_window.py#L505)) | 221 | 176 |
| `_create_ribbon_point_page` ([L1789](app_window.py#L1789)) | 203 | 165 |
| `_create_ribbon_settings_page` ([L2087](app_window.py#L2087)) | 132 | 110 |
| `create_delineation_tab` ([L726](app_window.py#L726)) | 108 | 82 |

**Duplication, measured:** three near-identical ~35-line staging blocks ([1226-1261](app_window.py#L1226-L1261) / [1262-1294](app_window.py#L1262-L1294) / [1295-1330](app_window.py#L1295-L1330)); four copies of the same 12-line "block signals, toggle checkbox, call JS" pattern ([883-926](app_window.py#L883-L926)); a byte-identical 14-line delete-button stylesheet duplicated at [L1387-1400](app_window.py#L1387-L1400) and [L1440-1453](app_window.py#L1440-L1453); size-value sync written three times ([965-978](app_window.py#L965-L978), [2233-2244](app_window.py#L2233-L2244), [2246-2257](app_window.py#L2246-L2257)); colour sync twice; `show_all_layers`/`hide_all_layers` send 4 calls and then `setAllLayersVisibility`, which re-sends the same 4 → 8 redundant round-trips and 5 toasts per click.

**Fragile construction order.** 39 `hasattr()` guards defend sibling widgets, while equally cross-referenced attributes are used unguarded (`self.web_view` in every JS call). `create_ribbon_bar()` runs at [L407](app_window.py#L407) and `create_sidebar()` at [L413](app_window.py#L413) — both *before* `self.web_view` exists at [L418](app_window.py#L418). Nothing crashes today only because every `connect()` is deliberately placed after its programmatic `setValue`/`setChecked`. Reorder one line and it's an `AttributeError`.

**A hidden menu that is also dead UI.** `init_menu_bar` builds 5 actions ([L447-480](app_window.py#L447-L480)) and then `self.menuBar().setVisible(False)` ([L115](app_window.py#L115)) — every action duplicates a ribbon button, though Qt may keep the shortcuts live.

**O(N) GUI-thread churn.** `refresh_staged_edits_table` ([L1360-1456](app_window.py#L1360-L1456)) rebuilds **both** tables from scratch — ~2 widgets, 1 `QPixmap`, 1 `QIcon`, 2 stylesheet parses per row — on every add/remove/toggle/commit, and row↔edit binding is positional ([L949-953](app_window.py#L949-L953)), so a HUD-side add mid-session makes double-click zoom to the wrong tree.

**No coordinate/unit abstraction.** `proj4.js` is loaded ([web/index.html:21](web/index.html#L21)) and never used; there is no EPSG handling anywhere. JS treats Potree scene units as LAS world units and every decoration is hardcoded in dataset units (marker radius `0.35`, label lift `+1.5`, `max_dist=2.0` in [pointcloud_manager.py:505](pointcloud_manager.py#L505)) — ~100× wrong on a centimetre- or feet-scaled cloud. Polygon queries are 2D-only, so the `points_3d` Z is transported and then ignored.

**Git health:** 6 commits, one author, all within 2 days; the last added 2,099 lines to `app_window.py` in one shot. 12 tracked files, no tests, no CI, no linter config, no `pyproject.toml`.

---

## 6. Reproducibility and portability

This is the part most likely to bite, because it fails silently.

- **The renderer is not in the repo.** `index.html` loads 16 root-relative assets from `/build/` and `/libs/` and **nothing is CDN-hosted and nothing is vendored** — they resolve against the local server, which maps them onto an **external Potree 1.8 checkout** ([main.py:31-40](main.py#L31-L40), [server.py:76-80](server.py#L76-L80)). Only jQuery carries a version string; `potree.js` and `three.module.js` are unversioned, yet `app.js` hard-depends on Potree-1.8-shaped internals (`viewer.volumeTool`, `Potree.Utils.getMousePointCloudIntersection`, `Potree.BoxVolume`, `pc.getAttributes().attributes`). **Any Potree revision change can break the app with no signal.**
- **The fallback is a personal path.** [main.py:40](main.py#L40) ends the search at `C:\Users\w0491597\Documents\Projects\potree` with no existence check and no error → a fresh clone shows a blank black canvas, not a message. On this machine `D:\Projects\potree` does exist and is picked up by the `app_dir.parent / "potree"` branch, so it works here by luck of layout.
- **Docs drift from reality.** [README.md:75](README.md#L75) claims the venv is Python **3.12**; `.venv/pyvenv.cfg` says **3.14.6**. [README.md:65](README.md#L65) and [:70](README.md#L70) say `E:\Projects\PotreeEditor`; the checkout is on `D:`. `requirements.txt` has no upper bounds on laspy/numpy/scipy/matplotlib — and the P0 crash above is exactly the kind of thing an unpinned `laspy[lazrs]` invites.
- **`run_potree_editor.bat` falls back to a sibling project's venv** (`..\TreeAIBox\.venv`, [L15-17](run_potree_editor.bat#L15-L17)) — convenient, and a portability trap for anyone without that repo.
- **The converter path is environment-locked**: `potree_dir/libs/PotreeConverter2/PotreeConverter.exe` ([app_window.py:1103](app_window.py#L1103)); if missing you get a hard "Converter Not Found" dialog and conversion is simply unavailable.

---

## 7. What I'd fix, in order

**Stop the bleeding (hours, high confidence):**
1. Delete the duplicate `add_extra_dim` block — [pointcloud_manager.py:237-239](pointcloud_manager.py#L237-L239) — and drop the never-read `target_itc_field`. This un-breaks the flagship export. *(reproduced, fix is a delete)*
2. Decide `delete` semantics: either handle it in the commit loop or remove it from the staged vocabulary. Today it lies. *(reproduced)*
3. Wrap `do_POST` body handling in one `try`/`except` returning `400`/`500` with a JSON body, and return `404` for unmatched paths ([server.py:104-213](server.py#L104-L213)). Then add `res.ok` + `.catch` to all five JS fetches.
4. Fix the range parser: handle suffix ranges, reject `end < start`, return 416 for unsatisfiable ranges ([server.py:238-261](server.py#L238-L261)). Stop echoing absolute paths in 404 bodies.
5. Gate all `runJavaScript` on `loadFinished`, and add a `javaScriptConsoleMessage` override so bridge errors stop being invisible.

**Cheap correctness wins:** `json.dumps`-escape the URL and `edit_id` interpolations; add a `threading.RLock` to `PointCloudManager` (or snapshot `staged_edits` at commit start); reset `staged_edits` + `has_unsaved_changes` in `load_las_header`; replace `hash()` aliases with `secrets.token_hex`/`uuid4` and prune routes; make Cancel actually stop the worker and fix the `total_pts = 100` default; add `closeEvent`; convert `start_conversion`'s thread to signals; use `viewer.scene.removePointCloud` + `dispose()`; clear the toast timer.

**Structural (do incrementally, behind the existing behaviour):**
6. Add a smoke test that round-trips `commit_staged_edits` on a synthetic 1k-point LAZ with and without an existing `itc` dim. That one file would have caught the P0, the silent `delete`, and the double-count. There are currently **zero** tests.
7. Extract `PointCloudManager`'s geometry-compilation out of `commit_staged_edits`, then split `MainWindow` into ribbon / sidebar / bridge / commit modules; move the 290-line stylesheet string into a `.qss` resource.
8. Extract the Potree asset tree into `web/vendor/` (or a submodule + `POTREE_DIR` checked loudly at startup) and pin it.
9. Reconcile docs and pin the environment: fix `README.md` (D:, Python 3.14) or the venv; add `pyproject.toml` with bounds; add a CI job that imports `main`, constructs `MainWindow` offscreen, and runs the commit round-trip.
10. Delete the dead chain in one PR: `toggleSettingsMenu` + `style.css:184-335` + `toggle_in_viewport_settings` + the unused libraries in `index.html`.

**Not yet urgent** but worth a tracked issue: unit-aware rendering (no EPSG/scale abstraction), the CORS + arbitrary-`save_las`-path exposure, and the colour-combo mismatch.

---

## 8. One-line summary

Right architecture, wrong maturity: **the streaming-commit engine that makes this project interesting is currently broken by three duplicated lines** ([pointcloud_manager.py:237-239](pointcloud_manager.py#L237-L239)), and nothing in the repo — no tests, no CI, no pinned renderer — would have noticed.
