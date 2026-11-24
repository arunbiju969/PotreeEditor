import csv
import json
import os
import sys
import tempfile
import unittest
import urllib.request

import laspy
import numpy as np

# Ensure offscreen Qt platform for headless test execution
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication

app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

from crown_extractor import (
    CrownExtractionCancelled,
    SEMANTIC_GROUND,
    SEMANTIC_LEAF,
    SEMANTIC_WOOD,
    _extract_crs_info,
    compute_concave_hull_2d,
    compute_convex_hull_2d,
    export_crowns_csv,
    export_crowns_geojson,
    extract_crowns_from_arrays,
    extract_crowns_from_las,
)
from pointcloud_manager import PointCloudManager
from server import HybridServer
from app_window import MainWindow


def _build_l_shaped_crown_points(cx: float, cy: float, z_base: float, z_top: float, step: float = 0.25):
    """Build dense 2D grid points shaped like an 'L' so concave hull area < convex hull area."""
    pts = []
    # Vertical bar of L: x in [0, 1.5], y in [0, 4.0]
    for x in np.arange(0.0, 1.51, step):
        for y in np.arange(0.0, 4.01, step):
            pts.append((cx + x, cy + y))
    # Horizontal bar of L: x in [1.5, 4.0], y in [0, 1.5]
    for x in np.arange(1.75, 4.01, step):
        for y in np.arange(0.0, 1.51, step):
            pts.append((cx + x, cy + y))
    arr = np.asarray(pts, dtype=np.float64)
    z = np.linspace(z_base, z_top, len(arr))
    return arr[:, 0], arr[:, 1], z


def _create_segmented_las(path: str, include_semantic: bool = True):
    """Create a LAS file with ground (sem=0), protruding stem/twig (sem=1), and two trees (sem=2)."""
    xs, ys, zs, itcs, sems = [], [], [], [], []

    # Tree 1: L-shaped leaf crown in [335000..335004, 4982000..4982004], z in [8..18]
    tx1, ty1, tz1 = _build_l_shaped_crown_points(335000.0, 4982000.0, 8.0, 18.0)
    xs.append(tx1)
    ys.append(ty1)
    zs.append(tz1)
    itcs.append(np.full(len(tx1), 1, dtype=np.int32))
    sems.append(np.full(len(tx1), SEMANTIC_LEAF, dtype=np.int32))

    # Protruding fallen/leaning wood branch for Tree 1 at x=+10m (must be excluded in "leaf" mode)
    wx1 = np.linspace(335008.0, 335012.0, 30)
    wy1 = np.linspace(4982002.0, 4982002.5, 30)
    wz1 = np.linspace(1.0, 4.0, 30)
    xs.append(wx1)
    ys.append(wy1)
    zs.append(wz1)
    itcs.append(np.full(len(wx1), 1, dtype=np.int32))
    sems.append(np.full(len(wx1), SEMANTIC_WOOD, dtype=np.int32))

    # Tree 2: Square canopy patch + ground noise tagged with tree_id=2
    gx2, gy2 = np.meshgrid(np.linspace(335020.0, 335023.0, 8), np.linspace(4982020.0, 4982023.0, 8))
    lx2 = gx2.ravel()
    ly2 = gy2.ravel()
    lz2 = np.linspace(10.0, 22.0, len(lx2))
    xs.append(lx2)
    ys.append(ly2)
    zs.append(lz2)
    itcs.append(np.full(len(lx2), 2, dtype=np.int32))
    sems.append(np.full(len(lx2), SEMANTIC_LEAF, dtype=np.int32))

    # Ground points under Tree 2 (sem=0)
    xs.append(np.array([335021.0, 335022.0, 335028.0]))
    ys.append(np.array([4982021.0, 4982022.0, 4982028.0]))
    zs.append(np.array([0.2, 0.3, 0.4]))
    itcs.append(np.array([2, 2, 0], dtype=np.int32))
    sems.append(np.full(3, SEMANTIC_GROUND, dtype=np.int32))

    # Small Tree 3: only 4 leaf points (should be skipped when min_points=10)
    xs.append(np.array([335035.0, 335036.0, 335035.5, 335036.5]))
    ys.append(np.array([4982035.0, 4982035.0, 4982036.0, 4982036.0]))
    zs.append(np.array([6.0, 6.2, 6.4, 6.6]))
    itcs.append(np.full(4, 3, dtype=np.int32))
    sems.append(np.full(4, SEMANTIC_LEAF, dtype=np.int32))

    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)
    all_z = np.concatenate(zs)
    all_itc = np.concatenate(itcs)
    all_sem = np.concatenate(sems)

    hdr = laspy.LasHeader(point_format=3, version="1.4")
    hdr.offsets = [335000.0, 4982000.0, 0.0]
    hdr.scales = [0.01, 0.01, 0.01]
    hdr.add_extra_dim(laspy.ExtraBytesParams(name="itc", type=np.int32))
    if include_semantic:
        hdr.add_extra_dim(laspy.ExtraBytesParams(name="semantic_pred", type=np.int32))

    las = laspy.LasData(hdr)
    las.x = all_x
    las.y = all_y
    las.z = all_z
    las.itc = all_itc
    if include_semantic:
        las.semantic_pred = all_sem
    las.write(path)


class TestCrownExtractorCore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="crown_core_")
        self.las_sem = os.path.join(self._tmp.name, "seg_sem.las")
        self.las_nosem = os.path.join(self._tmp.name, "seg_nosem.las")
        _create_segmented_las(self.las_sem, include_semantic=True)
        _create_segmented_las(self.las_nosem, include_semantic=False)

    def tearDown(self):
        self._tmp.cleanup()

    def test_semantic_filtering_excludes_wood_and_ground(self):
        """Leaf-only mode strictly excludes stem/wood (sem=1) and ground (sem=0)."""
        res_leaf = extract_crowns_from_las(
            self.las_sem, semantic_mode="leaf", method="concave", alpha=1.2, min_points=10, simplify_tol=0.05
        )
        res_wood = extract_crowns_from_las(
            self.las_sem, semantic_mode="wood_and_leaf", method="convex", min_points=10, simplify_tol=0.0
        )

        self.assertEqual(res_leaf["summary"]["count"], 2)
        self.assertEqual(res_leaf["summary"]["skipped_trees"], 1)  # Tree 3 (< 10 pts)

        c1_leaf = next(c for c in res_leaf["crowns"] if c["tree_id"] == 1)
        c1_wood = next(c for c in res_wood["crowns"] if c["tree_id"] == 1)

        # Leaf-only footprint max X stays near 335004 (excluding the wood points at 335008..335012)
        max_x_leaf = max(p[0] for p in c1_leaf["coords"])
        max_x_wood = max(p[0] for p in c1_wood["coords"])
        self.assertLess(max_x_leaf, 335004.5)
        self.assertGreater(max_x_wood, 335011.0)
        self.assertAlmostEqual(c1_leaf["z_base"], 8.0, places=1)
        self.assertAlmostEqual(c1_wood["z_base"], 1.0, places=1)

    def test_concave_alpha_tighter_than_convex_on_l_shape(self):
        """Concave hull fits indented L-shape with smaller area than convex hull."""
        tx, ty, _ = _build_l_shaped_crown_points(0.0, 0.0, 5.0, 15.0)
        pts = np.column_stack((tx, ty))
        concave = compute_concave_hull_2d(pts, alpha=1.2)
        convex = compute_convex_hull_2d(pts)
        self.assertIsNotNone(concave)
        self.assertIsNotNone(convex)

        from crown_extractor import polygon_metrics

        area_concave = polygon_metrics(concave)["area_m2"]
        area_convex = polygon_metrics(convex)["area_m2"]
        self.assertLess(area_concave, area_convex * 0.85)

        # Regression: centroid and area remain exact at large UTM coordinates
        utm_sq = np.array([
            [335454.6, 4982725.6],
            [335455.6, 4982725.6],
            [335455.6, 4982726.6],
            [335454.6, 4982726.6],
        ])
        utm_m = polygon_metrics(utm_sq)
        self.assertAlmostEqual(utm_m["area_m2"], 1.0, places=6)
        self.assertAlmostEqual(utm_m["centroid_x"], 335455.1, places=6)
        self.assertAlmostEqual(utm_m["centroid_y"], 4982726.1, places=6)

    def test_streaming_matches_in_memory_and_supports_cancel(self):
        """Disk streaming (chunk_size=120) matches full RAM extraction and honors cancel_check."""
        las_mem = laspy.read(self.las_sem)
        res_mem = extract_crowns_from_las(
            self.las_sem, las_data=las_mem, semantic_mode="leaf", method="concave", alpha=1.2, min_points=10
        )
        res_stream = extract_crowns_from_las(
            self.las_sem, las_data=None, semantic_mode="leaf", method="concave", alpha=1.2, min_points=10, chunk_size=120
        )

        self.assertEqual(len(res_mem["crowns"]), len(res_stream["crowns"]))
        for a, b in zip(res_mem["crowns"], res_stream["crowns"]):
            self.assertEqual(a["tree_id"], b["tree_id"])
            self.assertEqual(a["point_count"], b["point_count"])
            self.assertAlmostEqual(a["area_m2"], b["area_m2"], delta=0.5)

        with self.assertRaises(CrownExtractionCancelled):
            extract_crowns_from_las(self.las_sem, cancel_check=lambda: True)

    def test_missing_semantic_field_validation_and_all_mode(self):
        """Requires semantic_pred for leaf mode, but works cleanly in 'all' mode."""
        with self.assertRaises(ValueError):
            extract_crowns_from_las(self.las_nosem, semantic_mode="leaf")

        res_all = extract_crowns_from_las(self.las_nosem, semantic_mode="all", min_points=10)
        self.assertEqual(res_all["summary"]["count"], 2)

        with self.assertRaises(ValueError):
            extract_crowns_from_arrays(
                np.array([0.0, 1.0, 2.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([5.0, 6.0, 7.0]),
                np.array([1, 1, 1]),
                tree_field=None,
            )

    def test_geojson_and_csv_export(self):
        """GeoJSON and CSV/WKT exports write valid closed polygons and tree metrics."""
        res = extract_crowns_from_las(self.las_sem, semantic_mode="leaf", method="concave", min_points=10)
        geo_path = os.path.join(self._tmp.name, "crowns.geojson")
        csv_path = os.path.join(self._tmp.name, "crowns.csv")

        n_geo = export_crowns_geojson(res, geo_path, include_z=True, height_mode="top")
        n_csv = export_crowns_csv(res, csv_path)
        self.assertEqual(n_geo, 2)
        self.assertEqual(n_csv, 2)

        with open(geo_path, "r", encoding="utf-8") as f:
            geo = json.load(f)
        self.assertEqual(geo["type"], "FeatureCollection")
        self.assertEqual(len(geo["features"]), 2)
        ring0 = geo["features"][0]["geometry"]["coordinates"][0]
        self.assertEqual(ring0[0], ring0[-1])  # closed polygon
        self.assertEqual(len(ring0[0]), 3)  # XYZ coordinates

        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["wkt_polygon"].startswith("POLYGON (("))
        self.assertGreater(float(rows[0]["area_m2"]), 0.0)

    def test_extract_crs_info_parses_raw_wkt_vlr(self):
        """Raw OGC WKT VLR (record_id=2112) produces a UTM proj4_hint even without pyproj."""
        class _FakeVlr:
            record_id = 2112
            string = 'PROJCRS["NAD83(CSRS) / UTM zone 20N",BASEGEOGCRS["NAD83(CSRS)"]] '
            record_data = b""

        class _FakeHdr:
            vlrs = [_FakeVlr()]
            def parse_crs(self):
                raise ModuleNotFoundError("No module named 'pyproj'")

        wkt, hint = _extract_crs_info(_FakeHdr())
        self.assertIn("UTM zone 20N", wkt)
        self.assertEqual(hint, "+proj=utm +zone=20 +datum=NAD83 +units=m +no_defs")


class TestCrownRibbonIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="crown_win_")
        cls.las_path = os.path.join(cls._tmp.name, "segmented_plot.las")
        _create_segmented_las(cls.las_path, include_semantic=True)

        cls.mgr = PointCloudManager()
        web_dir = os.path.join(os.path.dirname(__file__), "web")
        cls.srv = HybridServer(potree_root=cls._tmp.name, web_root=web_dir, manager=cls.mgr, port=18963)
        cls.srv.start()
        cls.win = MainWindow(server=cls.srv, manager=cls.mgr, potree_dir=cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.win.close()
        cls.srv.stop()
        cls._tmp.cleanup()

    def setUp(self):
        self.js_calls = []
        self._orig_run_js = self.win._run_js
        self.win._run_js = lambda js, cb=None: self.js_calls.append(js)

    def tearDown(self):
        self.win._run_js = self._orig_run_js

    def test_crown_tab_position_and_lifecycle(self):
        """Crown tab sits at ribbon index 7, updates /api/crowns, styles, and clears cleanly."""
        self.assertTrue(hasattr(self.win, "btn_tab_crown"))
        self.assertEqual(self.win.btn_tab_crown.text(), "Crown")

        # Segmentation stays at index 6; Crown is index 7
        self.win.btn_tab_segmentation.click()
        self.assertEqual(self.win.ribbon_stack.currentIndex(), 6)
        self.win.btn_tab_crown.click()
        self.assertEqual(self.win.ribbon_stack.currentIndex(), 7)

        res = extract_crowns_from_las(self.las_path, semantic_mode="leaf", min_points=10)
        self.js_calls.clear()
        self.win._on_crown_finished(res)

        self.assertTrue(self.win.btn_crown_export.isEnabled())
        self.assertTrue(self.win.btn_crown_clear.isEnabled())
        self.assertIn("2 crowns", self.win.lbl_crown_status.text())
        self.assertTrue(any("window.loadCrownPolygonsFromServer()" in c for c in self.js_calls))

        # HTTP endpoint /api/crowns returns the stored polygons
        with urllib.request.urlopen(f"http://127.0.0.1:{self.srv.port}/api/crowns") as resp:
            self.assertEqual(resp.status, 200)
            api_data = json.loads(resp.read().decode("utf-8"))
        self.assertEqual(len(api_data["crowns"]), 2)

        # Toggle display options pushes style updates
        self.js_calls.clear()
        self.win.chk_crown_labels.setChecked(True)
        self.assertTrue(any("window.setCrownPolygonsStyle(" in c and '"showLabels": true' in c for c in self.js_calls))

        # Clear empties manager store and notifies JS
        self.js_calls.clear()
        self.win.clear_crown_polygons_action(silent=True)
        self.assertIsNone(self.mgr.crown_polygons)
        self.assertFalse(self.win.btn_crown_export.isEnabled())
        self.assertTrue(any("window.clearCrownPolygons()" in c for c in self.js_calls))

    def test_crown_map_and_top_down_buttons(self):
        """2D Map View and Top-Down 3D ribbon buttons call the expected viewer JS APIs."""
        self.assertTrue(hasattr(self.win, "btn_crown_map"))
        self.assertTrue(hasattr(self.win, "btn_crown_top_view"))

        self.js_calls.clear()
        self.win.btn_crown_map.setChecked(True)
        self.assertTrue(any("window.openCrownMap(true)" in c for c in self.js_calls))

        self.js_calls.clear()
        self.win.btn_crown_map.setChecked(False)
        self.assertTrue(any("window.openCrownMap(false)" in c for c in self.js_calls))

        self.js_calls.clear()
        self.win.btn_crown_top_view.click()
        self.assertTrue(any("window.crownTopDownView()" in c for c in self.js_calls))


if __name__ == "__main__":
    unittest.main()
