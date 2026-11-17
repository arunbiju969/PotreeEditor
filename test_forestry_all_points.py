import os
import sys
import unittest
import numpy as np
import laspy

# Ensure offscreen Qt platform for headless test execution
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

from pointcloud_manager import PointCloudManager
from server import HybridServer
from app_window import MainWindow


class TestForestryAllPoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "/tmp/test_forestry_all_points"
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.las_path = os.path.join(cls.test_dir, "multi_tree_sample.laz")

        # Create synthetic LAS file with 3 distinct trees (ITC 10, 20, 30) plus ground (ITC 0)
        n = 1000
        hdr = laspy.LasHeader(point_format=3, version="1.4")
        hdr.offsets = [100.0, 200.0, 10.0]
        hdr.scales = [0.01, 0.01, 0.01]
        hdr.add_extra_dim(laspy.ExtraBytesParams(name="itc", type=np.int32))

        las = laspy.LasData(hdr)
        las.x = np.linspace(100, 150, n)
        las.y = np.linspace(200, 250, n)
        las.z = np.linspace(10, 40, n)
        las.intensity = np.full(n, 500, dtype=np.uint16)

        # 4 groups: 250 ground (class 2, itc 0), 250 tree 10, 250 tree 20, 250 tree 30 (class 5)
        cls_arr = np.full(n, 5, dtype=np.uint8)
        cls_arr[:250] = 2
        las.classification = cls_arr

        itc_arr = np.zeros(n, dtype=np.int32)
        itc_arr[250:500] = 10
        itc_arr[500:750] = 20
        itc_arr[750:1000] = 30
        las.itc = itc_arr

        las.red = np.full(n, 12000, dtype=np.uint16)
        las.green = np.full(n, 34000, dtype=np.uint16)
        las.blue = np.full(n, 8000, dtype=np.uint16)

        las.write(cls.las_path)

        cls.mgr = PointCloudManager()
        web_dir = os.path.join(os.path.dirname(__file__), "web")
        cls.srv = HybridServer(potree_root=cls.test_dir, web_root=web_dir, manager=cls.mgr, port=18955)
        cls.win = MainWindow(server=cls.srv, manager=cls.mgr, potree_dir=cls.test_dir)
        cls.win.show()

    def setUp(self):
        self.win._js_calls = []
        self.win._orig_run_js = self.win._run_js
        self.win._run_js = lambda js: self.win._js_calls.append(js)
        self.mgr.las_path = self.las_path
        self.mgr.las_data = None

    def tearDown(self):
        self.win._run_js = self.win._orig_run_js

    def test_bring_all_points_button_removed(self):
        """The "Bring in All Points (Full Cloud)" button was removed from the Forestry tab.

        The backend capability (/api/get_all_points + manager.get_all_points) is still
        exercised by the tests below; only the UI entry point is gone.
        """
        self.assertFalse(hasattr(self.win, "btn_bring_all_points"))
        self.assertFalse(hasattr(self.win, "view_all_points_action"))

    def test_next_tree_button_advances_to_next_existing_id(self):
        """Next Tree steps to the next real ITC, skipping gaps, and wraps at the end."""
        self.assertTrue(hasattr(self.win, "btn_next_tree"))
        self.assertIn("Next Tree", self.win.btn_next_tree.text())

        original_index = self.mgr._tree_index
        try:
            # Deliberate gaps: 10 -> 20 -> 45 (30 is absent).
            self.mgr._tree_index = {10: [0], 20: [1], 45: [2]}

            self.win.spin_tree_view.setValue(10)
            self.win.next_tree_action()
            self.assertEqual(self.win.spin_tree_view.value(), 20)

            self.win.next_tree_action()
            self.assertEqual(self.win.spin_tree_view.value(), 45, "must skip the missing ID 30")

            # Past the last tree it wraps back to the first.
            self.win.next_tree_action()
            self.assertEqual(self.win.spin_tree_view.value(), 10)
            self.assertIn("wrapped", self.win.lbl_tree_view_status.text().lower())
        finally:
            self.mgr._tree_index = original_index

    def test_auto_review_visits_every_tree_as_a_single_tree(self):
        """Auto review walks 1..N and asks for one tree at a time, never its stand.

        Regression: auto mode used to route through view_tree_action() with the
        Include Neighbors setting still applied, so every step rendered the
        surrounding neighbour trees instead of the tree being reviewed.
        """
        original_index = self.mgr._tree_index
        original_neighbors = self.win.chk_tree_neighbors.isChecked()
        try:
            self.mgr._tree_index = {1: [0], 2: [1], 3: [2], 4: [3]}
            # Neighbours ON: auto mode must ignore this and still ask for one tree.
            self.win.chk_tree_neighbors.setChecked(True)
            self.win._js_calls.clear()

            self.win.start_auto_review()
            self.assertTrue(self.win._auto_active)
            self.assertEqual(self.win.spin_tree_view.value(), 1, "a run starts at the first tree")

            # Drive the walk to completion without waiting on the timer.
            for _ in range(10):
                if not self.win._auto_active:
                    break
                self.win._auto_timer.stop()
                self.win._auto_step()

            self.assertFalse(self.win._auto_active, "walk should finish at the last tree")
            self.assertEqual(self.win.spin_tree_view.value(), 4)
            self.assertIn("Finished", self.win.lbl_auto_status.text())

            calls = [c for c in self.win._js_calls if "viewTreeById" in c]
            self.assertEqual(len(calls), 4, f"expected one view per tree, got {calls}")
            for expected_itc, call in zip((1, 2, 3, 4), calls):
                self.assertIn(f"viewTreeById({expected_itc},", call)
                # radius 0 + ground 0 + full=false => just this tree's own points.
                self.assertIn(", 0.0, 'all', 0.0,", call,
                              f"tree #{expected_itc} was queried with a stand radius: {call}")
                self.assertTrue(call.rstrip().endswith("false);"), f"full neighbours leaked: {call}")
        finally:
            self.mgr._tree_index = original_index
            self.win.chk_tree_neighbors.setChecked(original_neighbors)
            if self.win._auto_active:
                self.win.stop_auto_review()

    def test_get_all_points_manager(self):
        """Verify get_all_points retrieves full point cloud payload with correct structure."""
        payload = self.mgr.get_all_points()
        self.assertIsNotNone(payload)
        self.assertEqual(payload["count"], 1000)
        self.assertTrue(payload["is_all_points"])
        self.assertEqual(payload["itc"], 0)
        self.assertIn("xyz", payload)
        self.assertEqual(len(payload["xyz"]), 1000)
        self.assertIn("scalars", payload)
        self.assertIn("itc", payload["scalars"])
        self.assertEqual(len(payload["scalars"]["itc"]), 1000)
        self.assertIsNotNone(payload["classification"])
        self.assertIsNotNone(payload["rgb"])

    def test_in_memory_selection_after_get_all_points(self):
        """Verify that bringing in all points populates las_data and enables polygon & box selection."""
        self.mgr.get_all_points()
        self.assertIsNotNone(self.mgr.las_data)

        # 3D Box select a subregion (e.g. tree 10: points 250..500)
        min_pt = [112.0, 212.0, 15.0]
        max_pt = [125.0, 226.0, 26.0]
        box_res = self.mgr.select_by_box_3d(min_pt, max_pt)
        self.assertGreater(box_res["count"], 0)
        self.assertIn(10, box_res["itc_values"])

        # 2D Polygon select
        poly = [[110.0, 210.0], [130.0, 210.0], [130.0, 230.0], [110.0, 230.0]]
        poly_res = self.mgr.select_by_polygon_2d(poly)
        self.assertGreater(poly_res["count"], 0)
        self.assertIn(10, poly_res["itc_values"])

    def test_stand_loaded_all_points_populates_list_and_enables_filter(self):
        """Verify tree_stand_loaded with is_all_points=True shows Stand Filter and lists all trees."""
        self.win.sidebar_tabs.setCurrentIndex(1)  # Forestry tab
        data = {
            "target_itc": 0,
            "count": 1000,
            "target_count": 0,
            "neighbor_count": 1000,
            "radius": 0.0,
            "is_all_points": True,
            "trees": [
                {"id": 10, "isTarget": False, "count": 250},
                {"id": 20, "isTarget": False, "count": 250},
                {"id": 30, "isTarget": False, "count": 250},
                {"id": 0, "isTarget": False, "count": 250}
            ]
        }
        self.win.signaler.tree_stand_loaded.emit(data)
        # The pick controls live on the ribbon Filter page now, not in the sidebar:
        # the card stays hidden while the list is populated and the toolbar arms.
        self.assertTrue(self.win.review_panel.isAncestorOf(self.win.list_stand_trees))
        self.assertTrue(self.win.btn_tree_filter_mode.isEnabled())
        self.assertEqual(self.win.list_stand_trees.count(), 4)
        self.assertIn("All Points", self.win.lbl_top_stand_info.text())

        # Test Clear button
        self.win._on_filter_select_all()
        for i in range(self.win.list_stand_trees.count()):
            self.assertEqual(self.win.list_stand_trees.item(i).checkState(), Qt.CheckState.Checked)

        self.win._on_filter_select_none()
        for i in range(self.win.list_stand_trees.count()):
            self.assertEqual(self.win.list_stand_trees.item(i).checkState(), Qt.CheckState.Unchecked)

    def test_export_all_points_with_filter(self):
        """Verify export_tree_points with is_all_points and keep_tree_ids filters properly."""
        out_path = os.path.join(self.test_dir, "filtered_out.laz")
        # Keep only tree 10 and 20 (500 points total)
        cnt = self.mgr.export_tree_points(out_path, 0, keep_tree_ids=[10, 20], is_all_points=True)
        self.assertEqual(cnt, 500)
        self.assertTrue(os.path.exists(out_path))

        out_las = laspy.read(out_path)
        self.assertEqual(len(out_las), 500)
        self.assertTrue(all(itc in (10, 20) for itc in np.unique(out_las.itc)))

    def test_tree_loaded_when_tree_view_state_is_none(self):
        """Verify _on_tree_loaded does not crash when _tree_view_state is None."""
        self.win._tree_view_state = None
        data = {
            "itc": 0,
            "count": 2000000,
            "total_points": 213325437,
            "target_count": 0,
            "neighbor_count": 2000000,
            "radius": 0.0,
            "is_all_points": True,
            "trees": []
        }
        # Should execute cleanly without AttributeError
        self.win.on_tree_loaded(data)
        self.assertIsNotNone(self.win._tree_view_state)
        self.assertEqual(self.win._tree_view_state["itc"], 0)
        self.assertEqual(self.win._tree_view_state["is_all_points"], True)
        self.assertIn("displayed in 3D", self.win.lbl_tree_view_status.text())

    def test_decimation_for_large_cloud(self):
        """Verify that get_all_points caps display points to max_display_points while preserving full data in RAM."""
        # Create a mock/synthetic cloud with 10,000 points
        n = 10000
        hdr = laspy.LasHeader(point_format=3, version="1.4")
        hdr.add_extra_dim(laspy.ExtraBytesParams(name="itc", type=np.int32))
        las = laspy.LasData(hdr)
        las.x = np.linspace(0, 100, n)
        las.y = np.linspace(0, 100, n)
        las.z = np.linspace(0, 50, n)
        las.itc = np.repeat(np.arange(1, 11, dtype=np.int32), 1000)  # 10 trees of 1000 pts
        las.classification = np.full(n, 5, dtype=np.uint8)

        mgr = PointCloudManager()
        mgr.las_data = las

        # Request with max_display_points = 2000
        payload = mgr.get_all_points(max_display_points=2000)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["total_points"], 10000)
        self.assertLessEqual(payload["count"], 2000)
        self.assertEqual(payload["display_points"], payload["count"])
        self.assertEqual(len(payload["xyz"]), payload["count"])
        # Ensure RAM point cloud was not downsampled
        self.assertEqual(len(mgr.las_data), 10000)

        # Trees summary should still accurately report full counts (1000 per tree)
        trees = payload.get("trees_summary", [])
        self.assertEqual(len(trees), 10)
        for t in trees:
            self.assertEqual(t["count"], 1000)

        # Exporting without filter should write all 10,000 points, not just the 2000 displayed
        out_path = os.path.join(self.test_dir, "full_export.laz")
        cnt = mgr.export_tree_points(out_path, 0, is_all_points=True)
        self.assertEqual(cnt, 10000)
        out_las = laspy.read(out_path)
        self.assertEqual(len(out_las), 10000)

    def test_binary_server_metadata_with_all_points(self):
        """Verify that _send_tree_binary serializes total_points, display_points, and trees summary."""
        import io
        import struct
        import json

        payload = self.mgr.get_all_points()
        mock_wfile = io.BytesIO()

        class MockHandler:
            def __init__(self, wfile):
                self.wfile = wfile
            def send_response(self, code): pass
            def send_header(self, k, v): pass
            def send_cors_headers(self): pass
            def end_headers(self): pass

        from server import HybridRequestHandler
        handler = MockHandler(mock_wfile)
        HybridRequestHandler._send_tree_binary(handler, payload)

        raw = mock_wfile.getvalue()
        self.assertGreater(len(raw), 64)
        magic, version, count, flags = struct.unpack_from("<4sIII", raw, 0)
        self.assertEqual(magic, b"TREE")
        self.assertEqual(count, payload["count"])
        self.assertTrue(bool(flags & 8))  # scalars / metadata flag

        meta_len = struct.unpack_from("<I", raw, 64)[0]
        meta_json_bytes = raw[68:68 + meta_len]
        meta = json.loads(meta_json_bytes.decode("utf-8"))
        self.assertTrue(meta.get("is_all_points"))
        self.assertEqual(meta.get("total_points"), 1000)
        self.assertEqual(meta.get("display_points"), 1000)
        self.assertIn("trees", meta)
        self.assertEqual(len(meta["trees"]), 4)

    def test_get_tree_points_full_neighbors_vs_cropped(self):
        """Verify that full_neighbors=False crops neighbors at radius, while full_neighbors=True
        includes the 100% complete point cloud of all neighbor trees touching the radius."""
        # 1. Cropped mode (full_neighbors=False)
        cropped_res = self.mgr.get_tree_points(10, radius=12.0, neighbor_mode="trees", full_neighbors=False)
        self.assertIsNotNone(cropped_res)
        self.assertFalse(cropped_res["full_neighbors"])
        itc_arr_crop = np.asarray(cropped_res["scalars"]["itc"])
        # Target tree 10 has all 250 points
        self.assertEqual((itc_arr_crop == 10).sum(), 250)
        # Tree 20 touches radius 12m but is cropped
        neigh_crop_count = (itc_arr_crop == 20).sum()
        self.assertGreater(neigh_crop_count, 0)
        self.assertLess(neigh_crop_count, 250)
        # Tree 30 is beyond radius
        self.assertEqual((itc_arr_crop == 30).sum(), 0)

        # 2. Full neighbors mode (full_neighbors=True)
        full_res = self.mgr.get_tree_points(10, radius=12.0, neighbor_mode="trees", full_neighbors=True)
        self.assertIsNotNone(full_res)
        self.assertTrue(full_res["full_neighbors"])
        itc_arr_full = np.asarray(full_res["scalars"]["itc"])
        # Target tree 10 has all 250 points
        self.assertEqual((itc_arr_full == 10).sum(), 250)
        # Tree 20 touches radius, so all 250 points of Tree 20 are included!
        self.assertEqual((itc_arr_full == 20).sum(), 250)
        # Tree 30 is still excluded because it never touches the radius
        self.assertEqual((itc_arr_full == 30).sum(), 0)
        # Full count is target (250) + neighbor (250) = 500
        self.assertEqual(full_res["count"], 500)

    def test_export_tree_points_full_neighbors(self):
        """Verify that export_tree_points respects full_neighbors=True and writes complete uncropped neighbor trees."""
        out_path = os.path.join(self.test_dir, "export_full_neighbors.laz")
        cnt = self.mgr.export_tree_points(out_path, 10, radius=12.0, neighbor_mode="trees", full_neighbors=True)
        self.assertEqual(cnt, 500)
        self.assertTrue(os.path.exists(out_path))

        out_las = laspy.read(out_path)
        self.assertEqual(len(out_las), 500)
        out_itc = np.asarray(out_las.itc)
        self.assertEqual((out_itc == 10).sum(), 250)
        self.assertEqual((out_itc == 20).sum(), 250)

    def test_full_neighbors_ui_sync_and_actions(self):
        """Verify MainWindow checkboxes sync between Forestry and AI tabs, pass correct parameters to JS, and handle tree_loaded."""
        # 1. UI Checkboxes exist
        self.assertTrue(hasattr(self.win, "chk_full_neighbors"))
        self.assertTrue(hasattr(self.win, "chk_ai_full_neighbors"))

        # 2. Sync Forestry -> AI tab
        self.win.chk_full_neighbors.setChecked(True)
        self.assertTrue(self.win.chk_ai_full_neighbors.isChecked())

        # 3. Sync AI tab -> Forestry
        self.win.chk_ai_full_neighbors.setChecked(False)
        self.assertFalse(self.win.chk_full_neighbors.isChecked())

        # 4. Enabling / disabling neighbors enables / disables full_neighbors checkboxes
        self.win._on_ai_neighbors_toggled(False)
        self.assertFalse(self.win.chk_ai_full_neighbors.isEnabled())
        self.assertFalse(self.win.chk_full_neighbors.isEnabled())

        self.win._on_ai_neighbors_toggled(True)
        self.assertTrue(self.win.chk_ai_full_neighbors.isEnabled())
        self.assertTrue(self.win.chk_full_neighbors.isEnabled())

        # 5. view_tree_action with full_neighbors passes fullNeighbors: true
        self.win.chk_full_neighbors.setChecked(True)
        self.win.chk_tree_neighbors.setChecked(True)
        self.win.spin_tree_view.setValue(10)
        self.win._js_calls.clear()
        self.win.view_tree_action()
        self.assertTrue(any("viewTreeById(10" in c and "true" in c for c in self.win._js_calls))
        self.assertTrue(self.win._tree_view_state.get("full_neighbors"))

        # 6. on_tree_loaded updates UI status
        data = {
            "itc": 10,
            "count": 500,
            "target_count": 250,
            "neighbor_count": 250,
            "radius": 12.0,
            "full_neighbors": True,
            "trees": [{"id": 10, "isTarget": True, "count": 250}, {"id": 20, "isTarget": False, "count": 250}]
        }
        self.win.on_tree_loaded(data)
        self.assertIn("full neighbor", self.win.lbl_tree_view_status.text().lower())
        self.assertTrue(self.win.chk_full_neighbors.isChecked())


if __name__ == "__main__":
    unittest.main()
