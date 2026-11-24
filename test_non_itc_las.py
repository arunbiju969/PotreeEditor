import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import laspy
import numpy as np

# Ensure offscreen Qt platform for headless test execution
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication

app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

from pointcloud_manager import PointCloudManager
from server import HybridServer
from app_window import MainWindow


def _create_non_itc_las(path: str, count: int = 800):
    """Create a synthetic LAS file with XYZ, RGB, intensity, and ASPRS class, without ITC."""
    hdr = laspy.LasHeader(point_format=3, version="1.4")
    hdr.offsets = [335000.0, 4982000.0, 100.0]
    hdr.scales = [0.01, 0.01, 0.01]

    las = laspy.LasData(hdr)
    las.x = np.linspace(335000.0, 335040.0, count)
    las.y = np.linspace(4982000.0, 4982040.0, count)
    las.z = np.linspace(100.0, 125.0, count)
    las.intensity = np.full(count, 450, dtype=np.uint16)

    cls_arr = np.full(count, 5, dtype=np.uint8)
    cls_arr[: count // 2] = 2  # half ground, half high vegetation
    las.classification = cls_arr

    las.red = np.full(count, 10000, dtype=np.uint16)
    las.green = np.full(count, 30000, dtype=np.uint16)
    las.blue = np.full(count, 12000, dtype=np.uint16)

    las.write(path)


class TestNonItcPointCloudManager(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="non_itc_mgr_")
        self.las_path = os.path.join(self._tmp.name, "raw_scan.las")
        _create_non_itc_las(self.las_path, count=800)
        self.mgr = PointCloudManager()

    def tearDown(self):
        self._tmp.cleanup()

    def test_header_chunk_index_and_in_memory_without_itc(self):
        """PointCloudManager handles LAS files without ITC across header, chunk index, and RAM."""
        hdr_info = self.mgr.load_las_header(self.las_path)
        self.assertFalse(hdr_info["has_itc"])
        self.assertEqual(hdr_info["total_points"], 800)

        # Build chunk index synchronously and verify min_itc/max_itc defaults exist
        self.mgr._build_chunk_index_in_background(self.las_path)
        self.assertIsNotNone(self.mgr._chunk_index)
        self.assertGreaterEqual(len(self.mgr._chunk_index), 1)
        for chunk in self.mgr._chunk_index:
            self.assertEqual(chunk["min_itc"], 0)
            self.assertEqual(chunk["max_itc"], 0)

        # Querying a tree ID on a non-ITC cloud must not raise KeyError
        self.assertIsNone(self.mgr.get_tree_points(1))

        # Full in-memory load
        mem_info = self.mgr.load_las(self.las_path)
        self.assertFalse(mem_info["has_itc"])
        self.assertEqual(self.mgr.get_tree_ids(), [])

        # All-points payload falls back cleanly to ASPRS classification summary
        all_pts = self.mgr.get_all_points()
        self.assertIsNotNone(all_pts)
        self.assertTrue(all_pts["is_all_points"])
        self.assertEqual(all_pts["count"], 800)
        summary = all_pts.get("trees_summary", [])
        self.assertTrue(len(summary) >= 2)
        self.assertTrue(all(item.get("isClass") for item in summary))
        cls_ids = {item["id"] for item in summary}
        self.assertEqual(cls_ids, {2, 5})


class TestNonItcMainWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="non_itc_win_")
        cls.las_path = os.path.join(cls._tmp.name, "clipped_stand.las")
        _create_non_itc_las(cls.las_path, count=600)

        cls.mgr = PointCloudManager()
        web_dir = os.path.join(os.path.dirname(__file__), "web")
        cls.srv = HybridServer(potree_root=cls._tmp.name, web_root=web_dir, manager=cls.mgr, port=18959)
        cls.win = MainWindow(server=cls.srv, manager=cls.mgr, potree_dir=cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.win.close()
        cls._tmp.cleanup()

    def setUp(self):
        self.js_calls = []
        self._orig_run_js = self.win._run_js
        self.win._run_js = lambda js, cb=None: self.js_calls.append(js)
        self.mgr.las_path = None
        self.mgr.las_data = None
        self.mgr.header_info = {}
        self.mgr._chunk_index = None
        self.mgr._tree_index = None
        self.mgr._tree_cache.clear()

    def tearDown(self):
        self.win._run_js = self._orig_run_js

    def test_open_potree_path_delegates_raw_las(self):
        """open_potree_path delegates raw .las/.laz (excluding .copc.laz) to open_las_path."""
        with patch.object(self.win, "open_las_path") as mock_open_las:
            self.win.open_potree_path(self.las_path)
            mock_open_las.assert_called_once_with(self.las_path)

    def test_open_las_auto_loads_non_itc_and_updates_ui(self):
        """Opening a moderate non-ITC LAS with offer_conversion=False auto-loads & shows all points."""
        self.win.load_las_header_only(self.las_path)
        self.assertIn("no itc", self.win.lbl_tree_view_status.text().lower())

        self.win.open_las_path(self.las_path, offer_conversion=False)

        self.assertIsNotNone(self.mgr.las_data)
        self.assertFalse(self.mgr.header_info.get("has_itc"))
        self.assertTrue(any("window.viewAllPoints()" in c for c in self.js_calls))
        self.assertIn("no (not segmented)", self.win.lbl_cloud_info.text().lower())

        # ITC option should not appear in combo_tree_color for non-ITC cloud
        combo_data = [self.win.combo_tree_color.itemData(i) for i in range(self.win.combo_tree_color.count())]
        self.assertNotIn("itc", combo_data)

    def test_tree_actions_guard_gracefully_without_itc(self):
        """view_tree_action, next_tree_action, and start_auto_review handle non-ITC clouds safely."""
        self.win.load_las_for_editing(self.las_path)
        self.js_calls.clear()

        self.win.spin_tree_view.setValue(1)
        self.win.view_tree_action()
        self.assertFalse(any("viewTreeById" in c for c in self.js_calls))
        self.assertIn("no itc", self.win.lbl_tree_view_status.text().lower())

        self.win.next_tree_action()
        self.assertFalse(any("viewTreeById" in c for c in self.js_calls))
        self.assertIn("no itc", self.win.lbl_tree_view_status.text().lower())

        self.win.start_auto_review()
        self.assertFalse(getattr(self.win, "_auto_active", False))
        self.assertIn("itc", self.win.status_bar.currentMessage().lower())


if __name__ == "__main__":
    unittest.main()
