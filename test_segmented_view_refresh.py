"""
Test for Segmentation View & Stand Filter population when segmenting stand points.
"""
import os
import sys
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv)

from pointcloud_manager import PointCloudManager
from server import HybridServer
from app_window import MainWindow, ConfirmSplitDialog
from settings_store import SettingsStore


class TestSegmentedViewRefresh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.potree_dir = "/tmp/test_seg_refresh_potree"
        os.makedirs(cls.potree_dir, exist_ok=True)
        cls.db_path = "/tmp/test_seg_refresh.db"
        if os.path.exists(cls.db_path):
            os.remove(cls.db_path)
        cls.settings = SettingsStore(cls.db_path)
        web_dir = os.path.join(os.path.dirname(__file__), "web")
        cls.mgr = PointCloudManager()
        cls.mgr.settings_store = cls.settings
        cls.srv = HybridServer(potree_root=cls.potree_dir, web_root=web_dir, manager=cls.mgr, port=18960)
        cls.win = MainWindow(server=cls.srv, manager=cls.mgr, potree_dir=cls.potree_dir)
        cls.win.show()

    def test_stand_filter_populated_on_segmented_trees(self):
        """When segmentation finishes and produces trees (even with n_cnt == 0), stand filter should populate."""
        trees = [
            {"id": 1, "isTarget": False, "count": 25000},
            {"id": 2, "isTarget": False, "count": 18000},
            {"id": 3, "isTarget": False, "count": 12000},
            {"id": 0, "isTarget": False, "count": 5000}
        ]
        self.win._on_tree_stand_loaded({
            "target_itc": 7876,
            "neighbor_count": 0,
            "target_count": 0,
            "radius": 10.0,
            "trees": trees
        })
        self.win.sidebar_tabs.setCurrentIndex(1)
        self.assertFalse(self.win.grp_stand_filter.isHidden())
        self.assertEqual(self.win.list_stand_trees.count(), 4)
        item0 = self.win.list_stand_trees.item(0)
        self.assertIn("Tree #1", item0.text())


if __name__ == "__main__":
    unittest.main()
