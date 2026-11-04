import os
import sys
import json
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

app = QApplication.instance()
if not app:
    app = QApplication(sys.argv)

from pointcloud_manager import PointCloudManager
from server import HybridServer
from app_window import MainWindow, ConfirmSplitDialog, EditHistoryDialog
from settings_store import SettingsStore


class TestFilteredStandSegmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.potree_dir = "/tmp/test_filtered_seg_potree"
        os.makedirs(cls.potree_dir, exist_ok=True)
        cls.db_path = "/tmp/test_filtered_seg.db"
        if os.path.exists(cls.db_path):
            os.remove(cls.db_path)
        cls.settings = SettingsStore(cls.db_path)
        web_dir = os.path.join(os.path.dirname(__file__), "web")
        cls.mgr = PointCloudManager()
        cls.mgr.settings_store = cls.settings
        cls.srv = HybridServer(potree_root=cls.potree_dir, web_root=web_dir, manager=cls.mgr, port=18951)
        cls.win = MainWindow(server=cls.srv, manager=cls.mgr, potree_dir=cls.potree_dir)
        cls.win.show()

    def setUp(self):
        self.win._js_calls = []
        self.win._orig_run_js = self.win._run_js
        self.win._run_js = lambda js: self.win._js_calls.append(js)

    def tearDown(self):
        self.win._run_js = self.win._orig_run_js

    def test_confirm_split_dialog_target_included(self):
        """When target tree is included, cluster 0 should keep target_itc."""
        dlg = ConfirmSplitDialog(self.win, target_itc=2969, n_trees=3, seg_las_path="dummy.las", max_itc=11590, is_target_included=True)
        id_map = dlg.get_id_mapping()
        self.assertEqual(id_map[0], 2969)
        self.assertEqual(id_map[1], 11591)
        self.assertEqual(id_map[2], 11592)
        self.assertIn("Tree #2969", dlg.windowTitle())

    def test_confirm_split_dialog_unassigned_points(self):
        """When only unassigned points (ITC 0) are segmented, all clusters must get NEW IDs."""
        dlg = ConfirmSplitDialog(self.win, target_itc=2969, n_trees=3, seg_las_path="dummy.las", max_itc=11590, is_target_included=False)
        id_map = dlg.get_id_mapping()
        # Should not overwrite 2969!
        self.assertNotEqual(id_map[0], 2969)
        self.assertEqual(id_map[0], 11591)
        self.assertEqual(id_map[1], 11592)
        self.assertEqual(id_map[2], 11593)
        self.assertIn("New Trees", dlg.windowTitle())

    def test_segment_tree_action_forwards_stand_filter(self):
        """segment_tree_action passes checked keep_ids to window.segmentTreeById."""
        self.win.spin_tree_view.setValue(2969)
        # Mock las_path to bypass check
        self.win.manager.las_path = "/tmp/dummy.las"

        # Populate stand filter list with Tree 2969, Tree 102, Canopy (0)
        self.win.list_stand_trees.clear()
        stand_data = {
            "target_itc": 2969,
            "count": 1000,
            "target_count": 500,
            "neighbor_count": 500,
            "radius": 15.0,
            "trees": [
                {"id": 2969, "isTarget": True, "count": 500},
                {"id": 102, "isTarget": False, "count": 300},
                {"id": 0, "isTarget": False, "count": 200}
            ]
        }
        self.win.signaler.tree_stand_loaded.emit(stand_data)

        # Check ONLY Canopy / Ground (id 0)
        for i in range(self.win.list_stand_trees.count()):
            item = self.win.list_stand_trees.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == 0:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)

        self.win.segment_tree_action()
        self.assertTrue(len(self.win._js_calls) > 0)
        last_call = self.win._js_calls[-1]
        self.assertIn("segmentTreeById", last_call)
        self.assertIn("[0]", last_call)

    def test_pointcloud_manager_edited_tree_info_target_none(self):
        """get_edited_tree_info handles staged edit with target_itc=None without TypeError."""
        staged = {
            "type": "ff3d_split",
            "target_itc": None,
            "segmented_las": "dummy.las",
            "id_mapping": {0: 11591, 1: 11592},
            "db_edit_id": 1
        }
        self.win.manager.staged_edits = [staged]
        info = self.win.manager.get_edited_tree_info(11591)
        self.assertIsNotNone(info)
        self.assertEqual(info["itc_id"], 11591)
        self.assertEqual(info["parent_itc"], 11591)


if __name__ == "__main__":
    unittest.main()
