import os
import sys
import unittest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

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
from app_window import MainWindow, BridgeSignaler


class TestStandFilterUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        potree_dir = "/tmp/test_stand_filter_potree"
        os.makedirs(potree_dir, exist_ok=True)
        web_dir = os.path.join(os.path.dirname(__file__), "web")
        cls.mgr = PointCloudManager()
        cls.srv = HybridServer(potree_root=potree_dir, web_root=web_dir, manager=cls.mgr, port=18949)
        cls.win = MainWindow(server=cls.srv, manager=cls.mgr, potree_dir=potree_dir)
        cls.win.show()

    def setUp(self):
        self.win._js_calls = []
        self.win._orig_run_js = self.win._run_js
        self.win._run_js = lambda js: self.win._js_calls.append(js)

        # Standard stand test payload
        self.stand_data = {
            "target_itc": 105,
            "count": 50000,
            "target_count": 12000,
            "neighbor_count": 38000,
            "radius": 25.0,
            "trees": [
                {"id": 105, "isTarget": True, "count": 12000},
                {"id": 106, "isTarget": False, "count": 15000},
                {"id": 107, "isTarget": False, "count": 13000},
                {"id": 0, "isTarget": False, "count": 10000}
            ]
        }
        self.win.signaler.tree_stand_loaded.emit(self.stand_data)

    def tearDown(self):
        self.win._run_js = self.win._orig_run_js

    def test_controls_exist(self):
        """Verify all Stand Filter UI controls exist in Forestry tab and Top Category Bar."""
        self.assertTrue(hasattr(self.win, "grp_stand_filter"))
        self.assertTrue(hasattr(self.win, "lbl_stand_filter_info"))
        self.assertTrue(hasattr(self.win, "btn_tree_filter_mode"))
        self.assertTrue(hasattr(self.win, "list_stand_trees"))
        self.assertTrue(hasattr(self.win, "btn_apply_tree_filter"))
        self.assertTrue(hasattr(self.win, "btn_reset_tree_filter"))
        self.assertTrue(hasattr(self.win, "bar_stand_filter"))
        self.assertTrue(hasattr(self.win, "lbl_top_stand_info"))
        self.win.close_tree_view_action()
        self.assertFalse(self.win.grp_stand_filter.isVisible())
        self.assertFalse(self.win.bar_stand_filter.isVisible())

    def test_stand_loaded_populates_list(self):
        """Verify tree_stand_loaded event shows group, populates top bar and list with checkboxes."""
        self.win.sidebar_tabs.setCurrentIndex(1)  # Forestry tab
        stand_data = {
            "target_itc": 105,
            "count": 50000,
            "target_count": 12000,
            "neighbor_count": 38000,
            "radius": 25.0,
            "trees": [
                {"id": 105, "isTarget": True, "count": 12000},
                {"id": 106, "isTarget": False, "count": 15000},
                {"id": 107, "isTarget": False, "count": 13000},
                {"id": 0, "isTarget": False, "count": 10000}
            ]
        }
        self.win.signaler.tree_stand_loaded.emit(stand_data)

        self.assertFalse(self.win.grp_stand_filter.isHidden())
        self.assertFalse(self.win.bar_stand_filter.isHidden())
        self.assertIn("Tree #105", self.win.lbl_top_stand_info.text())
        self.assertEqual(self.win.list_stand_trees.count(), 4)

        # Target tree is first
        item0 = self.win.list_stand_trees.item(0)
        self.assertEqual(item0.data(Qt.ItemDataRole.UserRole), 105)
        self.assertTrue("★ Target Tree #105" in item0.text())
        self.assertEqual(item0.checkState(), Qt.CheckState.Unchecked)

        # Neighbors
        item1 = self.win.list_stand_trees.item(1)
        self.assertEqual(item1.data(Qt.ItemDataRole.UserRole), 106)
        self.assertEqual(item1.checkState(), Qt.CheckState.Unchecked)

        item2 = self.win.list_stand_trees.item(2)
        self.assertEqual(item2.data(Qt.ItemDataRole.UserRole), 107)
        self.assertEqual(item2.checkState(), Qt.CheckState.Unchecked)

        item3 = self.win.list_stand_trees.item(3)
        self.assertEqual(item3.data(Qt.ItemDataRole.UserRole), 0)
        self.assertEqual(item3.checkState(), Qt.CheckState.Unchecked)

    def test_mode_filter_toggle(self):
        """Verify Mode: Filter toggle button communicates with WebGL."""
        # Toggle ON
        self.win.btn_tree_filter_mode.setChecked(True)
        self.assertTrue(any("window.setTreeFilterMode(true)" in c for c in self.win._js_calls))
        self.assertIn("Filter Mode ON", self.win.btn_tree_filter_mode.text())

        # Toggle OFF
        self.win._js_calls.clear()
        self.win.btn_tree_filter_mode.setChecked(False)
        self.assertTrue(any("window.setTreeFilterMode(false)" in c for c in self.win._js_calls))
        self.assertIn("Filter Mode", self.win.btn_tree_filter_mode.text())

    def test_hud_filter_mode_sync(self):
        """Verify toggling filter mode in WebGL HUD updates the Qt button."""
        self.win.btn_tree_filter_mode.setChecked(False)
        self.win.signaler.tree_filter_mode_toggled.emit({"filter_mode": True})
        self.assertTrue(self.win.btn_tree_filter_mode.isChecked())
        self.assertIn("Filter Mode ON", self.win.btn_tree_filter_mode.text())

        self.win.signaler.tree_filter_mode_toggled.emit({"filter_mode": False})
        self.assertFalse(self.win.btn_tree_filter_mode.isChecked())

    def test_3d_pick_syncs_list_selection(self):
        """Verify raycast pick event in 3D viewport updates list checkboxes."""
        # Unselect tree 107 via 3D event
        self.win.signaler.tree_filter_selection_changed.emit({
            "selected_ids": [105, 106, 0],
            "toggled_id": 107,
            "is_selected": False
        })
        item_107 = self.win.list_stand_trees.item(2)
        self.assertEqual(item_107.checkState(), Qt.CheckState.Unchecked)

        # Re-select tree 107
        self.win.signaler.tree_filter_selection_changed.emit({
            "selected_ids": [105, 106, 107, 0],
            "toggled_id": 107,
            "is_selected": True
        })
        self.assertEqual(item_107.checkState(), Qt.CheckState.Checked)

    def test_quick_selection_and_actions(self):
        """Verify Target Only, Select All, Filter and Reset buttons."""
        self.win.spin_tree_view.setValue(105)

        # Target Only
        self.win._js_calls.clear()
        self.win._on_filter_select_target_only()
        self.assertEqual(self.win.list_stand_trees.item(0).checkState(), Qt.CheckState.Checked)
        self.assertEqual(self.win.list_stand_trees.item(1).checkState(), Qt.CheckState.Unchecked)
        self.assertTrue(any("setTreeFilterSelection([105])" in c for c in self.win._js_calls))

        # Select All
        self.win._js_calls.clear()
        self.win._on_filter_select_all()
        for i in range(self.win.list_stand_trees.count()):
            self.assertEqual(self.win.list_stand_trees.item(i).checkState(), Qt.CheckState.Checked)
        self.assertTrue(any("setTreeFilterSelection([105, 106, 107, 0])" in c for c in self.win._js_calls))

        # Apply Filter
        self.win._js_calls.clear()
        self.win._apply_tree_filter()
        self.assertTrue(any("applyTreeFilter([105, 106, 107, 0])" in c for c in self.win._js_calls))

        # Reset Filter
        self.win._js_calls.clear()
        self.win._reset_tree_filter()
        self.assertTrue(any("resetTreeFilter()" in c for c in self.win._js_calls))

    def test_tab_isolation_disables_filter_mode(self):
        """Verify switching away from Forestry tab disables 3D filter mode."""
        self.win.sidebar_tabs.setCurrentIndex(1)  # Forestry
        self.win.btn_tree_filter_mode.setChecked(True)

        self.win._js_calls.clear()
        self.win.sidebar_tabs.setCurrentIndex(0)  # Layers
        self.assertFalse(self.win.btn_tree_filter_mode.isChecked())
        self.assertTrue(any("setTreeFilterMode(false)" in c for c in self.win._js_calls))

    def test_close_tree_view_hides_filter_and_cleans_up(self):
        """Verify close_tree_view_action hides stand filter and clears state."""
        self.win.sidebar_tabs.setCurrentIndex(1)
        self.win.grp_stand_filter.setVisible(True)
        self.win.btn_tree_filter_mode.setChecked(True)

        self.win.close_tree_view_action()
        self.assertFalse(self.win.grp_stand_filter.isVisible())
        self.assertEqual(self.win.list_stand_trees.count(), 0)
        self.assertFalse(self.win.btn_tree_filter_mode.isChecked())


if __name__ == "__main__":
    unittest.main()
