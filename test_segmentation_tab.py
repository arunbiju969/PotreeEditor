"""
Segmentation ribbon tab: full ForestFormer3D pipeline on a chosen point cloud.

The engine is stubbed, so the test proves the UI plumbing: parameter capture,
output routing into the workspace, run -> Load Result -> viewer handoff, and
that settings round-trip through the workspace database.
"""
import os
import shutil
import time
import sys
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import laspy
import numpy as np
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv)

import ff3d_module
from pointcloud_manager import PointCloudManager
from server import HybridServer
from app_window import MainWindow
from workspace import Workspace, workspace_path_for

MASTER = "/tmp/seg_tab/tract.laz"
POTREE = "/tmp/seg_tab_potree"
PORT = 18985


def make_master(path, n=500):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(1)
    las = laspy.create(point_format=3, file_version="1.4")
    las.x = rng.uniform(0, 50, n)
    las.y = rng.uniform(0, 50, n)
    las.z = rng.uniform(0, 8, n)
    las.write(path)


class StubModule:
    """Stands in for ff3d_module.ForestFormer3DModule."""

    calls = []

    def predict_file(self, in_las, out_dir, mode="tract", *, tile_size=70.0,
                     overlap=10.0, roi=None, model=None):
        StubModule.calls.append({"in_las": in_las, "out_dir": out_dir, "mode": mode,
                                 "tile_size": tile_size, "overlap": overlap,
                                 "roi": roi, "model": dict(model or {})})
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(in_las))[0]
        name = (f"{stem}_entire_tract_stitched.las" if mode == "tract"
                else f"{stem}_ff3d_segmented.las")
        out = os.path.join(out_dir, name)
        las = laspy.create(point_format=3, file_version="1.4")
        las.x = np.zeros(4); las.y = np.zeros(4); las.z = np.zeros(4)
        las.add_extra_dim(laspy.ExtraBytesParams(name="tree_id", type="int32"))
        las.tree_id = np.arange(4, dtype=np.int32)
        las.write(out)
        # Tiled runs leave a progress file the ribbon polls for tile counts.
        if mode == "tract":
            with open(os.path.join(out_dir, "tract_tiling_progress.json"), "w") as f:
                import json
                json.dump({"completed_tiles": {"t1": {}, "t2": {}},
                           "global_tree_offset": 7}, f)
        return {"ok": True, "out_las": out, "n_trees": 3}


class TestSegmentationTab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(POTREE, exist_ok=True)
        cls.mgr = PointCloudManager()
        cls.srv = HybridServer(potree_root=POTREE, web_root=os.path.join(
            os.path.dirname(__file__), "web"), manager=cls.mgr, port=PORT)
        cls.win = MainWindow(server=cls.srv, manager=cls.mgr, potree_dir=POTREE)
        cls.win.show()

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def setUp(self):
        StubModule.calls = []
        # test_finished_run... stubs open_las_path on the shared window; restore it
        self.win.__dict__.pop("open_las_path", None)
        shutil.rmtree(workspace_path_for(MASTER), ignore_errors=True)
        make_master(MASTER)
        self.win.open_las_path(MASTER, offer_conversion=False)
        ff3d_module.get_module = lambda: StubModule()
        self.win.seg_edit_input.setText("")
        self.win._seg_advanced = {}
        self.win._seg_result = None
        # Shared window: reset parameters to the shipped defaults per test.
        self.win.combo_seg_mode.setCurrentIndex(0)
        for widget, value in ((self.win.spin_seg_radius, 16.0), (self.win.spin_seg_step, 6.0),
                              (self.win.spin_seg_score, 0.40), (self.win.spin_seg_tile, 70.0),
                              (self.win.spin_seg_overlap, 10.0)):
            widget.setValue(value)
        self.win.chk_seg_roi.setChecked(False)
        self.win.btn_seg_load.setEnabled(False)

    def _run_to_completion(self, timeout_ms=20000):
        from PyQt6.QtCore import QDeadlineTimer, QCoreApplication
        deadline = QDeadlineTimer(timeout_ms)
        while not deadline.hasExpired():
            QCoreApplication.processEvents()
            time.sleep(0.005)          # let the worker thread get the GIL
            worker = self.win.seg_worker
            if worker is not None and not worker.isRunning():
                QCoreApplication.processEvents()   # deliver finished/error
                return True
        self.fail("segmentation run did not finish")

    def test_tab_exists_with_pipeline_defaults(self):
        self.win.btn_tab_segmentation.click()
        self.assertEqual(self.win.ribbon_stack.currentIndex(), 6)
        self.assertEqual(self.win.combo_seg_mode.currentData(), "tract")
        self.assertAlmostEqual(self.win.spin_seg_radius.value(), 16.0)
        self.assertAlmostEqual(self.win.spin_seg_step.value(), 6.0)
        self.assertAlmostEqual(self.win.spin_seg_score.value(), 0.40)
        self.assertAlmostEqual(self.win.spin_seg_tile.value(), 70.0)
        self.assertAlmostEqual(self.win.spin_seg_overlap.value(), 10.0)
        self.assertFalse(self.win.chk_seg_roi.isChecked())
        self.assertFalse(self.win.spin_seg_xmin.isEnabled())
        self.assertTrue(self.win.btn_seg_run.isEnabled())
        self.assertFalse(self.win.btn_seg_load.isEnabled())

    def test_uses_open_cloud_and_writes_into_workspace(self):
        self.win.seg_use_open_cloud()
        self.assertEqual(self.win.seg_edit_input.text(), MASTER)

        self.win.seg_worker = None
        self.win.run_segmentation_action()
        self.assertTrue(self._run_to_completion())

        call = StubModule.calls[0]
        self.assertEqual(call["in_las"], MASTER)
        self.assertEqual(call["mode"], "tract")
        self.assertEqual(call["out_dir"],
                         os.path.join(self.win.workspace.segmented_dir, "pipeline"))
        self.assertAlmostEqual(call["tile_size"], 70.0)
        self.win._seg_timer.stop()

    def test_parameters_and_roi_reach_the_engine(self):
        self.win.combo_seg_mode.setCurrentIndex(1)          # single cloud
        self.win.spin_seg_radius.setValue(20.0)
        self.win.spin_seg_step.setValue(4.0)
        self.win.spin_seg_score.setValue(0.55)
        self.win.chk_seg_roi.setChecked(True)
        self.assertTrue(self.win.spin_seg_xmin.isEnabled())
        self.win.spin_seg_xmin.setValue(10.0)
        self.win.spin_seg_xmax.setValue(30.0)
        self.win.spin_seg_ymin.setValue(5.0)
        self.win.spin_seg_ymax.setValue(25.0)
        self.win._seg_advanced = {"config": "/tmp/my.py", "device": "cuda:1"}

        self.win.seg_edit_input.setText(MASTER)
        self.win.run_segmentation_action()
        self.assertTrue(self._run_to_completion())

        call = StubModule.calls[0]
        self.assertEqual(call["mode"], "single")
        self.assertEqual(call["roi"], [10.0, 5.0, 30.0, 25.0])
        self.assertAlmostEqual(call["model"]["radius"], 20.0)
        self.assertAlmostEqual(call["model"]["step_size"], 4.0)
        self.assertAlmostEqual(call["model"]["score_th"], 0.55)
        self.assertEqual(call["model"]["config"], "/tmp/my.py")
        self.assertEqual(call["model"]["device"], "cuda:1")
        self.win._seg_timer.stop()

    def test_finished_run_enables_load_result(self):
        self.win.seg_edit_input.setText(MASTER)
        self.win.run_segmentation_action()
        self.assertTrue(self._run_to_completion())

        self.assertTrue(self.win.btn_seg_load.isEnabled())
        out = self.win._seg_result
        self.assertTrue(out.endswith("_entire_tract_stitched.las"))
        self.assertTrue(os.path.isfile(out))
        self.assertEqual(self.win.bar_seg_progress.value(), 100)

        loaded = []
        self.win.open_las_path = lambda path, offer_conversion=True: loaded.append(path)
        self.win.load_segmentation_result()
        self.assertEqual(loaded, [out])
        self.win._seg_timer.stop()

    def test_tile_progress_is_surfaced(self):
        self.win.seg_edit_input.setText(MASTER)
        self.win.run_segmentation_action()
        self.assertTrue(self._run_to_completion())
        self.win._on_seg_tick()
        self.assertIn("Tiles done: 2", self.win.lbl_seg_status.text())
        self.win._seg_timer.stop()

    def test_parameters_persist_to_the_workspace(self):
        self.win.spin_seg_radius.setValue(22.5)
        self.win.combo_seg_mode.setCurrentIndex(1)
        self.win.chk_seg_roi.setChecked(True)
        self.win._seg_advanced = {"device": "cuda:1"}
        self.win._save_settings()

        fresh = MainWindow(server=self.srv, manager=self.mgr, potree_dir=POTREE)
        fresh.show()
        fresh.open_las_path(MASTER, offer_conversion=False)   # attach the workspace store
        fresh._apply_loaded_settings()
        self.assertAlmostEqual(fresh.spin_seg_radius.value(), 22.5)
        self.assertEqual(fresh.combo_seg_mode.currentData(), "single")
        self.assertTrue(fresh.chk_seg_roi.isChecked())
        self.assertEqual(fresh._seg_advanced.get("device"), "cuda:1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
