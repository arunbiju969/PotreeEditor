"""
Segmentation ribbon tab: full ForestFormer3D pipeline on a chosen point cloud.

The engine is stubbed, so the test proves the UI plumbing: parameter capture,
output routing into the workspace, run -> Load Result -> viewer handoff, and
that settings round-trip through the workspace database.
"""
import json
import os
import shutil
import tempfile
import time
import sys
import unittest

from PyQt6.QtCore import QCoreApplication, QDeadlineTimer

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
    delay = 0.0

    def predict_file(self, in_las, out_dir, mode="tract", *, tile_size=70.0,
                     overlap=10.0, roi=None, model=None, on_output=None):
        StubModule.calls.append({"in_las": in_las, "out_dir": out_dir, "mode": mode,
                                 "tile_size": tile_size, "overlap": overlap,
                                 "roi": roi, "model": dict(model or {})})
        if on_output:
            on_output(f"[stub] {mode} pass over {os.path.basename(in_las)}")
            on_output("[stub] merge + seam stitch complete")
        if StubModule.delay:
            time.sleep(StubModule.delay)      # keep the worker in flight
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
        StubModule.delay = 0.0
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
        # Progress state is per run; clear leftovers from the previous test.
        self.win._seg_mode_running = "single"
        self.win._seg_tiles_total = 0
        self.win._seg_tile_index = 0
        self.win._seg_pct_logged = 0
        self.win._seg_diag_n = 0
        self.win.bar_seg_progress.setRange(0, 100)
        self.win.bar_seg_progress.setValue(0)

    def _run_to_completion(self, timeout_ms=20000):
        deadline = QDeadlineTimer(timeout_ms)
        while not deadline.hasExpired():
            QCoreApplication.processEvents()
            time.sleep(0.005)          # let the worker thread get the GIL
            worker = self.win.seg_worker
            if worker is not None and not worker.isRunning():
                QCoreApplication.processEvents()   # deliver finished/error
                return True
        self.fail("segmentation run did not finish")

    def _settle(self, ms=150):
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(ms, lambda: None)
        deadline = QDeadlineTimer(ms)
        while not deadline.hasExpired():
            QCoreApplication.processEvents()
            time.sleep(0.005)

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

    def test_engine_output_streams_to_the_console(self):
        self.win.txt_console.clear()
        self.win._on_seg_output("[tile_0_1] cylinder inference written in 12.3 s")
        self.win._on_seg_output("Processing regions (FAST): 44%|████ | 4/9 [00:01<00:01]")
        self.win._on_seg_output("Error: CUDA out of memory")
        text = self.win.txt_console.toPlainText()
        self.assertIn("[tile_0_1] cylinder inference written in 12.3 s", text)
        self.assertIn("CUDA out of memory", text)
        # The region bar feeds the progress bar instead of the log.
        self.assertEqual(self.win.bar_seg_progress.value(), 44)
        self.assertEqual(text.count("Processing regions"), 0)
        # The label mirrors the most recent engine line.
        self.assertIn("CUDA out of memory", self.win.lbl_seg_status.text())

    def test_tile_counter_is_logged_once_per_tile(self):
        self.win.txt_console.clear()
        self.win._seg_run_dir = self.win._seg_output_dir()
        self.win._seg_tiles_logged = 0
        os.makedirs(self.win._seg_run_dir, exist_ok=True)
        with open(os.path.join(self.win._seg_run_dir, "tract_tiling_progress.json"), "w") as f:
            json.dump({"completed_tiles": {"a": {}}, "global_tree_offset": 9}, f)
        self.win._on_seg_tick()
        self.win._on_seg_tick()
        text = self.win.txt_console.toPlainText()
        self.assertEqual(text.count("1 tile(s) segmented"), 1)
        self.assertIn("9 trees found so far", text)

    def test_closing_the_window_stops_the_engine(self):
        """The CUDA child must not outlive the window."""
        closed = []
        real_shutdown = ff3d_module.shutdown
        ff3d_module.shutdown = lambda: closed.append(True)
        self.win._seg_closing = False
        StubModule.delay = 0.6
        try:
            self.win.seg_edit_input.setText(MASTER)
            self.win.run_segmentation_action()
            deadline = QDeadlineTimer(3000)
            while not self.win.seg_worker.isRunning() and not deadline.hasExpired():
                QCoreApplication.processEvents()
                time.sleep(0.005)
            self.assertTrue(self.win.seg_worker.isRunning(), "run should be in flight")
            from PyQt6.QtGui import QCloseEvent
            self.win.closeEvent(QCloseEvent())
            self.assertTrue(closed, "ff3d_module.shutdown() was not called on close")
            self.assertFalse(self.win.seg_worker.isRunning())
            self._settle(200)   # deliver the queued cancellation signal
            self.assertIn("cancelling the running", self.win.txt_console.toPlainText())
            # The engine child was told to stop; a real CUDA child dies mid-run
            # (Cancelled), a stubbed one finishes its pending call (Finished).
            self.assertIn(self.win.lbl_seg_status.text(), ("Cancelled", "Finished"))
            StubModule.delay = 0.0
            self.win.show()   # shared window for the remaining tests
        finally:
            ff3d_module.shutdown = real_shutdown

    def test_region_bar_drives_percent_and_eta(self):
        self.win._seg_mode_running = "single"
        self.win._seg_tiles_total = 0
        self.win._seg_pct_logged = 0
        self.win._seg_diag_n = 0
        self.win._seg_t0 = __import__("time").time() - 60
        self.win.txt_console.clear()
        self.win._on_seg_output("Processing regions (FAST): 50%|█████ | 5/10 [01:00<01:00, 1.02it/s]")
        self.assertEqual(self.win.bar_seg_progress.value(), 50)
        self.assertIn("50%", self.win.lbl_seg_status.text())
        self.assertIn("left", self.win.lbl_seg_status.text())

        self.win._on_seg_output("Processing regions (FAST): 60%|██████ | 6/10 [01:10<00:50, 1.01it/s]")
        self.win._on_seg_output("Processing regions (FAST): 60%|██████ | 6/10 [01:12<00:48, 1.00it/s]")
        text = self.win.txt_console.toPlainText()
        self.assertEqual(text.count("Progress 50%"), 1)   # one line per decile
        self.assertIn("Progress 60%", text)

    def test_tract_percent_counts_tiles_and_regions(self):
        self.win._seg_mode_running = "tract"
        self.win._seg_tiles_total = 0
        self.win._seg_tile_index = 0
        self.win._seg_pct_logged = 100
        self.win._seg_t0 = __import__("time").time() - 100
        self.win._on_seg_output("[Tract Fast Grid] 4 active tiles contain points.")
        self.win._on_seg_output("[2/4] Fast Processing Tile 002 [tile_002_x1_y1] (229,667 core points)")
        self.assertEqual(self.win._seg_tiles_total, 4)
        self.assertIn("Tile 002", self.win.txt_console.toPlainText())
        self.win._on_seg_output("Processing regions (FAST): 33%|███ | 3/9 [00:02<00:05, 1.20it/s]")
        # Tile 2 of 4 with a third of its regions: (1 + 3/9) / 4 = 33 %.
        self.assertEqual(self.win.bar_seg_progress.value(), 33)
        self.assertIn("tile 2/4", self.win.lbl_seg_status.text())

    def test_verbose_diagnostics_are_thinned(self):
        self.win._seg_diag_n = 0
        self.win.txt_console.clear()
        for i in range(16):
            self.win._on_seg_output(f"[VRAM] regions {i*32}-{i*32+32}: peak 5.4 GB | masks {i}")
        self.assertEqual(self.win.txt_console.toPlainText().count("[VRAM]"), 2)

    def test_run_opens_the_console_drawer(self):
        self.win._set_console_open(False)
        self.win.seg_edit_input.setText(MASTER)
        self.win.run_segmentation_action()
        self.assertTrue(self.win.console_panel.isVisible(),
                        "a run should open the console drawer")
        self.assertIn("Hide Console", self.win.btn_toggle_console.text())
        self.assertTrue(self._run_to_completion())
        self.assertIn("Pipeline run started", self.win.txt_console.toPlainText())

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


class TestEngineProcess(unittest.TestCase):
    """The engine runs in its own interpreter: shutdown() must kill the child."""

    def test_worker_child_terminates_on_shutdown(self):
        tmp = tempfile.mkdtemp(prefix="ff3d_fake_")
        with open(os.path.join(tmp, "ff3d_api.py"), "w", encoding="utf-8") as f:
            f.write(
                "import numpy as np\n"
                "class ForestFormer3D:\n"
                "    def __init__(self, **kw):\n"
                "        self.kw = kw\n"
                "    def predict_points(self, xyz):\n"
                "        print('[fake] cylinder inference', flush=True)\n"
                "        n = len(xyz)\n"
                "        return (np.zeros(n, np.int16), np.arange(n, dtype=np.int32),\n"
                "                np.zeros(n, np.float32))\n")
        real_repo, real_py = ff3d_module.FF3D_REPO, ff3d_module.FF3D_PYTHON
        ff3d_module.FF3D_REPO, ff3d_module.FF3D_PYTHON = tmp, sys.executable
        mod = ff3d_module.ForestFormer3DModule()
        try:
            heard = []
            sem, inst, _ = mod.predict(np.zeros((4, 3), dtype=np.float32),
                                       on_output=heard.append)
            self.assertEqual(len(sem), 4)
            self.assertIn("[fake] cylinder inference", " ".join(heard))

            pid = mod._proc.pid
            self.assertIsNone(mod._proc.poll(), "child should be warm and alive")
            ff3d_module._module = mod
            ff3d_module.shutdown()
            self.assertIsNone(mod._proc, "module forgets the terminated child")
            with self.assertRaises(OSError):   # the CUDA child is really gone
                os.kill(pid, 0)
        finally:
            ff3d_module.FF3D_REPO, ff3d_module.FF3D_PYTHON = real_repo, real_py


if __name__ == "__main__":
    unittest.main(verbosity=2)
