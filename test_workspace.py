"""
Per-master editing workspace: <master>.potreeedit/ sidecar.

Covers the three promises of a workspace:
  1. commits stack onto one cumulative committed.laz (append semantics),
  2. staged edits survive an app restart,
  3. ForestFormer3D split files land inside the workspace.
"""
import os
import shutil
import sys
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import laspy
import numpy as np
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv)

from pointcloud_manager import PointCloudManager
from server import HybridServer
from app_window import MainWindow
from workspace import Workspace, workspace_path_for

MASTER = "/tmp/ws_probe/tract.laz"
POTREE = "/tmp/ws_probe_potree"
SRV_PORT = 18975


def make_master(path, n=4000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(0)
    las = laspy.create(point_format=3, file_version="1.4")
    las.x = rng.uniform(0, 10, n)
    las.y = rng.uniform(0, 10, n)
    las.z = rng.uniform(0, 5, n)
    las.add_extra_dim(laspy.ExtraBytesParams(name="itc", type="int32"))
    las.itc = np.ones(n, dtype=np.int32)
    las.write(path)


def commit(manager, source, target):
    """Mirror CommitWorker: stream to a temp beside the target, then swap."""
    tmp = target + ".tmp.laz"
    res = manager.commit_staged_edits(source, tmp)
    os.replace(tmp, target)
    return res


class TestWorkspaceLayout(unittest.TestCase):
    def setUp(self):
        shutil.rmtree("/tmp/ws_probe", ignore_errors=True)
        shutil.rmtree(workspace_path_for(MASTER), ignore_errors=True)
        make_master(MASTER)
        self.ws = Workspace(MASTER).ensure()

    def test_sidecar_folder_is_created_next_to_master(self):
        self.assertEqual(self.ws.root, "/tmp/ws_probe/tract.potreeedit")
        self.assertTrue(os.path.isdir(self.ws.segmented_dir))
        self.assertTrue(os.path.isfile(self.ws.manifest_path))
        self.assertEqual(self.ws.manifest()["master"], os.path.abspath(MASTER))
        self.assertEqual(self.ws.manifest()["commit_count"], 0)
        # Only the master and one sidecar folder sit in the data folder.
        self.assertEqual(sorted(os.listdir("/tmp/ws_probe")),
                         ["tract.laz", "tract.potreeedit"])

    def test_commits_stack_into_one_cumulative_file(self):
        mgr = PointCloudManager()
        mgr.load_las(MASTER)

        mgr.add_staged_edit({"type": "box", "min": [0, 0, 0], "max": [5, 5, 5],
                             "action": "assign_itc", "value": 42})
        commit(mgr, self.ws.commit_source, self.ws.committed_laz)

        mgr.add_staged_edit({"type": "box", "min": [0, 5, 0], "max": [10, 10, 10],
                             "action": "set_classification", "value": 7})
        self.assertTrue(self.ws.has_committed, "second commit should read committed.laz")
        commit(mgr, self.ws.commit_source, self.ws.committed_laz)

        out = laspy.read(self.ws.committed_laz)
        self.assertGreater(int((np.asarray(out.itc) == 42).sum()), 0,
                           "commit #1 must still be present after commit #2")
        self.assertGreater(int((np.asarray(out.classification) == 7).sum()), 0,
                           "commit #2 must be applied")
        self.assertEqual(self.ws.manifest()["commit_count"], 0)  # manifest written by the UI

    def test_record_commit_updates_manifest(self):
        self.ws.record_commit(self.ws.committed_laz, edits_count=2,
                              modified_points=1234, elapsed_seconds=3.45,
                              potree_dir="/tmp/converted")
        m = self.ws.manifest()
        self.assertEqual(m["commit_count"], 1)
        self.assertEqual(m["last_commit"]["edits"], 2)
        self.assertEqual(m["last_commit"]["modified_points"], 1234)
        self.assertEqual(m["last_commit"]["potree_dir"], "/tmp/converted")


class TestWorkspaceInMainWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(POTREE, exist_ok=True)
        cls.mgr = PointCloudManager()
        cls.srv = HybridServer(potree_root=POTREE, web_root=os.path.join(
            os.path.dirname(__file__), "web"), manager=cls.mgr, port=SRV_PORT)
        cls.win = MainWindow(server=cls.srv, manager=cls.mgr, potree_dir=POTREE)
        cls.win.show()

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def setUp(self):
        shutil.rmtree(workspace_path_for(MASTER), ignore_errors=True)
        make_master(MASTER)
        self.win.open_las_path(MASTER, offer_conversion=False)
        self.win.manager.clear_staged_edits()

    def test_opening_master_binds_its_workspace_database(self):
        ws = self.win.workspace
        self.assertIsNotNone(ws)
        self.assertEqual(ws.root, "/tmp/ws_probe/tract.potreeedit")
        self.assertEqual(self.win.settings_store.path, ws.db_path)
        self.assertIs(self.win.manager.workspace, ws)
        self.assertTrue(os.path.isfile(ws.db_path))

    def test_staged_edits_are_restored_for_the_next_session(self):
        self.win.manager.add_staged_edit({"type": "polygon", "action": "assign_itc",
                                          "value": 21, "coords": [[0, 0], [1, 0], [1, 1]]})
        self.win.refresh_staged_edits_table()
        self.assertGreater(self.win.table_staged.rowCount(), 0)

        # Reopening the master (as a fresh session would) restores the snapshot.
        self.win._attach_workspace(MASTER)
        restored = self.win.manager.get_staged_edits()
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0]["value"], 21)
        self.assertEqual(self.win.table_staged.rowCount(), 1)

    def test_ff3d_segmented_output_lands_in_workspace(self):
        from server import HybridRequestHandler
        handler = HybridRequestHandler.__new__(HybridRequestHandler)   # no socket needed
        handler.manager = self.win.manager
        path = handler._write_ff3d_las(
            1234, np.zeros((3, 3), dtype=np.float32),
            np.zeros(3, dtype=np.uint8), np.arange(3, dtype=np.int32),
            np.ones(3, dtype=np.float32), {"count": 3})
        self.assertEqual(os.path.dirname(os.path.abspath(path)),
                         self.win.workspace.segmented_dir)
        self.assertTrue(os.path.isfile(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
