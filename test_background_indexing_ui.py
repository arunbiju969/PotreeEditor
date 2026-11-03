"""
Unit test for Automatic Background Indexing notifications, progress events, and UI pill integration.
"""
import os
import sys
import tempfile
import time
import numpy as np
import laspy

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QCoreApplication

from pointcloud_manager import PointCloudManager
from server import HybridServer
from app_window import MainWindow, BridgeSignaler


def create_synthetic_laz(path: str, num_points: int = 15000, with_itc: bool = True):
    hdr = laspy.LasHeader(point_format=3, version="1.4")
    hdr.offsets = [0.0, 0.0, 0.0]
    hdr.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(hdr)

    las.x = np.linspace(10.0, 200.0, num_points)
    las.y = np.linspace(10.0, 200.0, num_points)
    las.z = np.linspace(0.0, 30.0, num_points)
    las.red = (np.random.rand(num_points) * 65535).astype(np.uint16)
    las.green = (np.random.rand(num_points) * 65535).astype(np.uint16)
    las.blue = (np.random.rand(num_points) * 65535).astype(np.uint16)
    las.intensity = np.random.randint(0, 1000, num_points, dtype=np.uint16)
    las.classification = np.full(num_points, 1, dtype=np.uint8)

    if with_itc:
        las.add_extra_dim(laspy.ExtraBytesParams(name="itc", type=np.int32))
        tids = np.zeros(num_points, dtype=np.int32)
        # Assign tree IDs 1 to 5
        pts_per_tree = num_points // 5
        for i in range(5):
            tids[i * pts_per_tree:(i + 1) * pts_per_tree] = i + 1
        las.itc[:] = tids

    las.write(path)


def test_manager_indexing_callbacks():
    print("[TEST] test_manager_indexing_callbacks starting...")
    with tempfile.TemporaryDirectory() as tmpdir:
        laz_path = os.path.join(tmpdir, "test_trees.laz")
        create_synthetic_laz(laz_path, num_points=12000, with_itc=True)

        events_received = []
        mgr = PointCloudManager()
        mgr.add_indexing_callback(lambda ev: events_received.append(ev))

        # First load: no precomputed index exists, background indexer should start
        info = mgr.load_las_header(laz_path)
        assert info["total_points"] == 12000
        assert info["has_itc"] is True

        # Wait for background indexing thread to finish
        t0 = time.time()
        while time.time() - t0 < 8.0:
            if mgr.get_indexing_status().get("stage") == "ready" and mgr._chunk_index is not None and mgr._tree_index is not None:
                break
            time.sleep(0.05)

        status = mgr.get_indexing_status()
        assert status["has_chunk_index"] is True, "Chunk index should be built"
        assert status["has_tree_index"] is True, "Tree index should be built"
        assert status["tree_count"] >= 5, f"Should have indexed trees: {status['tree_count']}"

        # Verify sidecars written to disk
        assert os.path.exists(laz_path + ".chunk_idx.json"), "Sidecar .chunk_idx.json should exist"
        assert os.path.exists(laz_path + ".tree_idx.json"), "Sidecar .tree_idx.json should exist"

        # Verify events recorded
        event_types = [e["type"] for e in events_received]
        assert "started" in event_types, f"Started event should be fired: {event_types}"
        assert "completed" in event_types, f"Completed event should be fired: {event_types}"
        print(f"[PASS] Manager indexing events captured: {event_types}")

        # Test precomputed cache fast-path: second load should immediately report 'ready'
        events_second = []
        mgr2 = PointCloudManager()
        mgr2.add_indexing_callback(lambda ev: events_second.append(ev))
        mgr2.load_las_header(laz_path)
        sec_types = [e["type"] for e in events_second]
        assert "ready" in sec_types, f"Second load should immediately dispatch 'ready': {sec_types}"
        print(f"[PASS] Fast-path precomputed index load dispatches 'ready': {sec_types}")


def test_mainwindow_indexing_ui():
    print("[TEST] test_mainwindow_indexing_ui starting...")
    app = QApplication.instance() or QApplication(sys.argv)

    with tempfile.TemporaryDirectory() as tmpdir:
        potree_dir = os.path.join(tmpdir, "potree")
        os.makedirs(potree_dir, exist_ok=True)
        web_dir = os.path.join(os.path.dirname(__file__), "web")

        mgr = PointCloudManager()
        srv = HybridServer(potree_root=potree_dir, web_root=web_dir, manager=mgr, port=18942)
        win = MainWindow(server=srv, manager=mgr, potree_dir=potree_dir)
        win.show()

        # 1. Verify permanent status bar widget exists
        assert hasattr(win, "indexing_pill"), "indexing_pill should exist on MainWindow"
        assert hasattr(win, "lbl_indexing_icon"), "lbl_indexing_icon should exist"
        assert hasattr(win, "lbl_indexing_text"), "lbl_indexing_text should exist"
        assert hasattr(win, "indexing_progress_bar"), "indexing_progress_bar should exist"

        # 2. Verify tooltip contains exact requirements text
        tip = win.indexing_pill.toolTip()
        assert "Automatic Background Indexing" in tip, "Tooltip must explain Automatic Background Indexing"
        assert "Zero-Dependency Streaming Fallback" in tip, "Tooltip must explain Zero-Dependency Streaming Fallback"
        print("[PASS] Permanent status bar indexing pill created with architecture tooltip")

        # 3. Simulate started event (chunks)
        win.on_indexing_event({
            "type": "started",
            "stage": "chunks",
            "file": "forest.laz",
            "msg": "Automatic Background Indexing started..."
        })
        assert win.lbl_indexing_icon.text() == "⚡"
        assert not win.indexing_progress_bar.isHidden()
        print("[PASS] Pill shows indexing state and progress bar on started event")

        # 4. Simulate progress event
        win.on_indexing_event({
            "type": "progress",
            "stage": "chunks",
            "current": 3,
            "total": 10,
            "percent": 30,
            "msg": "Chunks 3/10 (30%)"
        })
        assert win.indexing_progress_bar.value() == 30
        assert "30%" in win.lbl_indexing_text.text()
        print("[PASS] Pill reflects real-time progress percentage")

        # 5. Simulate completed event
        win.on_indexing_event({
            "type": "completed",
            "stage": "done",
            "chunks": 10,
            "trees": 2450,
            "msg": "Tree spatial index ready: 2,450 trees"
        })
        assert win.lbl_indexing_icon.text() == "✓"
        assert win.indexing_progress_bar.isHidden()
        assert "2,450 trees" in win.lbl_indexing_text.text()
        print("[PASS] Pill reflects ready checkmark and tree count on completion")

        # 6. Simulate error/fallback event
        win.on_indexing_event({
            "type": "error",
            "stage": "error",
            "error": "Simulated disk issue",
            "msg": "Streaming fallback active"
        })
        assert win.lbl_indexing_icon.text() == "⚠"
        assert "Streaming Fallback Active" in win.lbl_indexing_text.text()
        print("[PASS] Pill gracefully falls back to warning badge on error")

        # 7. Test show_indexing_details_dialog
        from unittest.mock import patch
        with patch("PyQt6.QtWidgets.QMessageBox.information") as mock_info:
            win.show_indexing_details_dialog()
            assert mock_info.called, "show_indexing_details_dialog should call QMessageBox.information"
            args, _ = mock_info.call_args
            assert "Automatic Background Indexing" in args[2]
            assert "Zero-Dependency Streaming Fallback" in args[2]
            print("[PASS] Indexing details dialog displays architecture details accurately")


if __name__ == "__main__":
    test_manager_indexing_callbacks()
    test_mainwindow_indexing_ui()
    print("\nALL BACKGROUND INDEXING NOTIFICATION & UI TESTS PASSED!")
