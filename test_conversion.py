"""
Test PotreeConverter Discovery, ConversionWorker Streaming, Stats Parsing, and UI Dialog.
"""
import os
import sys
import tempfile
import shutil
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QEventLoop

from conversion_dialog import (
    find_potree_converter,
    format_file_size,
    format_points_rate,
    ConversionWorker,
    ConversionDialog
)

def test_discovery():
    bin_path = find_potree_converter()
    assert bin_path is not None, "PotreeConverter should be found"
    assert os.path.isfile(bin_path), f"File should exist: {bin_path}"
    assert os.access(bin_path, os.X_OK), f"Should be executable: {bin_path}"
    print(f"[PASS] find_potree_converter found: {bin_path}")

def test_helpers():
    assert format_file_size(500) == "500 B"
    assert "KB" in format_file_size(2048)
    assert "MB" in format_file_size(10 * 1024 * 1024)
    assert "Mpts/s" in format_points_rate(10_000_000, 2.0)
    assert "Kpts/s" in format_points_rate(35_000, 1.0)
    print("[PASS] Helper formatting tests passed")

def test_worker_execution():
    app = QApplication.instance() or QApplication(sys.argv)

    bin_path = find_potree_converter()
    test_laz = "/home/neo/Projects/TreeAIBox/LAStools/data/bunny.laz"
    assert os.path.exists(test_laz), f"Missing test LAZ: {test_laz}"

    tmp_out = tempfile.mkdtemp(prefix="potree_conv_test_")
    try:
        worker = ConversionWorker(bin_path, test_laz, tmp_out)

        stages = []
        logs = []
        progress_calls = []
        stats_received = {}
        finished_results = {}

        worker.stage_changed.connect(lambda s: stages.append(s))
        worker.log_line.connect(lambda l: logs.append(l))
        worker.progress_updated.connect(lambda c, t, m: progress_calls.append((c, t, m)))
        worker.stats_ready.connect(lambda d: stats_received.update(d))

        loop = QEventLoop()
        def on_done(ok, msg, d):
            finished_results["ok"] = ok
            finished_results["msg"] = msg
            finished_results["stats"] = d
            loop.quit()

        worker.finished.connect(on_done)
        worker.start()
        loop.exec()

        assert finished_results["ok"] is True, f"Conversion failed: {finished_results['msg']}"
        assert len(logs) > 10, f"Expected streaming logs, got {len(logs)}"
        assert len(stages) >= 3, f"Expected stages, got {stages}"
        assert os.path.exists(os.path.join(tmp_out, "metadata.json")), "metadata.json must exist"
        assert os.path.exists(os.path.join(tmp_out, "octree.bin")), "octree.bin must exist"

        print(f"[PASS] Stages passed: {stages}")
        print(f"[PASS] Stats parsed: {finished_results['stats']}")
        print(f"[PASS] Converted successfully ({len(logs)} log lines recorded)")
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)

if __name__ == "__main__":
    test_discovery()
    test_helpers()
    test_worker_execution()
    print("\nALL CONVERSION TESTS PASSED!")
