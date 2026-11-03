"""
Integration test for ConversionDialog UI, logging, stats population, and completion signal.
"""
import os
import sys
import tempfile
import shutil
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QEventLoop

from conversion_dialog import ConversionDialog, find_potree_converter

def test_conversion_dialog_lifecycle():
    app = QApplication.instance() or QApplication(sys.argv)

    converter_bin = find_potree_converter()
    assert converter_bin is not None

    test_laz = "/home/neo/Projects/TreeAIBox/LAStools/data/bunny.laz"
    tmp_out = tempfile.mkdtemp(prefix="dialog_test_")

    try:
        dialog = ConversionDialog(
            converter_bin=converter_bin,
            input_las=test_laz,
            out_dir=tmp_out,
            auto_start=False
        )

        # 1. Verify initial UI state
        assert dialog.lbl_badge.text() == "READY"
        assert dialog.btn_open.isEnabled() is False
        assert dialog.btn_cancel.isEnabled() is True
        assert dialog.log_text.toPlainText() == ""

        # 2. Start conversion and wait for completion
        completed_signal = {}
        loop = QEventLoop()

        def on_completed(out_dir, stats):
            completed_signal["out_dir"] = out_dir
            completed_signal["stats"] = stats
            loop.quit()

        dialog.conversion_completed.connect(on_completed)
        dialog.start_conversion()
        loop.exec()

        # 3. Verify final UI state
        assert dialog.lbl_badge.text() == "COMPLETE"
        assert dialog.btn_open.isEnabled() is True
        assert dialog.btn_cancel.isEnabled() is False
        assert len(dialog.log_text.toPlainText()) > 100

        # 4. Verify structured stats populated
        assert dialog.kpi_points.text() != "--"
        assert dialog.kpi_duration.text() != "--"
        assert dialog.kpi_size.text() != "--"
        print(f"[PASS] Dialog points KPI: {dialog.kpi_points.text()}")
        print(f"[PASS] Dialog duration KPI: {dialog.kpi_duration.text()}")
        print(f"[PASS] Dialog rate KPI: {dialog.kpi_rate.text()}")
        print(f"[PASS] Dialog size KPI: {dialog.kpi_size.text()}")

        # 5. Verify signal payload
        assert completed_signal["out_dir"] == tmp_out
        assert "#points" in completed_signal["stats"]

        # 6. Test copy log functionality
        dialog._copy_log()
        clipboard_text = QApplication.clipboard().text()
        assert len(clipboard_text) == len(dialog.log_text.toPlainText())

        print("[PASS] ConversionDialog UI lifecycle and stats verification successful!")
    finally:
        shutil.rmtree(tmp_out, ignore_errors=True)

if __name__ == "__main__":
    test_conversion_dialog_lifecycle()
