"""
Unit test for Console Panel and Indexing Pill Visibility in PotreeEditor.
"""
import os
import sys
import tempfile

from PyQt6.QtWidgets import QApplication

from pointcloud_manager import PointCloudManager
from settings_store import SettingsStore
from server import HybridServer
from app_window import MainWindow


def test_console_and_indexing_pill():
    print("[TEST] test_console_and_indexing_pill starting...")
    app = QApplication.instance() or QApplication(sys.argv)

    with tempfile.TemporaryDirectory() as tmpdir:
        potree_dir = os.path.join(tmpdir, "potree")
        os.makedirs(potree_dir, exist_ok=True)
        web_dir = os.path.join(os.path.dirname(__file__), "web")

        mgr = PointCloudManager()
        srv = HybridServer(potree_root=potree_dir, web_root=web_dir, manager=mgr, port=18943)
        win = MainWindow(server=srv, manager=mgr, potree_dir=potree_dir)
        win.show()

        # ==========================================
        # 1. Test Indexing Pill Visibility Lifecycle
        # ==========================================
        # Pill should be hidden by default upon initialization
        assert not win.indexing_pill.isVisible(), "Indexing pill must be hidden by default when idle"

        # Pill becomes visible when indexing starts
        win.on_indexing_event({
            "type": "started",
            "stage": "chunks",
            "file": "forest.laz",
            "msg": "Automatic Background Indexing started..."
        })
        assert win.indexing_pill.isVisible(), "Pill must be visible when indexing starts"

        # Pill remains visible during progress
        win.on_indexing_event({
            "type": "progress",
            "stage": "chunks",
            "current": 5,
            "total": 10,
            "percent": 50,
            "msg": "Indexing 5/10 (50%)"
        })
        assert win.indexing_pill.isVisible(), "Pill must remain visible during progress"

        # Pill stays visible upon completion
        win.on_indexing_event({
            "type": "completed",
            "stage": "done",
            "chunks": 10,
            "trees": 1500,
            "msg": "Tree spatial index ready: 1,500 trees"
        })
        assert win.indexing_pill.isVisible(), "Pill should be visible immediately after completion"

        # When idle, calling _hide_indexing_pill_if_idle hides the pill
        win._hide_indexing_pill_if_idle()
        assert not win.indexing_pill.isVisible(), "Pill should hide when idle"

        # Precomputed ready event should not make pill visible
        win.on_indexing_event({
            "type": "ready",
            "stage": "ready",
            "trees": 1500,
            "msg": "Tree spatial index loaded from disk"
        })
        assert not win.indexing_pill.isVisible(), "Pill should stay hidden for precomputed index load"
        print("[PASS] Indexing pill visibility lifecycle verified.")

        # ==========================================
        # 2. Test Console Panel & Step Logging
        # ==========================================
        assert hasattr(win, "console_panel"), "MainWindow must have console_panel"
        assert hasattr(win, "txt_console"), "MainWindow must have txt_console"
        assert hasattr(win, "btn_toggle_console"), "MainWindow must have btn_toggle_console in status bar"
        assert hasattr(win, "lbl_status_live_msg"), "MainWindow must have lbl_status_live_msg in status bar"

        # The console is the window into long-running processes, so it opens
        # with the app at a useful height.
        assert win.console_panel.isVisible(), "Console panel should start open"
        assert "▼ Hide Console" in win.btn_toggle_console.text()
        assert win.main_vsplitter.sizes()[1] >= 200, "Console starts tall enough to read"

        # Collapse and restore from the status bar button
        win.toggle_console_panel()
        assert not win.console_panel.isVisible(), "Console panel should collapse on toggle"
        assert "▲ Console" in win.btn_toggle_console.text()

        win.toggle_console_panel()
        assert win.console_panel.isVisible(), "Console panel should be visible after toggle_console_panel()"
        assert "▼ Hide Console" in win.btn_toggle_console.text()

        # Log console messages
        win.log_console("AI", "Testing AI segmentation message", level="info", step=1, total=5)
        text = win.txt_console.toPlainText()
        assert "[AI]" in text
        assert "[1/5]" in text
        assert "Testing AI segmentation message" in text
        assert "Testing AI segmentation message" in win.lbl_status_live_msg.text()

        # Expand console
        initial_h = win.main_vsplitter.sizes()[1]
        win._toggle_console_expand()
        expanded_h = win.main_vsplitter.sizes()[1]
        assert expanded_h >= initial_h, "Splitter size should increase on expand"
        win._toggle_console_expand()
        restored_h = win.main_vsplitter.sizes()[1]
        assert restored_h < expanded_h, "Splitter size should decrease on restore"

        # Open/closed state and drawer height are remembered with the settings
        win._save_settings()
        stored = SettingsStore(win.settings_store.path)
        assert stored.get("console_open") is True, "Open console is persisted"
        assert list(stored.get("console_split")) == list(win.main_vsplitter.sizes()), \
            "Drawer height is persisted"
        stored.close()

        # Clear console
        win._clear_console_logs()
        assert win.txt_console.toPlainText() == ""
        assert win.lbl_console_status.text() == "Ready"

        # Test unread count when closed
        win.toggle_console_panel()
        assert not win.console_panel.isVisible()
        win.log_console("STAND", "Stand filter applied: 5,000 points retained", level="success")
        assert win._unread_logs == 1
        assert "(1)" in win.btn_toggle_console.text()

        # Opening console resets unread counter
        win.toggle_console_panel()
        assert win._unread_logs == 0
        assert "(1)" not in win.btn_toggle_console.text()

        # Test _on_segmentation_step event processing
        win._on_segmentation_step({
            "stage": "extract_roi",
            "category": "AI",
            "msg": "Extracting 15000 points from master LAS for ITC 2969...",
            "step": 1,
            "total": 5
        })
        text = win.txt_console.toPlainText()
        assert "[1/5]" in text
        assert "Extracting 15000 points" in text
        assert "[1/5]" in win.lbl_status_live_msg.text()

        print("[PASS] Console panel and step logging verified.")


if __name__ == "__main__":
    test_console_and_indexing_pill()
    print("\nALL CONSOLE AND INDEXING PILL TESTS PASSED!")
