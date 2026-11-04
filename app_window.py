"""
Main Application Window for Potree Hybrid Viewer.
Combines:
- QWebEngineView hosting the Potree WebGL streaming renderer.
- Native PyQt6 sidebar with Optimistic Visual Staging, ITC assignment, noise tagging, and deferred LAS commit.
- Top Ribbon Category Bar (File, Point, Settings) with quick tools and HUD integration.
- High-performance streaming deferred commit for massive 200M+ point clouds.
"""

import os
import re
import sys
import json
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox, QProgressBar, QSplitter,
    QFrame, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QSlider, QGroupBox, QStatusBar,
    QListWidget, QListWidgetItem, QPlainTextEdit,
    QScrollArea, QStackedWidget, QButtonGroup, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QProgressDialog, QSizePolicy, QTabWidget, QGridLayout,
    QDialog
)
from PyQt6.QtCore import Qt, QUrl, QStandardPaths, QByteArray, pyqtSignal, QObject, QThread, QTimer
from PyQt6.QtGui import QFont, QIcon, QColor, QAction, QActionGroup, QPixmap
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings

from pointcloud_manager import PointCloudManager
from server import HybridServer
from settings_store import SettingsStore
from conversion_dialog import ConversionDialog, find_potree_converter


class BridgeSignaler(QObject):
    selection_changed = pyqtSignal(dict)
    point_picked = pyqtSignal(dict)
    tree_view_requested = pyqtSignal(dict)
    tree_loaded = pyqtSignal(dict)
    tree_stand_loaded = pyqtSignal(dict)
    tree_filter_selection_changed = pyqtSignal(dict)
    tree_filter_mode_toggled = pyqtSignal(dict)
    edit_applied = pyqtSignal(dict)
    edit_staged = pyqtSignal(dict)
    edit_removed = pyqtSignal(dict)
    edits_cleared = pyqtSignal(dict)
    indexing_event = pyqtSignal(dict)
    segmentation_step = pyqtSignal(dict)
    console_log = pyqtSignal(dict)


class CommitWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, manager: PointCloudManager, input_path: str, output_path: str):
        super().__init__()
        self.manager = manager
        self.input_path = input_path
        self.output_path = output_path

    def run(self):
        try:
            def cb(processed: int, total: int, msg: str):
                self.progress.emit(processed, total, msg)

            result = self.manager.commit_staged_edits(
                self.input_path,
                self.output_path,
                progress_cb=cb,
                chunk_size=1_000_000
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ConfirmSplitDialog(QDialog):
    """Dialog to review and confirm individual Tree IDs when ForestFormer3D splits a tree or segments unassigned stand points."""
    def __init__(self, parent, target_itc: int, n_trees: int, seg_las_path: str, max_itc: int, is_target_included: bool = True):
        super().__init__(parent)
        self.setWindowTitle(f"Confirm ForestFormer3D Tree Split: Tree #{target_itc}" if is_target_included else "Confirm ForestFormer3D Segmentation: New Trees")
        self.resize(460, 360)
        self.target_itc = target_itc
        self.n_trees = n_trees
        self.seg_las_path = seg_las_path
        self.max_itc = max_itc
        self.is_target_included = is_target_included
        self.id_spinners = []
        self.action_choice = "cancel"
        self.init_ui()

    def init_ui(self):
        v = QVBoxLayout(self)
        v.setSpacing(12)

        if self.is_target_included:
            lbl_title = QLabel(f"<b>ForestFormer3D Split: Tree #{self.target_itc}</b>")
            lbl_title.setStyleSheet("font-size: 13px; color: #10b981;")
            v.addWidget(lbl_title)

            lbl_desc = QLabel(
                f"ForestFormer3D detected <b>{self.n_trees} individual trees</b> in Tree #{self.target_itc}.<br>"
                "Assign Tree IDs below to update the master point cloud:"
            )
        else:
            lbl_title = QLabel(f"<b>ForestFormer3D Segmentation: New Trees (Stand Points)</b>")
            lbl_title.setStyleSheet("font-size: 13px; color: #10b981;")
            v.addWidget(lbl_title)

            lbl_desc = QLabel(
                f"ForestFormer3D detected <b>{self.n_trees} new individual trees</b> in the selected stand points near Tree #{self.target_itc}.<br>"
                "Assign new Tree IDs below to update the master point cloud:"
            )
        lbl_desc.setStyleSheet("color: #cbd5e1; font-size: 11px;")
        v.addWidget(lbl_desc)

        grp = QGroupBox("Tree ID Assignments")
        v_grp = QVBoxLayout(grp)
        v_grp.setSpacing(8)

        for i in range(self.n_trees):
            h = QHBoxLayout()
            lbl = QLabel(f"Tree {i + 1} (Cluster {i}):")
            lbl.setStyleSheet("font-weight: 600; color: #f8fafc;")
            spin = QSpinBox()
            spin.setRange(1, 9999999)
            if i == 0 and self.is_target_included:
                spin.setValue(self.target_itc)
            else:
                offset = i if self.is_target_included else (i + 1)
                spin.setValue(self.max_itc + offset)
            spin.setStyleSheet("padding: 4px 8px; font-size: 12px;")
            self.id_spinners.append(spin)
            h.addWidget(lbl)
            h.addWidget(spin)
            v_grp.addLayout(h)

        lbl_ground = QLabel("• Surrounding non-tree points will be classified as Ground (Class 2, ITC 0)")
        lbl_ground.setStyleSheet("color: #94a3b8; font-size: 10px; font-style: italic;")
        v_grp.addWidget(lbl_ground)

        v.addWidget(grp)

        h_btns = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("padding: 6px 12px; font-size: 11px;")
        btn_cancel.clicked.connect(self.reject)

        self.btn_stage = QPushButton("Stage for LAS Commit")
        self.btn_stage.setStyleSheet("padding: 6px 12px; background-color: #334155; color: #e2e8f0; font-size: 11px; border-radius: 4px;")
        self.btn_stage.setToolTip("Add this split to the Staged Edits list to review and commit later.")
        self.btn_stage.clicked.connect(lambda: self._finish("stage"))

        self.btn_commit_octree = QPushButton("Commit to Octree Now (Instant)")
        self.btn_commit_octree.setStyleSheet("padding: 6px 14px; background-color: #0284c7; color: white; font-weight: 700; font-size: 11px; border-radius: 4px;")
        self.btn_commit_octree.setToolTip("Patch octree.bin and metadata.json in place immediately (~2s) and refresh viewer.")
        self.btn_commit_octree.clicked.connect(lambda: self._finish("octree"))

        self.btn_commit_both = QPushButton("Commit to File & Octree...")
        self.btn_commit_both.setStyleSheet("padding: 6px 14px; background-color: #059669; color: white; font-weight: 700; font-size: 11px; border-radius: 4px;")
        self.btn_commit_both.setToolTip("Save new LAS/LAZ file AND update the loaded Potree octree in place.")
        self.btn_commit_both.clicked.connect(lambda: self._finish("both"))

        h_btns.addWidget(btn_cancel)
        h_btns.addStretch()
        h_btns.addWidget(self.btn_stage)
        h_btns.addWidget(self.btn_commit_octree)
        h_btns.addWidget(self.btn_commit_both)
        v.addLayout(h_btns)

    def _finish(self, choice: str):
        self.action_choice = choice
        self.accept()

    def get_id_mapping(self) -> dict:
        mapping = {}
        for i, spin in enumerate(self.id_spinners):
            mapping[i] = spin.value()
        return mapping


class EditHistoryDialog(QDialog):
    """Dialog to review logged edits and revert/undo changes in the active dataset."""

    def __init__(self, parent, settings_store: SettingsStore, potree_dir: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Edit History & Undo / Revert")
        self.resize(820, 500)
        self.settings_store = settings_store
        self.potree_dir = potree_dir
        self.selected_edit = None
        self.edits = []
        self.init_ui()
        self.load_history()

    def init_ui(self):
        v = QVBoxLayout(self)
        v.setSpacing(10)

        lbl_title = QLabel("<b>Point Cloud Edit History (Database Log)</b>")
        lbl_title.setStyleSheet("font-size: 14px; color: #10b981;")
        v.addWidget(lbl_title)

        lbl_desc = QLabel(
            "Below is the persistent log of all tree splits and octree edits recorded in <code>settings.db</code>.<br>"
            "You can review previous actions and revert any committed split in-place."
        )
        lbl_desc.setStyleSheet("color: #94a3b8; font-size: 11px;")
        v.addWidget(lbl_desc)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "ID", "Timestamp", "Type", "Target Tree", "Assigned IDs", "Points Changed", "Status"
        ])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        v.addWidget(self.table)

        self.lbl_details = QLabel("Select an edit above to view details.")
        self.lbl_details.setStyleSheet("font-size: 11px; color: #cbd5e1; background-color: #1e293b; padding: 8px; border-radius: 4px;")
        self.lbl_details.setWordWrap(True)
        v.addWidget(self.lbl_details)

        h_btns = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.load_history)
        h_btns.addWidget(self.btn_refresh)

        h_btns.addStretch()

        self.btn_revert = QPushButton("Revert Selected Edit in Octree")
        self.btn_revert.setStyleSheet("padding: 6px 14px; background-color: #dc2626; color: white; font-weight: 700; border-radius: 4px;")
        self.btn_revert.setEnabled(False)
        self.btn_revert.clicked.connect(self._revert_selected)
        h_btns.addWidget(self.btn_revert)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        h_btns.addWidget(btn_close)

        v.addLayout(h_btns)

    def load_history(self):
        self.edits = self.settings_store.get_edit_history(self.potree_dir)
        self.table.setRowCount(len(self.edits))
        for row, rec in enumerate(self.edits):
            self.table.setItem(row, 0, QTableWidgetItem(f"#{rec.get('id')}"))
            self.table.setItem(row, 1, QTableWidgetItem(str(rec.get("timestamp", ""))[:19]))
            self.table.setItem(row, 2, QTableWidgetItem(str(rec.get("edit_type", ""))))
            self.table.setItem(row, 3, QTableWidgetItem(f"Tree #{rec.get('target_itc')}" if rec.get("target_itc") else "-"))
            assigned = rec.get("assigned_itcs", [])
            self.table.setItem(row, 4, QTableWidgetItem(str(assigned)))
            pts = rec.get("points_affected", 0)
            self.table.setItem(row, 5, QTableWidgetItem(f"{pts:,}"))

            status = rec.get("status", "committed")
            item_status = QTableWidgetItem(status.capitalize())
            if status == "committed":
                item_status.setForeground(QColor("#10b981"))
            else:
                item_status.setForeground(QColor("#94a3b8"))
            self.table.setItem(row, 6, item_status)

        self.btn_revert.setEnabled(False)
        self.lbl_details.setText(f"Loaded {len(self.edits)} edit records from database.")

    def _on_row_selected(self):
        rows = self.table.selectedIndexes()
        if not rows:
            self.btn_revert.setEnabled(False)
            return
        row = rows[0].row()
        if 0 <= row < len(self.edits):
            rec = self.edits[row]
            self.selected_edit = rec
            status = rec.get("status", "committed")
            self.btn_revert.setEnabled(status == "committed")
            seg_path = rec.get("segmented_las_path", "")
            self.lbl_details.setText(
                f"<b>Edit #{rec.get('id')}</b> | Timestamp: {rec.get('timestamp')}<br>"
                f"• Target Tree: #{rec.get('target_itc')} &rarr; Assigned IDs: {rec.get('assigned_itcs')}<br>"
                f"• Points Changed: {rec.get('points_affected', 0):,} | Status: <b>{status}</b><br>"
                f"• Segmented LAS: <code>{os.path.basename(seg_path) if seg_path else 'None'}</code>"
            )

    def _revert_selected(self):
        if not self.selected_edit:
            return
        rec = self.selected_edit
        edit_id = rec.get("id")
        target_itc = rec.get("target_itc")
        target_desc = f"Tree #{target_itc}" if target_itc else "Stand Segmentation (Unassigned Points)"
        reply = QMessageBox.question(
            self,
            "Confirm Revert",
            f"Are you sure you want to revert Edit #{edit_id} ({target_desc})?\n\n"
            f"This will restore the original points in the octree in-place and mark this edit as reverted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        parent = self.parent()
        potree_dir = self.potree_dir or getattr(parent, "current_potree_dir", None)
        if not potree_dir or not os.path.isdir(potree_dir):
            QMessageBox.warning(self, "No Octree", "No active Potree octree folder loaded to revert.")
            return

        try:
            parent.log_console("OCTREE", f"Reverting Edit #{edit_id} ({target_desc}) in octree...", level="info")
            res = parent.manager.revert_potree_octree(potree_dir, rec)
            rev_pts = res.get("reverted_points", 0)
            self.settings_store.mark_edit_reverted(edit_id)
            self.settings_store.revert_tree_edits_by_edit_id(edit_id)
            if target_itc:
                self.settings_store.revert_tree_edits_by_parent_itc(target_itc)
                parent.manager.clear_tree_cache(target_itc)
                parent._run_js(f"if (window.invalidateTreeCache) window.invalidateTreeCache({target_itc});")

            # Invalidate caches in Python and JavaScript
            for a_id in rec.get("assigned_itcs", []):
                parent.manager.clear_tree_cache(int(a_id))
                parent._run_js(f"if (window.invalidateTreeCache) window.invalidateTreeCache({int(a_id)});")
            parent.manager.sync_tree_edits_from_db(self.settings_store)

            # Remove from staged edits if currently present
            parent.manager.staged_edits = [
                e for e in parent.manager.staged_edits
                if e.get("db_edit_id") != edit_id and (target_itc is None or e.get("target_itc") != target_itc)
            ]
            parent.refresh_staged_edits_table()
            parent._run_js("window.reloadCurrentPointCloud();")
            if target_itc:
                parent._run_js(f"window.viewTreeById({target_itc});")

            parent.log_console("OCTREE", f"✓ Reverted {rev_pts:,} points in octree.bin for Edit #{edit_id}.", level="success")
            parent.log_console("DB", f"✓ Marked Edit #{edit_id} as reverted in settings.db.", level="success")

            QMessageBox.information(
                self,
                "Revert Complete",
                f"Successfully reverted Edit #{edit_id} in-place!\n\n"
                f"• Tree #{target_itc} restored ({rev_pts:,} points updated).\n"
                f"• Octree reloaded in viewer."
            )
            self.load_history()
        except Exception as e:
            QMessageBox.critical(self, "Revert Failed", f"Failed to revert edit in octree:\n\n{e}")


class MainWindow(QMainWindow):
    def __init__(self, server: HybridServer, manager: PointCloudManager, potree_dir: str,
                 initial_path: Optional[str] = None):
        super().__init__()
        self.server = server
        self.manager = manager
        self.potree_dir = potree_dir

        self.current_las_path: Optional[str] = None
        self.current_potree_url: Optional[str] = None
        self.current_potree_dir: Optional[str] = None
        self.commit_dialog: Optional[QProgressDialog] = None
        self.commit_worker: Optional[CommitWorker] = None
        self.conversion_dialog: Optional[ConversionDialog] = None
        self._active_tool = "nav"
        self._tool_buttons: dict = {}

        # Persisted settings (SQLite key/value in the project folder)
        potree_dir_abs = os.path.abspath(potree_dir) if potree_dir else os.getcwd()
        self._settings_store = SettingsStore(
            os.path.join(os.path.dirname(potree_dir_abs), "settings.db"))
        self._settings_applied = False
        self._settings_timer = QTimer(self)
        self._settings_timer.setSingleShot(True)
        self._settings_timer.setInterval(450)
        self._settings_timer.timeout.connect(self._save_settings)

        self.signaler = BridgeSignaler()
        self.signaler.selection_changed.connect(self.on_selection_changed)
        self.signaler.point_picked.connect(self.on_point_picked)
        self.signaler.tree_view_requested.connect(self.on_tree_view_requested)
        self.signaler.tree_loaded.connect(self.on_tree_loaded)
        self.signaler.tree_stand_loaded.connect(self._on_tree_stand_loaded)
        self.signaler.tree_filter_selection_changed.connect(self._on_tree_filter_selection_changed)
        self.signaler.tree_filter_mode_toggled.connect(self._on_tree_filter_mode_toggled_from_hud)
        self.signaler.edit_applied.connect(self.on_edit_applied)
        self.signaler.edit_staged.connect(self.on_edit_staged)
        self.signaler.edit_removed.connect(self.on_edit_removed)
        self.signaler.edits_cleared.connect(self.on_edits_cleared)
        self.signaler.indexing_event.connect(self.on_indexing_event)
        self.signaler.segmentation_step.connect(self._on_segmentation_step)
        self.signaler.console_log.connect(self._on_console_log_event)

        # Wire server events to Qt signals (thread-safe UI updates)
        self.server.on_event(self._on_server_event)

        # Register manager background indexing callback
        self.manager.add_indexing_callback(lambda data: self.signaler.indexing_event.emit(data))
        self.manager.set_settings_store(self.settings_store)

        self.init_ui()
        self._wire_settings_persistence()

        # Auto-load a cloud given on the command line (runs after the viewer
        # page exists; _run_js queues JS until loadFinished).
        if initial_path:
            QTimer.singleShot(500, lambda: self._load_initial_path(initial_path))

    @property
    def settings_store(self) -> SettingsStore:
        potree_dir = getattr(self, "current_potree_dir", None) or getattr(self, "potree_dir", None)
        if potree_dir and os.path.exists(potree_dir):
            db_dir = os.path.dirname(os.path.abspath(potree_dir))
        else:
            db_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(db_dir, "settings.db")
        if not hasattr(self, "_settings_store") or self._settings_store.path != db_path:
            self._settings_store = SettingsStore(db_path)
        return self._settings_store

    @settings_store.setter
    def settings_store(self, store: SettingsStore):
        self._settings_store = store

    def open_edit_history_dialog(self):
        """Open the Edit History & Revert dialog."""
        potree_dir = getattr(self, "current_potree_dir", None) or getattr(self, "potree_dir", None)
        dlg = EditHistoryDialog(self, self.settings_store, potree_dir)
        dlg.exec()

    def _load_initial_path(self, path: str):
        path = os.path.abspath(path)
        if os.path.isdir(path):
            for entry in ("metadata.json", "cloud.js"):
                if os.path.exists(os.path.join(path, entry)):
                    self.open_potree_path(os.path.join(path, entry))
                    return
            # Search immediate subdirectories for converted cloud
            try:
                for sub in sorted(os.listdir(path)):
                    sub_path = os.path.join(path, sub)
                    if os.path.isdir(sub_path):
                        for entry in ("metadata.json", "cloud.js"):
                            if os.path.exists(os.path.join(sub_path, entry)):
                                self.open_potree_path(os.path.join(sub_path, entry))
                                return
                # Search for LAS/LAZ files in folder
                for f in sorted(os.listdir(path)):
                    if f.lower().endswith((".laz", ".las")):
                        self.open_las_path(os.path.join(path, f), offer_conversion=False)
                        return
            except Exception as e:
                print(f"[MainWindow] Error reading directory {path}: {e}")
        elif os.path.isfile(path):
            self.open_las_path(path, offer_conversion=False)

    def _on_server_event(self, event_name: str, data: dict):
        if event_name == "selection_changed":
            self.signaler.selection_changed.emit(data)
        elif event_name == "point_picked":
            self.signaler.point_picked.emit(data)
        elif event_name in ("tree_view_requested", "view_tree_requested"):
            self.signaler.tree_view_requested.emit(data)
        elif event_name == "tree_loaded":
            self.signaler.tree_loaded.emit(data)
        elif event_name == "tree_stand_loaded":
            self.signaler.tree_stand_loaded.emit(data)
        elif event_name == "tree_filter_selection_changed":
            self.signaler.tree_filter_selection_changed.emit(data)
        elif event_name == "tree_filter_mode_toggled":
            self.signaler.tree_filter_mode_toggled.emit(data)
        elif event_name == "edit_applied":
            self.signaler.edit_applied.emit(data)
        elif event_name == "edit_staged":
            self.signaler.edit_staged.emit(data)
        elif event_name == "edit_removed":
            self.signaler.edit_removed.emit(data)
        elif event_name == "edits_cleared":
            self.signaler.edits_cleared.emit(data)
        elif event_name == "segmentation_step":
            self.signaler.segmentation_step.emit(data)
        elif event_name == "console_log":
            self.signaler.console_log.emit(data)

    def init_ui(self):
        self.setWindowTitle("Potree Hybrid Editor (LOD Streaming + Optimistic Visual Staging)")
        self.resize(1440, 920)

        # Build application menu bar (hidden in favor of top Ribbon Category Bar)
        self.init_menu_bar()
        self.menuBar().setVisible(False)

        # Dark theme styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e24;
                color: #e2e8f0;
            }
            QMenuBar {
                background-color: #111827;
                color: #e2e8f0;
                border-bottom: 1px solid #334155;
                padding: 3px 6px;
                font-size: 13px;
            }
            QMenuBar::item {
                background: transparent;
                padding: 4px 10px;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background-color: #1e293b;
                color: #60a5fa;
            }
            QMenu {
                background-color: #181f2a;
                color: #f1f5f9;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 4px;
            }
            QMenu::item {
                padding: 5px 22px 5px 12px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #2563eb;
                color: #ffffff;
            }
            QMenu::separator {
                height: 1px;
                background-color: #334155;
                margin: 4px 8px;
            }
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 10px;
                margin-top: 16px;
                padding: 26px 10px 12px 10px;
                font-weight: 700;
                color: #60a5fa;
                font-size: 10px;
                letter-spacing: 1px;
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a212e, stop:1 #161b22);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                top: -3px;
                padding: 0 6px;
                background-color: transparent;
            }
            QPushButton {
                background-color: #2563eb;
                color: #ffffff;
                border: 1px solid #3b82f6;
                border-radius: 6px;
                padding: 7px 14px;
                font-weight: 500;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
            QPushButton:pressed {
                background-color: #1e40af;
            }
            QPushButton#btnSecondary {
                background-color: #374151;
                border-color: #4b5563;
                color: #e5e7eb;
            }
            QPushButton#btnSecondary:hover {
                background-color: #4b5563;
            }
            QPushButton#btnDanger {
                background-color: #b91c1c;
                border-color: #dc2626;
                color: #fef2f2;
            }
            QPushButton#btnDanger:hover {
                background-color: #991b1b;
            }
            QPushButton#btnSuccess {
                background-color: #059669;
                border-color: #10b981;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton#btnSuccess:hover {
                background-color: #047857;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 5px;
                color: #f1f5f9;
                padding: 4px 8px;
                font-size: 13px;
            }
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #3b82f6;
            }
            QSlider::groove:horizontal {
                height: 5px;
                background: #334155;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #3b82f6;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #60a5fa;
                width: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
            }
            QStatusBar {
                background-color: #0f172a;
                color: #94a3b8;
                border-top: 1px solid #1e293b;
                font-size: 12px;
            }
            /* Ribbon Category Header Bar */
            QFrame#ribbonCategoryBar {
                background-color: #0b0f17;
                border-bottom: 1px solid #1e293b;
                min-height: 30px;
                max-height: 30px;
            }
            QPushButton.ribbonCategoryTab {
                background-color: transparent;
                border: none;
                border-bottom: 2px solid transparent;
                color: #94a3b8;
                font-size: 12px;
                font-weight: 600;
                padding: 4px 14px;
                border-radius: 0px;
            }
            QPushButton.ribbonCategoryTab:hover {
                color: #f8fafc;
                background-color: rgba(255, 255, 255, 0.04);
            }
            QPushButton.ribbonCategoryTab:checked {
                color: #38bdf8;
                border-bottom: 2px solid #38bdf8;
                background-color: #161f30;
            }
            /* Ribbon Content Stack */
            QStackedWidget#ribbonStack {
                background-color: #131b26;
                border-bottom: 1px solid #233147;
                min-height: 92px;
                max-height: 92px;
            }
            QFrame.ribbonGroup {
                background-color: #182232;
                border: 1px solid #283952;
                border-radius: 6px;
            }
            QLabel.ribbonTitle {
                font-size: 10px;
                font-weight: 700;
                color: #64748b;
                letter-spacing: 0.06em;
            }
            QLabel.ribbonLabel {
                font-size: 11px;
                color: #94a3b8;
                font-weight: 500;
            }
            QComboBox.ribbonCombo {
                background-color: #0b111c;
                border: 1px solid #2b3d56;
                border-radius: 4px;
                color: #f8fafc;
                padding: 2px 6px;
                font-size: 11px;
                min-width: 85px;
                max-height: 22px;
            }
            QComboBox.ribbonCombo:hover {
                border-color: #38bdf8;
            }
            QDoubleSpinBox.ribbonSpin {
                background-color: #0b111c;
                border: 1px solid #2b3d56;
                border-radius: 4px;
                color: #f8fafc;
                padding: 2px 4px;
                font-size: 11px;
                max-height: 22px;
            }
            QPushButton.ribbonToolBtn {
                background-color: #1f2c40;
                border: 1px solid #334764;
                color: #f1f5f9;
                font-size: 11px;
                font-weight: 500;
                padding: 4px 10px;
                border-radius: 4px;
            }
            QPushButton.ribbonToolBtn:hover {
                background-color: #2b3e5a;
                border-color: #38bdf8;
            }
            QPushButton.ribbonToolBtn:checked {
                background-color: #155e75;
                border-color: #38bdf8;
                color: #ffffff;
            }
            QCheckBox {
                color: #e2e8f0;
                font-size: 11px;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border-radius: 3px;
                border: 1px solid #475569;
                background-color: #0f172a;
            }
            QCheckBox::indicator:hover {
                border-color: #38bdf8;
            }
            QCheckBox::indicator:checked {
                background-color: #0284c7;
                border-color: #38bdf8;
            }
            /* Staged Edits Table */
            QTableWidget {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 8px;
                gridline-color: #1e293b;
                color: #cbd5e1;
                font-size: 12px;
                padding: 2px;
            }
            QTableWidget::item {
                padding: 3px 4px;
                border-radius: 4px;
            }
            QTableWidget::item:hover {
                background-color: #1e293b;
            }
            QTableWidget::item:selected {
                background-color: #1d4ed8;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: transparent;
                color: #64748b;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 0.5px;
                border: none;
                border-bottom: 1px solid #1e293b;
                padding: 4px;
            }
            /* Sidebar TabWidget */
            QTabWidget::pane {
                border: 1px solid #30363d;
                background-color: #111827;
                border-radius: 6px;
            }
            QTabBar::tab {
                background-color: #0d1117;
                color: #8b949e;
                padding: 6px 12px;
                font-weight: 600;
                font-size: 11px;
                border: 1px solid #30363d;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #1e293b;
                color: #58a6ff;
                border-bottom: 2px solid #38bdf8;
            }
            QTabBar::tab:hover:!selected {
                background-color: #161f2e;
                color: #c9d1d9;
            }
        """)

        # Main Central Container with Ribbon on Top
        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        # 1. Top Ribbon Bar (Fixed height, never stretches vertically)
        ribbon_bar = self.create_ribbon_bar()
        central_layout.addWidget(ribbon_bar, 0)

        # 2. Splitter (Sidebar + Potree WebGL Viewport)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter = splitter

        sidebar = self.create_sidebar()
        sidebar.setMinimumWidth(340)
        sidebar.setMaximumWidth(450)

        # QWebEngineView setup
        self.web_view = QWebEngineView()
        self._page_ready = False
        self._js_queue: list = []
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)

        # Point WebEngine to the local hybrid server
        viewer_url = f"http://127.0.0.1:{self.server.port}/index.html"
        self.web_view.loadFinished.connect(self._on_page_load_finished)
        self.web_view.load(QUrl(viewer_url))

        splitter.addWidget(sidebar)
        splitter.addWidget(self.web_view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        # Open with the sidebar at its full width (persisted sizes take over
        # after the first relaunch).
        splitter.setSizes([450, 1200])

        # Main vertical splitter: top is sidebar+webview, bottom is resizable console drawer
        self.main_vsplitter = QSplitter(Qt.Orientation.Vertical)
        self.main_vsplitter.setObjectName("mainVSplitter")
        self.main_vsplitter.addWidget(splitter)

        # Activity & Console Log Panel
        self.console_panel = self._create_console_panel()
        self.main_vsplitter.addWidget(self.console_panel)

        self.main_vsplitter.setStretchFactor(0, 1)
        self.main_vsplitter.setStretchFactor(1, 0)
        self.main_vsplitter.setCollapsible(0, False)
        self.main_vsplitter.setCollapsible(1, True)
        self.console_panel.setVisible(False)

        central_layout.addWidget(self.main_vsplitter, 1)
        central_layout.setStretchFactor(ribbon_bar, 0)
        central_layout.setStretchFactor(self.main_vsplitter, 1)
        self.setCentralWidget(central_widget)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.btn_toggle_console = QPushButton("▲ Console")
        self.btn_toggle_console.setObjectName("btnToggleConsole")
        self.btn_toggle_console.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_console.setToolTip("Show activity & AI segmentation console logs (resizable)")
        self.btn_toggle_console.setStyleSheet("""
            QPushButton#btnToggleConsole {
                background-color: #1e293b;
                color: #38bdf8;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 2px 10px;
                font-size: 11px;
                font-weight: 700;
            }
            QPushButton#btnToggleConsole:hover {
                background-color: #334155;
                color: #f8fafc;
                border-color: #0284c7;
            }
        """)
        self.btn_toggle_console.clicked.connect(self.toggle_console_panel)
        self.status_bar.addWidget(self.btn_toggle_console)

        self.lbl_status_live_msg = QLabel("Ready. Open a point cloud or Potree dataset to begin.")
        self.lbl_status_live_msg.setStyleSheet("color: #94a3b8; font-size: 11px; padding-left: 8px;")
        self.status_bar.addWidget(self.lbl_status_live_msg, 1)

        # Indexing Indicator Pill (only visible when actively indexing)
        self._init_indexing_pill()

    # -------------------------------------------------------------------------
    # Activity & AI Segmentation Console
    # -------------------------------------------------------------------------
    def _create_console_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("consolePanel")
        panel.setStyleSheet("""
            QFrame#consolePanel {
                background-color: #0b0f19;
                border-top: 2px solid #1e293b;
            }
        """)
        v = QVBoxLayout(panel)
        v.setContentsMargins(8, 4, 8, 4)
        v.setSpacing(4)

        # Header toolbar
        h_head = QHBoxLayout()
        h_head.setContentsMargins(4, 2, 4, 2)
        h_head.setSpacing(8)

        lbl_icon = QLabel("📟")
        lbl_icon.setStyleSheet("font-size: 13px; background: transparent;")

        lbl_title = QLabel("Activity & AI Segmentation Console")
        lbl_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #f8fafc; background: transparent;")

        self.lbl_console_status = QLabel("Ready")
        self.lbl_console_status.setStyleSheet("""
            background-color: #1e293b;
            color: #38bdf8;
            border-radius: 3px;
            padding: 2px 6px;
            font-size: 10px;
            font-weight: 600;
        """)

        h_head.addWidget(lbl_icon)
        h_head.addWidget(lbl_title)
        h_head.addWidget(self.lbl_console_status)
        h_head.addStretch()

        btn_copy = QPushButton("Copy")
        btn_copy.setToolTip("Copy entire console log to clipboard")
        btn_copy.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #94a3b8;
                border: 1px solid #334155;
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #334155;
                color: #f8fafc;
            }
        """)
        btn_copy.clicked.connect(self._copy_console_logs)

        btn_clear = QPushButton("Clear")
        btn_clear.setToolTip("Clear console history")
        btn_clear.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #94a3b8;
                border: 1px solid #334155;
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #334155;
                color: #f8fafc;
            }
        """)
        btn_clear.clicked.connect(self._clear_console_logs)

        self.chk_autoscroll = QCheckBox("Auto-scroll")
        self.chk_autoscroll.setChecked(True)
        self.chk_autoscroll.setStyleSheet("color: #94a3b8; font-size: 11px;")

        btn_expand = QPushButton("⤢")
        btn_expand.setToolTip("Expand console height")
        btn_expand.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #94a3b8;
                font-size: 13px;
                font-weight: bold;
                border: none;
                padding: 0 4px;
            }
            QPushButton:hover {
                color: #38bdf8;
            }
        """)
        btn_expand.clicked.connect(self._toggle_console_expand)

        btn_close = QPushButton("✕")
        btn_close.setToolTip("Collapse console panel (drag splitter or click ▲ Console to restore)")
        btn_close.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #94a3b8;
                font-size: 13px;
                font-weight: bold;
                border: none;
                padding: 0 4px;
            }
            QPushButton:hover {
                color: #ef4444;
            }
        """)
        btn_close.clicked.connect(self.toggle_console_panel)

        h_head.addWidget(self.chk_autoscroll)
        h_head.addWidget(btn_copy)
        h_head.addWidget(btn_clear)
        h_head.addWidget(btn_expand)
        h_head.addWidget(btn_close)
        v.addLayout(h_head)

        # Log Text Area
        self.txt_console = QPlainTextEdit()
        self.txt_console.setReadOnly(True)
        self.txt_console.setMaximumBlockCount(2000)
        self.txt_console.setStyleSheet("""
            QPlainTextEdit {
                background-color: #060911;
                color: #e2e8f0;
                border: 1px solid #1e293b;
                border-radius: 4px;
                font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                padding: 4px 6px;
            }
        """)
        v.addWidget(self.txt_console)

        return panel

    def toggle_console_panel(self):
        """Toggle bottom activity console panel visibility."""
        if not hasattr(self, "console_panel"):
            return
        is_visible = self.console_panel.isVisible()
        self.console_panel.setVisible(not is_visible)
        self._unread_logs = 0
        if not is_visible:
            self.btn_toggle_console.setText("▼ Hide Console")
            self.btn_toggle_console.setStyleSheet("""
                QPushButton#btnToggleConsole {
                    background-color: #334155;
                    color: #f8fafc;
                    border: 1px solid #64748b;
                    border-radius: 4px;
                    padding: 2px 10px;
                    font-size: 11px;
                    font-weight: 700;
                }
            """)
            if hasattr(self, "main_vsplitter"):
                sizes = self.main_vsplitter.sizes()
                if len(sizes) >= 2 and sizes[1] < 80:
                    total_h = sum(sizes) or 800
                    self.main_vsplitter.setSizes([total_h - 160, 160])
        else:
            self.btn_toggle_console.setText("▲ Console")
            self.btn_toggle_console.setStyleSheet("""
                QPushButton#btnToggleConsole {
                    background-color: #1e293b;
                    color: #38bdf8;
                    border: 1px solid #334155;
                    border-radius: 4px;
                    padding: 2px 10px;
                    font-size: 11px;
                    font-weight: 700;
                }
                QPushButton#btnToggleConsole:hover {
                    background-color: #334155;
                    color: #f8fafc;
                    border-color: #0284c7;
                }
            """)

    def _toggle_console_expand(self):
        if not hasattr(self, "main_vsplitter"):
            return
        sizes = self.main_vsplitter.sizes()
        if len(sizes) >= 2:
            total_h = sum(sizes) or 800
            current_h = sizes[1]
            new_h = 320 if current_h < 220 else 140
            self.main_vsplitter.setSizes([total_h - new_h, new_h])

    def _copy_console_logs(self):
        if hasattr(self, "txt_console"):
            text = self.txt_console.toPlainText()
            QApplication.clipboard().setText(text)
            self.status_bar.showMessage("Copied console logs to clipboard.", 3000)

    def _clear_console_logs(self):
        if hasattr(self, "txt_console"):
            self.txt_console.clear()
            self.lbl_console_status.setText("Ready")
            self.lbl_console_status.setStyleSheet("background-color: #1e293b; color: #94a3b8; border-radius: 3px; padding: 2px 6px; font-size: 10px; font-weight: 600;")

    def log_console(self, category: str, message: str, level: str = "info", step: Optional[int] = None, total: Optional[int] = None):
        """Append a formatted entry to the Activity & Segmentation Console and update the bottom bar."""
        from datetime import datetime
        now_str = datetime.now().strftime("%H:%M:%S")

        color_map = {
            "QUERY": "#38bdf8",     # sky blue
            "STAND": "#06b6d4",     # cyan
            "AI": "#a855f7",        # purple
            "GPU": "#f59e0b",       # amber
            "EXPORT": "#10b981",    # emerald
            "OCTREE": "#0284c7",    # blue
            "DB": "#6366f1",        # indigo
            "INDEX": "#3b82f6",     # royal blue
            "COMPLETE": "#10b981",  # emerald
            "WARN": "#fbbf24",      # amber
            "ERROR": "#ef4444",     # red
            "SUCCESS": "#10b981",   # emerald
            "SYSTEM": "#94a3b8",    # slate
            "INFO": "#94a3b8"       # slate
        }
        badge_color = color_map.get(category.upper(), "#94a3b8")
        step_str = f"[{step}/{total}] " if (step and total) else ""
        line = f"[{now_str}] [{category.upper()}] {step_str}{message}"

        if hasattr(self, "txt_console"):
            self.txt_console.appendPlainText(line)
            if hasattr(self, "chk_autoscroll") and self.chk_autoscroll.isChecked():
                self.txt_console.verticalScrollBar().setValue(
                    self.txt_console.verticalScrollBar().maximum()
                )

        if hasattr(self, "lbl_console_status"):
            short_msg = message if len(message) <= 60 else f"{message[:57]}..."
            self.lbl_console_status.setText(f"{category.upper()}: {step_str}{short_msg}")
            self.lbl_console_status.setStyleSheet(f"""
                background-color: rgba(30, 41, 59, 0.9);
                color: {badge_color};
                border: 1px solid {badge_color};
                border-radius: 3px;
                padding: 2px 6px;
                font-size: 10px;
                font-weight: 600;
            """)

        # Update bottom status bar live step message
        if hasattr(self, "lbl_status_live_msg"):
            self.lbl_status_live_msg.setText(f"<b><span style='color:{badge_color}'>[{category.upper()}]</span></b> {step_str}{message}")
        elif hasattr(self, "status_bar"):
            self.status_bar.showMessage(f"[{category.upper()}] {step_str}{message}", 8000)

        # Notify unread count on toggle button if console panel is closed
        if hasattr(self, "console_panel") and not self.console_panel.isVisible():
            if hasattr(self, "btn_toggle_console"):
                self._unread_logs = getattr(self, "_unread_logs", 0) + 1
                self.btn_toggle_console.setText(f"▲ Console ({self._unread_logs})")
                self.btn_toggle_console.setStyleSheet("""
                    QPushButton#btnToggleConsole {
                        background-color: #0284c7;
                        color: white;
                        border: 1px solid #38bdf8;
                        border-radius: 4px;
                        padding: 2px 10px;
                        font-size: 11px;
                        font-weight: 700;
                    }
                    QPushButton#btnToggleConsole:hover {
                        background-color: #0369a1;
                    }
                """)

    def _on_segmentation_step(self, data: dict):
        step = data.get("step")
        total = data.get("total")
        cat = data.get("category", "AI")
        msg = data.get("msg", "")
        stage = data.get("stage", "")
        level = "error" if stage == "error" else ("success" if stage in ("inference_done", "complete", "export_done") else "info")
        self.log_console(cat, msg, level=level, step=step, total=total)

    def _on_console_log_event(self, data: dict):
        cat = data.get("category", "INFO")
        msg = data.get("msg", "")
        level = data.get("level", "info")
        self.log_console(cat, msg, level=level)

    # -------------------------------------------------------------------------
    # Background Indexing UI Notification & Status Indicator
    # -------------------------------------------------------------------------
    def _init_indexing_pill(self):
        """Create background indexing status badge in the status bar (only visible when indexing)."""
        self.indexing_pill = QFrame()
        self.indexing_pill.setObjectName("indexingPill")
        self.indexing_pill.setCursor(Qt.CursorShape.PointingHandCursor)
        pill_layout = QHBoxLayout(self.indexing_pill)
        pill_layout.setContentsMargins(10, 2, 10, 2)
        pill_layout.setSpacing(6)

        self.lbl_indexing_icon = QLabel("⚡")
        self.lbl_indexing_icon.setStyleSheet("font-size: 13px; background: transparent;")

        self.lbl_indexing_text = QLabel("Spatial Index: Ready")
        self.lbl_indexing_text.setStyleSheet("font-size: 11px; font-weight: 600; background: transparent;")

        self.indexing_progress_bar = QProgressBar()
        self.indexing_progress_bar.setRange(0, 100)
        self.indexing_progress_bar.setValue(0)
        self.indexing_progress_bar.setFixedSize(70, 10)
        self.indexing_progress_bar.setTextVisible(False)
        self.indexing_progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #38bdf8;
                border-radius: 3px;
            }
        """)
        self.indexing_progress_bar.setVisible(False)

        pill_layout.addWidget(self.lbl_indexing_icon)
        pill_layout.addWidget(self.lbl_indexing_text)
        pill_layout.addWidget(self.indexing_progress_bar)

        self.indexing_pill.mousePressEvent = lambda e: self.show_indexing_details_dialog()
        self._set_indexing_pill_tooltip()
        self.status_bar.addPermanentWidget(self.indexing_pill)
        self._update_indexing_pill_style("idle", "Spatial Index: Ready")
        # Hidden by default - only shown when actively indexing!
        self.indexing_pill.setVisible(False)

    def _set_indexing_pill_tooltip(self, extra_info: str = ""):
        tip = (
            "<b>Automatic Background Indexing:</b><br>"
            "As soon as any LAS/LAZ file is opened or converted, spatial chunk bounding boxes "
            "and tree index sidecars are automatically built in a background thread without freezing the UI.<br><br>"
            "<b>Zero-Dependency Streaming Fallback:</b><br>"
            "Even on a brand-new file before the background index finishes, get_tree_points streams "
            "chunks directly using the target tree's spatial centroid. Neighbor queries always bring "
            "in surrounding trees immediately on any file."
        )
        if extra_info:
            tip += f"<br><br><b>Status:</b> {extra_info}"
        tip += "<br><br><i>(Click to inspect indexing engine details)</i>"
        self.indexing_pill.setToolTip(tip)

    def _update_indexing_pill_style(self, state: str, text: str, percent: int = 0):
        self.lbl_indexing_text.setText(text)
        if state == "indexing":
            self.lbl_indexing_icon.setText("⚡" if percent <= 50 else "🌲")
            self.indexing_pill.setStyleSheet("""
                QFrame#indexingPill {
                    background-color: rgba(2, 132, 199, 0.20);
                    border: 1px solid #0284c7;
                    border-radius: 6px;
                }
            """)
            self.lbl_indexing_text.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: 600; background: transparent;")
            self.indexing_progress_bar.setValue(percent)
            self.indexing_progress_bar.setVisible(True)
        elif state == "ready":
            self.lbl_indexing_icon.setText("✓")
            self.indexing_pill.setStyleSheet("""
                QFrame#indexingPill {
                    background-color: rgba(16, 185, 129, 0.18);
                    border: 1px solid #059669;
                    border-radius: 6px;
                }
            """)
            self.lbl_indexing_text.setStyleSheet("color: #34d399; font-size: 11px; font-weight: 600; background: transparent;")
            self.indexing_progress_bar.setVisible(False)
        elif state == "error":
            self.lbl_indexing_icon.setText("⚠")
            self.indexing_pill.setStyleSheet("""
                QFrame#indexingPill {
                    background-color: rgba(239, 68, 68, 0.18);
                    border: 1px solid #dc2626;
                    border-radius: 6px;
                }
            """)
            self.lbl_indexing_text.setStyleSheet("color: #f87171; font-size: 11px; font-weight: 600; background: transparent;")
            self.indexing_progress_bar.setVisible(False)
        else:  # idle
            self.lbl_indexing_icon.setText("⚡")
            self.indexing_pill.setStyleSheet("""
                QFrame#indexingPill {
                    background-color: #1e293b;
                    border: 1px solid #334155;
                    border-radius: 6px;
                }
            """)
            self.lbl_indexing_text.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 600; background: transparent;")
            self.indexing_progress_bar.setVisible(False)

    def on_indexing_event(self, data: dict):
        """Handle background indexing lifecycle events dispatched from PointCloudManager."""
        if not hasattr(self, "indexing_pill"):
            return
        ev_type = data.get("type")
        stage = data.get("stage")
        msg = data.get("msg", "")

        # Forward status to Potree 3D WebGL HUD
        try:
            self._run_js(f"window.updateIndexingStatus({json.dumps(data)});")
        except Exception:
            pass

        if ev_type == "started":
            self.indexing_pill.setVisible(True)
            if stage == "chunks":
                self._update_indexing_pill_style("indexing", "⚡ Indexing Spatial Layout (0%)...", 0)
                self.log_console("INDEX", "Automatic Background Indexing started: scanning spatial chunks (streaming fallback active)...", level="info")
            elif stage == "trees":
                self._update_indexing_pill_style("indexing", "🌲 Indexing Tree Layout...", 0)
                self.log_console("INDEX", "Automatic Background Indexing: scanning tree boundaries...", level="info")
            self._set_indexing_pill_tooltip(msg)

        elif ev_type == "progress":
            self.indexing_pill.setVisible(True)
            pct = data.get("percent", 0)
            curr = data.get("current", 0)
            tot = data.get("total", 1)
            if stage == "chunks":
                self._update_indexing_pill_style("indexing", f"⚡ Chunks {curr}/{tot} ({pct}%)", pct)
            elif stage == "trees":
                trees_cnt = data.get("trees_found", 0)
                self._update_indexing_pill_style("indexing", f"🌲 Trees in Chunk {curr}/{tot} ({trees_cnt:,} trees)", pct)
            self._set_indexing_pill_tooltip(msg)

        elif ev_type in ("completed", "ready"):
            trees = data.get("trees", 0)
            chunks = data.get("chunks", 0)
            ready_text = (
                f"✓ Tree Spatial Index Ready ({trees:,} trees)"
                if trees > 0 else f"✓ Spatial Index Ready ({chunks} chunks)"
            )
            self._update_indexing_pill_style("ready", ready_text)
            self._set_indexing_pill_tooltip(msg)
            if ev_type == "completed":
                self.indexing_pill.setVisible(True)
                self.log_console("INDEX", f"✓ Background indexing complete — {trees:,} trees indexed for instant queries.", level="success")
                # Auto-hide the pill after 5 seconds so it does not permanently clutter the bottom bar!
                QTimer.singleShot(5000, self._hide_indexing_pill_if_idle)
            elif ev_type == "ready":
                # Precomputed sidecar was loaded at start, stay hidden unless indexing is happening
                self.indexing_pill.setVisible(False)

        elif ev_type == "error":
            self.indexing_pill.setVisible(True)
            err = data.get("error", "Unknown")
            self._update_indexing_pill_style("error", "⚠ Streaming Fallback Active")
            self._set_indexing_pill_tooltip(f"Indexing Warning: {err}\nStand queries continue using zero-dependency streaming fallback.")
            self.log_console("INDEX", f"Indexing Warning: {err} — streaming fallback active.", level="warn")

    def _hide_indexing_pill_if_idle(self):
        """Hide the indexing indicator badge when no indexing is in progress."""
        if not hasattr(self, "indexing_pill"):
            return
        status = self.manager.get_indexing_status() if hasattr(self, "manager") and self.manager else {}
        if not status.get("is_indexing", False):
            self.indexing_pill.setVisible(False)

    def show_indexing_details_dialog(self):
        """Show information dialog explaining Automatic Background Indexing and streaming fallback."""
        status = self.manager.get_indexing_status()
        active_las = self.manager.las_path or (self.current_las_path or "No file loaded")
        is_indexing = status.get("is_indexing", False)
        chunk_cnt = status.get("chunk_count", 0)
        tree_cnt = status.get("tree_count", 0)

        status_str = (
            "<span style='color:#38bdf8; font-weight:bold;'>Active (Building in Background)</span>"
            if is_indexing else
            "<span style='color:#34d399; font-weight:bold;'>Ready (Precomputed Sidecar Available)</span>"
            if (chunk_cnt > 0 or tree_cnt > 0) else
            "<span style='color:#94a3b8;'>Idle</span>"
        )

        body = (
            f"<h3>Automatic Background Indexing & Stand Acceleration</h3>"
            f"<p><b>Active Cloud:</b> {os.path.basename(active_las)}</p>"
            f"<p><b>Indexing Engine Status:</b> {status_str}</p>"
            f"<p><b>Indexed Spatial Chunks:</b> {chunk_cnt}</p>"
            f"<p><b>Indexed Individual Trees:</b> {tree_cnt:,}</p>"
            f"<hr>"
            f"<p><b>Engine Architecture:</b></p>"
            f"<ul>"
            f"<li><b>Automatic Background Indexing:</b> As soon as any LAS/LAZ file is opened or converted, "
            f"<code>_build_chunk_index_in_background</code> automatically builds spatial chunk and tree index "
            f"sidecars in a background thread without freezing the UI.</li>"
            f"<li><b>Zero-Dependency Streaming Fallback:</b> Even on a brand-new file before the background "
            f"index finishes, <code>get_tree_points</code> streams the chunks directly using the target tree's "
            f"spatial centroid (t_cx, t_cy). Neighbor queries will always bring in the surrounding trees immediately "
            f"on any file.</li>"
            f"</ul>"
        )
        QMessageBox.information(self, "Indexing Engine Architecture & Status", body)

    # -------------------------------------------------------------------------
    # Application Menu Bar (Fallback)
    # -------------------------------------------------------------------------
    def init_menu_bar(self):
        menubar = self.menuBar()

        # 1. File Menu
        menu_file = menubar.addMenu("&File")

        act_open_potree = QAction("Open Potree Converted Dataset...", self)
        act_open_potree.setShortcut("Ctrl+O")
        act_open_potree.triggered.connect(self.open_potree_file)
        menu_file.addAction(act_open_potree)

        act_open_las = QAction("Open LAS / LAZ Point Cloud...", self)
        act_open_las.setShortcut("Ctrl+Shift+O")
        act_open_las.triggered.connect(self.open_las_file)
        menu_file.addAction(act_open_las)

        act_convert_las = QAction("Convert LAS / LAZ to Potree...", self)
        act_convert_las.setShortcut("Ctrl+Alt+C")
        act_convert_las.triggered.connect(self.convert_las_action)
        menu_file.addAction(act_convert_las)

        act_open_folder = QAction("Open Point Cloud Folder...", self)
        act_open_folder.triggered.connect(self.open_folder_dialog)
        menu_file.addAction(act_open_folder)

        menu_file.addSeparator()

        act_commit = QAction("Commit & Save to LAS/LAZ...", self)
        act_commit.setShortcut("Ctrl+Shift+S")
        act_commit.triggered.connect(self.commit_staged_edits_action)
        menu_file.addAction(act_commit)

        act_save = QAction("Save Modified LAS (In-Memory)...", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self.save_modified_las)
        menu_file.addAction(act_save)

        menu_file.addSeparator()

        act_exit = QAction("Exit", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        menu_file.addAction(act_exit)

        # 2. Edit Menu
        menu_edit = menubar.addMenu("&Edit")

        act_history = QAction("Edit History & Undo / Revert...", self)
        act_history.setShortcut("Ctrl+H")
        act_history.triggered.connect(self.open_edit_history_dialog)
        menu_edit.addAction(act_history)

        menu_edit.addSeparator()

        act_clear_staged = QAction("Clear All Staged Edits", self)
        act_clear_staged.triggered.connect(self.clear_all_staged_edits)
        menu_edit.addAction(act_clear_staged)

    # -------------------------------------------------------------------------
    # Sidebar Construction
    # -------------------------------------------------------------------------
    def create_sidebar(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setObjectName("sidebarTabs")

        tab_layers = self.create_layer_manager_tab()
        tab_edit = self.create_delineation_tab()
        tab_dataset = self.create_dataset_tab()

        self.sidebar_tabs.addTab(tab_layers, "Layers")
        self.sidebar_tabs.addTab(tab_edit, "Forestry")
        self.sidebar_tabs.addTab(tab_dataset, "Dataset")
        self.sidebar_tabs.currentChanged.connect(self._on_sidebar_tab_changed)

        layout.addWidget(self.sidebar_tabs)
        return container

    def _on_sidebar_tab_changed(self, index: int):
        # Ribbon "Filter" and "AI" menus are only visible while the Forestry tab is open
        is_forestry = (index == 1)
        if hasattr(self, "btn_tab_filter"):
            self.btn_tab_filter.setVisible(is_forestry)
        if hasattr(self, "btn_tab_ai"):
            self.btn_tab_ai.setVisible(is_forestry)
        # Keep the ribbon page in sync if currently on Filter or AI tab and leaving Forestry
        if not is_forestry and self.ribbon_stack.currentIndex() in (4, 5):
            self.ribbon_stack.setCurrentIndex(2)  # Layers page
            self.btn_tab_layers.setChecked(True)

        if index == 0:  # Returned to Layers tab
            # Restore the full point cloud context on the main layer and hide individual tree overlay
            self._run_js("if (window.showMainCloudContext) window.showMainCloudContext();")
            self._disable_tree_filter_mode()
        elif index == 1:  # Switched to Forestry tab
            # Re-show the individual tree overlay and hide the main cloud context & inspect highlights
            self._run_js("if (window.showTreeOverlayContext) window.showTreeOverlayContext(); else { if (document.getElementById('inspectCard')) document.getElementById('inspectCard').style.display = 'none'; }")
        else:
            self._disable_tree_filter_mode()

    def _set_stand_filter_buttons_enabled(self, enabled: bool):
        """Enable/disable the ribbon Stand Filter buttons (not just show/hide them)."""
        for name in ("btn_tree_filter_mode", "btn_top_filter", "btn_top_reset"):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.setEnabled(enabled)

    def _disable_tree_filter_mode(self):
        if hasattr(self, "btn_tree_filter_mode") and self.btn_tree_filter_mode.isChecked():
            self.btn_tree_filter_mode.blockSignals(True)
            self.btn_tree_filter_mode.setChecked(False)
            self.btn_tree_filter_mode.setText("Filter Mode")
            self.btn_tree_filter_mode.blockSignals(False)
        self._run_js("if (window.setTreeFilterMode) window.setTreeFilterMode(false);")

    def _layer_size_slider_row(self) -> QHBoxLayout:
        """Slider + live value badge for the Point Size quick-styling row."""
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self.slider_layer_size, 1)
        row.addWidget(self.lbl_layer_size_val, 0, Qt.AlignmentFlag.AlignVCenter)
        return row

    def create_layer_manager_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        v_main = QVBoxLayout(content)
        v_main.setSpacing(28)
        v_main.setContentsMargins(10, 10, 10, 10)

        # 1. Global Toolbar Header
        h_header = QHBoxLayout()
        lbl_title = QLabel("Layer Manager")
        lbl_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #f8fafc;")
        h_header.addWidget(lbl_title)
        h_header.addStretch()

        btn_show_all = QPushButton("Show All")
        btn_show_all.setObjectName("btnSecondary")
        btn_show_all.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        btn_show_all.clicked.connect(self.show_all_layers)
        h_header.addWidget(btn_show_all)

        btn_hide_all = QPushButton("Hide All")
        btn_hide_all.setObjectName("btnSecondary")
        btn_hide_all.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        btn_hide_all.clicked.connect(self.hide_all_layers)
        h_header.addWidget(btn_hide_all)
        v_main.addLayout(h_header)

        # 2. Point Cloud Layer Card
        grp_pc = QGroupBox("POINT CLOUD LAYER")
        v_pc = QVBoxLayout(grp_pc)
        v_pc.setSpacing(8)

        # Header row: master visibility toggle
        h_pc_title = QHBoxLayout()
        self.chk_layer_pc = QCheckBox("Visible")
        self.chk_layer_pc.setChecked(True)
        self.chk_layer_pc.setStyleSheet("font-weight: 600; color: #e2e8f0; font-size: 12px;")
        self.chk_layer_pc.toggled.connect(self.toggle_pointcloud_layer)
        h_pc_title.addWidget(self.chk_layer_pc)
        h_pc_title.addStretch()
        v_pc.addLayout(h_pc_title)

        # Active cloud identity: name (prominent) + point count (monospace badge line)
        self.lbl_layer_cloud_name = QLabel("No point cloud loaded")
        self.lbl_layer_cloud_name.setStyleSheet(
            "color: #f1f5f9; font-size: 12px; font-weight: 600; padding: 2px 4px;")
        self.lbl_layer_cloud_name.setWordWrap(True)
        v_pc.addWidget(self.lbl_layer_cloud_name)

        self.lbl_layer_cloud_pts = QLabel("- points")
        self.lbl_layer_cloud_pts.setStyleSheet(
            "color: #38bdf8; font-size: 11px; font-family: monospace; padding: 0 4px 2px 4px;")
        v_pc.addWidget(self.lbl_layer_cloud_pts)

        # Layer list.  The clouds themselves are the layers (potree's scene list is the
        # manager), so every loaded cloud gets a row: the checkbox is pc.visible, and the
        # selected row is the layer the colour, size and edit controls drive.
        self.layer_list = QListWidget()
        self.layer_list.setMaximumHeight(120)
        self.layer_list.setStyleSheet("""
            QListWidget {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 3px;
                outline: none;
            }
            QListWidget::item {
                padding: 5px 7px;
                margin: 1px 1px;
                border-radius: 5px;
                color: #cbd5e1;
                font-size: 12px;
            }
            QListWidget::item:hover {
                background-color: #1e293b;
            }
            QListWidget::item:selected {
                background-color: #1d4ed8;
                color: #ffffff;
            }
        """)
        self.layer_list.itemSelectionChanged.connect(self._on_layer_row_selected)
        self.layer_list.itemChanged.connect(self._on_layer_row_checked)
        v_pc.addWidget(self.layer_list)

        h_layer_buttons = QHBoxLayout()
        h_layer_buttons.setSpacing(6)
        btn_remove_layer = QPushButton("Remove Layer")
        btn_remove_layer.setObjectName("btnSecondary")
        btn_remove_layer.setStyleSheet("padding: 4px 10px; font-size: 11px;")
        btn_remove_layer.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_remove_layer.clicked.connect(self._on_layer_remove_clicked)
        btn_refresh_layers = QPushButton("Refresh")
        btn_refresh_layers.setObjectName("btnSecondary")
        btn_refresh_layers.setStyleSheet("padding: 4px 10px; font-size: 11px;")
        btn_refresh_layers.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_refresh_layers.clicked.connect(lambda: self.refresh_layers(0))
        h_layer_buttons.addWidget(btn_remove_layer)
        h_layer_buttons.addWidget(btn_refresh_layers)
        h_layer_buttons.addStretch()
        v_pc.addLayout(h_layer_buttons)

        # Divider into the per-layer styling controls
        h_quick = QHBoxLayout()
        h_quick.setSpacing(8)
        lbl_quick = QLabel("QUICK STYLING")
        lbl_quick.setStyleSheet("color: #64748b; font-size: 9px; font-weight: 700; letter-spacing: 1px;")
        line_quick = QFrame()
        line_quick.setFrameShape(QFrame.Shape.HLine)
        line_quick.setStyleSheet("color: #1e293b; background-color: #1e293b; max-height: 1px; border: none;")
        h_quick.addWidget(lbl_quick)
        h_quick.addWidget(line_quick, 1)
        v_pc.addSpacing(2)
        v_pc.addLayout(h_quick)

        # Guards the row rebuild: re-adding rows fires selection and check signals, and
        # those must not bounce back into the viewer as user edits.
        self._layer_sync_guard = False

        # Quick Controls inside Point Cloud card
        g_quick = QGridLayout()
        g_quick.setHorizontalSpacing(10)
        g_quick.setVerticalSpacing(6)
        g_quick.setColumnMinimumWidth(0, 72)
        g_quick.setColumnStretch(1, 1)

        lbl_c = QLabel("Color:")
        lbl_c.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.combo_layer_color = QComboBox()
        self.combo_layer_color.setCursor(Qt.CursorShape.PointingHandCursor)
        self.combo_layer_color.setStyleSheet("font-size: 11px; padding: 3px 6px;")
        self.combo_layer_color.addItem("RGBA (True Color)", "rgba")
        self.combo_layer_color.addItem("ASPRS Classification", "classification")
        self.combo_layer_color.addItem("Intensity", "intensity")
        self.combo_layer_color.addItem("Elevation", "elevation")
        self.combo_layer_color.currentIndexChanged.connect(self._on_layer_color_changed)
        g_quick.addWidget(lbl_c, 0, 0)
        g_quick.addWidget(self.combo_layer_color, 0, 1)

        lbl_s = QLabel("Point Size:")
        lbl_s.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.slider_layer_size = QSlider(Qt.Orientation.Horizontal)
        self.slider_layer_size.setRange(1, 40)
        self.slider_layer_size.setValue(10)
        self.slider_layer_size.setCursor(Qt.CursorShape.PointingHandCursor)
        self.lbl_layer_size_val = QLabel("1.00")
        self.lbl_layer_size_val.setStyleSheet(
            "color: #38bdf8; font-size: 11px; font-family: monospace;"
            "background-color: #0f172a; border: 1px solid #334155;"
            "border-radius: 4px; padding: 1px 5px; min-width: 30px;")
        self.lbl_layer_size_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider_layer_size.valueChanged.connect(self._on_layer_size_slider_changed)
        g_quick.addWidget(lbl_s, 1, 0)
        g_quick.addLayout(self._layer_size_slider_row(), 1, 1)
        v_pc.addLayout(g_quick)

        v_main.addWidget(grp_pc)

        v_main.addStretch()
        scroll.setWidget(content)
        return scroll

    def create_delineation_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(28)
        layout.setContentsMargins(12, 12, 12, 12)

        # 1. Full-Density Tree View — queries every point of one ITC id from the
        # LAS master. Opening it isolates the tree and hides the octree context;
        # returning to Layers restores the full point cloud context.
        grp_tree = QGroupBox("VIEW TREE (FULL DENSITY)")
        v_tree = QVBoxLayout(grp_tree)
        v_tree.setSpacing(8)

        self.lbl_tree_view_status = QLabel("Ctrl+Click a tree point or enter Tree ID to view in full density.")
        self.lbl_tree_view_status.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.lbl_tree_view_status.setWordWrap(True)
        v_tree.addWidget(self.lbl_tree_view_status)

        h_tree = QHBoxLayout()
        lbl_tvid = QLabel("Tree ID:")
        lbl_tvid.setStyleSheet("color: #94a3b8; font-size: 11px;")
        h_tree.addWidget(lbl_tvid)
        self.spin_tree_view = QSpinBox()
        self.spin_tree_view.setRange(1, 999999)
        self.spin_tree_view.setValue(self.spin_ai_tree.value() if hasattr(self, "spin_ai_tree") else 1)
        self.spin_tree_view.setStyleSheet("font-size: 11px; padding: 3px 6px;")
        self.spin_tree_view.setCursor(Qt.CursorShape.PointingHandCursor)
        self.spin_tree_view.valueChanged.connect(self._on_tree_spin_changed)
        h_tree.addWidget(self.spin_tree_view)
        h_tree.addStretch()
        v_tree.addLayout(h_tree)

        # Tree-specific Color By Option
        h_tree_color = QHBoxLayout()
        lbl_tree_color = QLabel("Color by:")
        lbl_tree_color.setStyleSheet("color: #94a3b8; font-size: 11px;")
        h_tree_color.addWidget(lbl_tree_color)
        self.combo_tree_color = QComboBox()
        self.combo_tree_color.setStyleSheet("font-size: 11px; padding: 3px 6px;")
        self.combo_tree_color.setCursor(Qt.CursorShape.PointingHandCursor)
        self.combo_tree_color.addItem("Match Main Layer (Auto)", "auto")
        self.combo_tree_color.addItem("RGB / True Color", "rgba")
        self.combo_tree_color.addItem("ASPRS Classification", "classification")
        self.combo_tree_color.addItem("Elevation Gradient", "elevation")
        self.combo_tree_color.addItem("Intensity", "intensity")
        self.combo_tree_color.addItem("Tree / ITC (Segmented Stand)", "itc")
        self.combo_tree_color.addItem("Natural Foliage Green", "green")
        self.combo_tree_color.addItem("Viridis Heatmap", "viridis")
        self.combo_tree_color.currentIndexChanged.connect(self._on_tree_color_changed)
        h_tree_color.addWidget(self.combo_tree_color)
        v_tree.addLayout(h_tree_color)

        # Tree-specific Point Size Option
        h_tree_size = QHBoxLayout()
        lbl_tsize = QLabel("Point Size:")
        lbl_tsize.setStyleSheet("color: #94a3b8; font-size: 11px;")
        h_tree_size.addWidget(lbl_tsize)
        self.slider_tree_size = QSlider(Qt.Orientation.Horizontal)
        self.slider_tree_size.setRange(1, 40)
        self.slider_tree_size.setValue(10)
        self.slider_tree_size.setCursor(Qt.CursorShape.PointingHandCursor)
        self.lbl_tree_size_val = QLabel("1.00")
        self.lbl_tree_size_val.setStyleSheet(
            "color: #38bdf8; font-size: 11px; font-family: monospace;"
            "background-color: #0f172a; border: 1px solid #334155;"
            "border-radius: 4px; padding: 1px 5px; min-width: 30px;")
        self.lbl_tree_size_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider_tree_size.valueChanged.connect(self._on_tree_size_changed)
        h_tree_size.addWidget(self.slider_tree_size)
        h_tree_size.addWidget(self.lbl_tree_size_val)
        v_tree.addLayout(h_tree_size)

        # Tree Neighbor Options
        h_neighbors = QHBoxLayout()
        self.chk_tree_neighbors = QCheckBox("Include Neighbors")
        self.chk_tree_neighbors.setStyleSheet("color: #e2e8f0; font-size: 11px; font-weight: 600;")
        self.chk_tree_neighbors.setCursor(Qt.CursorShape.PointingHandCursor)
        if hasattr(self, "chk_ai_neighbors"):
            self.chk_tree_neighbors.setChecked(self.chk_ai_neighbors.isChecked())
        self.chk_tree_neighbors.toggled.connect(self._on_tree_neighbors_toggled)
        h_neighbors.addWidget(self.chk_tree_neighbors)

        lbl_radius = QLabel("Radius:")
        lbl_radius.setStyleSheet("color: #94a3b8; font-size: 11px;")
        h_neighbors.addWidget(lbl_radius)

        self.spin_tree_radius = QDoubleSpinBox()
        self.spin_tree_radius.setRange(2.0, 100.0)
        self.spin_tree_radius.setValue(self.spin_ai_radius.value() if hasattr(self, "spin_ai_radius") else 5.0)
        self.spin_tree_radius.setSingleStep(5.0)
        self.spin_tree_radius.setSuffix(" m")
        self.spin_tree_radius.setStyleSheet("font-size: 11px; padding: 2px 4px; max-width: 65px;")
        self.spin_tree_radius.setEnabled(self.chk_tree_neighbors.isChecked())
        self.spin_tree_radius.valueChanged.connect(self._on_tree_radius_changed)
        h_neighbors.addWidget(self.spin_tree_radius)
        h_neighbors.addStretch()
        v_tree.addLayout(h_neighbors)

        h_nmode = QHBoxLayout()
        lbl_nmode = QLabel("Stand Mode:")
        lbl_nmode.setStyleSheet("color: #94a3b8; font-size: 11px;")
        h_nmode.addWidget(lbl_nmode)

        self.combo_neighbor_mode = QComboBox()
        self.combo_neighbor_mode.setStyleSheet("font-size: 11px; padding: 3px 6px;")
        # First entry is the default stand mode: the whole stand cylinder under
        # Include Neighbors (canopy + ground) gives ForestFormer3D full context.
        self.combo_neighbor_mode.addItem("All Stand Points (Canopy + Ground) (Recommended)", "all")
        self.combo_neighbor_mode.addItem("Neighbor Trees & Canopy", "canopy")
        self.combo_neighbor_mode.addItem("Segmented Trees Only (ITC > 0)", "trees_only")
        self.combo_neighbor_mode.setEnabled(False)
        h_nmode.addWidget(self.combo_neighbor_mode)
        v_tree.addLayout(h_nmode)

        # Ground under the tree: ASPRS class 2 points within a radius of the tree
        h_ground = QHBoxLayout()
        self.chk_tree_ground = QCheckBox("Show Ground")
        self.chk_tree_ground.setChecked(False)
        self.chk_tree_ground.setStyleSheet("color: #e2e8f0; font-size: 11px; font-weight: 600;")
        self.chk_tree_ground.setCursor(Qt.CursorShape.PointingHandCursor)
        h_ground.addWidget(self.chk_tree_ground)

        lbl_grad = QLabel("Radius:")
        lbl_grad.setStyleSheet("color: #94a3b8; font-size: 11px;")
        h_ground.addWidget(lbl_grad)

        self.spin_tree_ground = QDoubleSpinBox()
        self.spin_tree_ground.setRange(2.0, 100.0)
        self.spin_tree_ground.setValue(10.0)
        self.spin_tree_ground.setSingleStep(5.0)
        self.spin_tree_ground.setSuffix(" m")
        self.spin_tree_ground.setStyleSheet("font-size: 11px; padding: 2px 4px; max-width: 65px;")
        self.spin_tree_ground.setEnabled(False)  # matches the unchecked default
        h_ground.addWidget(self.spin_tree_ground)
        h_ground.addStretch()
        v_tree.addLayout(h_ground)
        self.chk_tree_ground.toggled.connect(
            lambda on: (self.spin_tree_ground.setEnabled(on), self.view_tree_action()))
        self.spin_tree_ground.editingFinished.connect(self.view_tree_action)

        # Bounding box (visible by default, with tree height label). Measure
        # lives on the ribbon Point tab (chk_ribbon_measure).
        h_tools = QHBoxLayout()
        self.chk_tree_bbox = QCheckBox("Bounding Box")
        self.chk_tree_bbox.setChecked(True)
        self.chk_tree_bbox.setStyleSheet("color: #e2e8f0; font-size: 11px; font-weight: 600;")
        self.chk_tree_bbox.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_tree_bbox.toggled.connect(
            lambda on: self._run_js(f"if (window.setTreeBBox) window.setTreeBBox({'true' if on else 'false'});"))
        h_tools.addWidget(self.chk_tree_bbox)
        h_tools.addStretch()
        v_tree.addLayout(h_tools)

        btn_view_tree = QPushButton("View Tree (All Points)")
        btn_view_tree.setObjectName("btnSuccess")
        btn_view_tree.setStyleSheet("padding: 5px 10px; font-size: 11px; font-weight: 600;")
        btn_view_tree.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_view_tree.clicked.connect(self.view_tree_action)
        v_tree.addWidget(btn_view_tree)

        # Export the currently displayed full-density tree/stand back to LAS/LAZ
        # (extracted straight from the master file, honoring the stand filter)
        self.btn_extract_tree = QPushButton("Extract View as COPC / LAS / LAZ...")
        self.btn_extract_tree.setObjectName("btnSecondary")
        self.btn_extract_tree.setStyleSheet("padding: 5px 10px; font-size: 11px;")
        self.btn_extract_tree.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_extract_tree.setEnabled(False)
        self.btn_extract_tree.clicked.connect(self.extract_tree_view_action)
        v_tree.addWidget(self.btn_extract_tree)

        btn_close_tree = QPushButton("Close Tree View")
        btn_close_tree.setObjectName("btnSecondary")
        btn_close_tree.setStyleSheet("padding: 5px 10px; font-size: 11px;")
        btn_close_tree.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_close_tree.clicked.connect(self.close_tree_view_action)
        v_tree.addWidget(btn_close_tree)

        v_tree.addSpacing(14)

        # Stand Trees Filter Group (Only shown in Forestry tab when viewing a tree with neighbors)
        self.grp_stand_filter = QGroupBox("FILTER NEIGHBOR TREES")
        self.grp_stand_filter.setStyleSheet("""
            QGroupBox {
                border: 1px solid #0284c7;
                border-radius: 8px;
                margin-top: 26px;
                padding-top: 34px;
                font-weight: 700;
                color: #38bdf8;
                font-size: 11px;
                background-color: rgba(2, 132, 199, 0.04);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 14px;
                top: 6px;
                padding: 1px 8px;
                background-color: #141c28;
                border: 1px solid #0284c7;
                border-radius: 4px;
                color: #38bdf8;
                font-weight: 700;
            }
        """)
        v_filter = QVBoxLayout(self.grp_stand_filter)
        v_filter.setSpacing(6)
        v_filter.setContentsMargins(8, 12, 8, 8)

        self.lbl_stand_filter_info = QLabel("Filter Mode ON: click trees in 3D, or use checkboxes:")
        self.lbl_stand_filter_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.lbl_stand_filter_info.setWordWrap(True)
        v_filter.addWidget(self.lbl_stand_filter_info)

        # Quick Select buttons
        h_fquick = QHBoxLayout()
        btn_sel_all = QPushButton("Select All")
        btn_sel_all.setStyleSheet("font-size: 10px; padding: 2px 6px;")
        btn_sel_all.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_sel_all.clicked.connect(self._on_filter_select_all)
        h_fquick.addWidget(btn_sel_all)

        btn_sel_target = QPushButton("Target Tree Only")
        btn_sel_target.setStyleSheet("font-size: 10px; padding: 2px 6px;")
        btn_sel_target.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_sel_target.clicked.connect(self._on_filter_select_target_only)
        h_fquick.addWidget(btn_sel_target)
        v_filter.addLayout(h_fquick)

        # Multi-select tree list with checkboxes
        self.list_stand_trees = QListWidget()
        self.list_stand_trees.setMaximumHeight(130)
        self.list_stand_trees.setStyleSheet("""
            QListWidget {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 4px;
                color: #e2e8f0;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 2px 4px;
            }
            QListWidget::item:hover {
                background-color: #1e293b;
            }
        """)
        self.list_stand_trees.itemChanged.connect(self._on_stand_tree_item_changed)
        v_filter.addWidget(self.list_stand_trees)

        # Filter and Reset buttons
        h_factions = QHBoxLayout()
        self.btn_apply_tree_filter = QPushButton("Filter")
        self.btn_apply_tree_filter.setObjectName("btnSuccess")
        self.btn_apply_tree_filter.setStyleSheet("padding: 4px 10px; font-size: 11px; font-weight: 700;")
        self.btn_apply_tree_filter.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_apply_tree_filter.clicked.connect(self._apply_tree_filter)
        h_factions.addWidget(self.btn_apply_tree_filter)

        self.btn_reset_tree_filter = QPushButton("Reset")
        self.btn_reset_tree_filter.setObjectName("btnSecondary")
        self.btn_reset_tree_filter.setStyleSheet("padding: 4px 8px; font-size: 11px;")
        self.btn_reset_tree_filter.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset_tree_filter.clicked.connect(self._reset_tree_filter)
        h_factions.addWidget(self.btn_reset_tree_filter)
        v_filter.addLayout(h_factions)

        self.grp_stand_filter.setVisible(False)
        v_tree.addWidget(self.grp_stand_filter)

        layout.addWidget(grp_tree)

        # 1. Active Selection Group
        grp_edit = QGroupBox("ACTIVE DELINEATION / SELECTION")
        v_edit = QVBoxLayout(grp_edit)
        v_edit.setSpacing(8)

        self.lbl_sel_count = QLabel("Selection: None")
        self.lbl_sel_count.setStyleSheet("font-weight: 600; color: #60a5fa; font-size: 12px;")
        v_edit.addWidget(self.lbl_sel_count)

        self.lbl_sel_classes = QLabel("Classes: -")
        self.lbl_sel_classes.setStyleSheet("color: #94a3b8; font-size: 11px;")
        v_edit.addWidget(self.lbl_sel_classes)

        self.lbl_sel_itc = QLabel("ITC IDs: -")
        self.lbl_sel_itc.setStyleSheet("color: #94a3b8; font-size: 11px;")
        v_edit.addWidget(self.lbl_sel_itc)

        # Action: Assign Tree ID (ITC)
        h_itc = QHBoxLayout()
        lbl_itc = QLabel("Tree ID:")
        lbl_itc.setStyleSheet("color: #94a3b8; font-size: 11px;")
        h_itc.addWidget(lbl_itc)
        self.spin_itc = QSpinBox()
        self.spin_itc.setRange(1, 999999)
        self.spin_itc.setValue(1)
        self.spin_itc.setStyleSheet("font-size: 11px; padding: 3px 6px;")
        self.spin_itc.setCursor(Qt.CursorShape.PointingHandCursor)
        h_itc.addWidget(self.spin_itc)
        h_itc.addStretch()
        v_edit.addLayout(h_itc)

        btn_assign_itc = QPushButton("Stage Tree ID to Polygon")
        btn_assign_itc.setStyleSheet("padding: 5px 10px; font-size: 11px;")
        btn_assign_itc.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_assign_itc.clicked.connect(self.apply_assign_itc)
        v_edit.addWidget(btn_assign_itc)

        # Action: Mark as Noise
        btn_noise = QPushButton("Stage Selection as Noise (Class 7)")
        btn_noise.setObjectName("btnDanger")
        btn_noise.setStyleSheet("padding: 5px 10px; font-size: 11px;")
        btn_noise.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_noise.clicked.connect(self.apply_mark_noise)
        v_edit.addWidget(btn_noise)

        # Action: Set Classification
        h_cls = QHBoxLayout()
        lbl_cls = QLabel("ASPRS Class:")
        lbl_cls.setStyleSheet("color: #94a3b8; font-size: 11px;")
        h_cls.addWidget(lbl_cls)
        self.combo_class = QComboBox()
        self.combo_class.setStyleSheet("font-size: 11px; padding: 3px 6px;")
        self.combo_class.setCursor(Qt.CursorShape.PointingHandCursor)
        self.combo_class.addItem("0: Unclassified", 0)
        self.combo_class.addItem("1: Unassigned", 1)
        self.combo_class.addItem("2: Ground", 2)
        self.combo_class.addItem("3: Low Vegetation", 3)
        self.combo_class.addItem("4: Medium Vegetation", 4)
        self.combo_class.addItem("5: High Vegetation", 5)
        self.combo_class.addItem("7: Noise", 7)
        h_cls.addWidget(self.combo_class)
        h_cls.addStretch()
        v_edit.addLayout(h_cls)

        btn_set_class = QPushButton("Stage Classification")
        btn_set_class.setObjectName("btnSecondary")
        btn_set_class.setStyleSheet("padding: 5px 10px; font-size: 11px;")
        btn_set_class.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_set_class.clicked.connect(self.apply_set_classification)
        v_edit.addWidget(btn_set_class)

        btn_clear = QPushButton("Clear Current Delineation")
        btn_clear.setObjectName("btnSecondary")
        btn_clear.setStyleSheet("padding: 5px 10px; font-size: 11px;")
        btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_clear.clicked.connect(self.clear_selection)
        v_edit.addWidget(btn_clear)

        layout.addWidget(grp_edit)

        # 2. Optimistic Visual Staging & Deferred Commit Group
        grp_staged = QGroupBox("STAGED EDITS (PENDING COMMIT)")
        v_staged = QVBoxLayout(grp_staged)
        v_staged.setSpacing(8)

        self.lbl_staged_count = QLabel("Staged Edits: 0 (Ready to Commit)")
        self.lbl_staged_count.setStyleSheet("font-weight: 600; color: #10b981;")
        v_staged.addWidget(self.lbl_staged_count)

        self.table_staged = QTableWidget()
        self.table_staged.setColumnCount(4)
        self.table_staged.setHorizontalHeaderLabels(["Label", "Action", "Type", ""])
        self.table_staged.horizontalHeader().setStretchLastSection(False)
        self.table_staged.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table_staged.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_staged.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_staged.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.table_staged.setColumnWidth(3, 40)
        self.table_staged.setMaximumHeight(220)
        self.table_staged.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_staged.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        v_staged.addWidget(self.table_staged)

        # Commit and Save to LAS/LAZ Button
        btn_commit = QPushButton("Commit & Save to LAS/LAZ...")
        btn_commit.setObjectName("btnSuccess")
        btn_commit.setStyleSheet("padding: 5px 10px; font-size: 11px;")
        btn_commit.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_commit.clicked.connect(self.commit_staged_edits_action)
        v_staged.addWidget(btn_commit)

        btn_clear_staged = QPushButton("Clear All Staged Edits")
        btn_clear_staged.setObjectName("btnDanger")
        btn_clear_staged.setStyleSheet("padding: 5px 10px; font-size: 11px;")
        btn_clear_staged.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_clear_staged.clicked.connect(self.clear_all_staged_edits)
        v_staged.addWidget(btn_clear_staged)

        btn_history = QPushButton("Edit History & Undo / Revert...")
        btn_history.setStyleSheet("padding: 5px 10px; font-size: 11px; background-color: #334155; color: #e2e8f0; border-radius: 4px;")
        btn_history.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_history.setToolTip("View persistent SQLite log of all octree patches and splits, and revert changes in-place.")
        btn_history.clicked.connect(self.open_edit_history_dialog)
        v_staged.addWidget(btn_history)

        layout.addWidget(grp_staged)
        layout.addStretch()

        scroll.setWidget(content)
        return scroll

    def create_dataset_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(28)
        layout.setContentsMargins(12, 12, 12, 12)

        # 1. Dataset Management Group
        grp_data = QGroupBox("POINT CLOUD DATASET")
        v_data = QVBoxLayout(grp_data)
        v_data.setSpacing(8)

        self.lbl_cloud_info = QLabel("No point cloud loaded.")
        self.lbl_cloud_info.setWordWrap(True)
        self.lbl_cloud_info.setStyleSheet("color: #cbd5e1; font-size: 11px;")
        v_data.addWidget(self.lbl_cloud_info)

        btn_open_potree = QPushButton("Open Potree Converted...")
        btn_open_potree.setStyleSheet("padding: 5px 10px; font-size: 11px;")
        btn_open_potree.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_open_potree.clicked.connect(self.open_potree_file)
        v_data.addWidget(btn_open_potree)

        btn_open_las = QPushButton("Open LAS / LAZ (Header Only)...")
        btn_open_las.setObjectName("btnSecondary")
        btn_open_las.setStyleSheet("padding: 5px 10px; font-size: 11px;")
        btn_open_las.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_open_las.clicked.connect(self.open_las_file)
        v_data.addWidget(btn_open_las)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        v_data.addWidget(self.progress_bar)

        layout.addWidget(grp_data)

        # 2. Save Status & Export
        grp_save = QGroupBox("EXPORT & STATUS")
        v_save = QVBoxLayout(grp_save)
        v_save.setSpacing(8)

        self.lbl_save_status = QLabel("Status: <span style='color:#10b981;'>Ready</span>")
        self.lbl_save_status.setStyleSheet("font-size: 11px;")
        v_save.addWidget(self.lbl_save_status)

        layout.addWidget(grp_save)
        layout.addStretch()

        scroll.setWidget(content)
        return scroll

    # -------------------------------------------------------------------------
    # Layer Manager Toggles & Synchronization
    # -------------------------------------------------------------------------
    def toggle_pointcloud_layer(self, visible: bool):
        if hasattr(self, "chk_layer_pc"):
            self.chk_layer_pc.blockSignals(True)
            self.chk_layer_pc.setChecked(visible)
            self.chk_layer_pc.blockSignals(False)
        if hasattr(self, "ribbon_chk_pc"):
            self.ribbon_chk_pc.blockSignals(True)
            self.ribbon_chk_pc.setChecked(visible)
            self.ribbon_chk_pc.blockSignals(False)
        self._run_js(f"window.setPointCloudVisibility({str(visible).lower()});")

    def toggle_staged_layer(self, visible: bool):
        if hasattr(self, "chk_layer_staged"):
            self.chk_layer_staged.blockSignals(True)
            self.chk_layer_staged.setChecked(visible)
            self.chk_layer_staged.blockSignals(False)
        if hasattr(self, "ribbon_chk_staged"):
            self.ribbon_chk_staged.blockSignals(True)
            self.ribbon_chk_staged.setChecked(visible)
            self.ribbon_chk_staged.blockSignals(False)
        self._run_js(f"window.setStagedAllVisibility({str(visible).lower()});")

    def toggle_pick_layer(self, visible: bool):
        if hasattr(self, "chk_layer_pick"):
            self.chk_layer_pick.blockSignals(True)
            self.chk_layer_pick.setChecked(visible)
            self.chk_layer_pick.blockSignals(False)
        if hasattr(self, "ribbon_chk_pick"):
            self.ribbon_chk_pick.blockSignals(True)
            self.ribbon_chk_pick.setChecked(visible)
            self.ribbon_chk_pick.blockSignals(False)
        self._run_js(f"window.setHighlightVisibility({str(visible).lower()});")

    def toggle_sel_layer(self, visible: bool):
        if hasattr(self, "chk_layer_sel"):
            self.chk_layer_sel.blockSignals(True)
            self.chk_layer_sel.setChecked(visible)
            self.chk_layer_sel.blockSignals(False)
        if hasattr(self, "ribbon_chk_sel"):
            self.ribbon_chk_sel.blockSignals(True)
            self.ribbon_chk_sel.setChecked(visible)
            self.ribbon_chk_sel.blockSignals(False)
        self._run_js(f"window.setSelectionVisibility({str(visible).lower()});")

    def show_all_layers(self):
        self.toggle_pointcloud_layer(True)
        self.toggle_staged_layer(True)
        self.toggle_pick_layer(True)
        self.toggle_sel_layer(True)
        self._run_js("window.setAllLayersVisibility(true);")

    def hide_all_layers(self):
        self.toggle_pointcloud_layer(False)
        self.toggle_staged_layer(False)
        self.toggle_pick_layer(False)
        self.toggle_sel_layer(False)
        self._run_js("window.setAllLayersVisibility(false);")

    def clear_picked_highlight(self):
        self._run_js("window.clearPickedPointHighlight();")
        if hasattr(self, "lbl_layer_pick_coords"):
            self.lbl_layer_pick_coords.setText("Position: Click a point to inspect")
        if hasattr(self, "lbl_layer_pick_attr"):
            self.lbl_layer_pick_attr.setText("Class: - | Tree ID: - | Intensity: -")
        if hasattr(self, "btn_layer_view_tree"):
            self.btn_layer_view_tree.setVisible(False)

    def _on_layer_staged_double_clicked(self, row: int, col: int):
        staged = self.manager.get_staged_edits()
        if 0 <= row < len(staged):
            edit_id = staged[row].get("id")
            if edit_id:
                self._run_js(f"window.zoomToStagedEdit('{edit_id}');")

    def _on_layer_color_changed(self):
        mode = self.combo_layer_color.currentData()
        self._sync_color_combos(mode)
        self.set_color_mode(mode)

    def _on_layer_size_slider_changed(self, ival: int):
        fval = round(ival / 10.0, 2)
        if hasattr(self, "lbl_layer_size_val"):
            self.lbl_layer_size_val.setText(f"{fval:.2f}")
        if hasattr(self, "ribbon_slider_size"):
            self.ribbon_slider_size.blockSignals(True)
            self.ribbon_slider_size.setValue(ival)
            self.ribbon_slider_size.blockSignals(False)
        if hasattr(self, "ribbon_spin_size"):
            self.ribbon_spin_size.blockSignals(True)
            self.ribbon_spin_size.setValue(fval)
            self.ribbon_spin_size.blockSignals(False)
        self.set_point_size(fval)

    def toggle_single_staged_visibility(self, edit_id: str, visible: bool):
        for edit in self.manager.staged_edits:
            if edit.get("id") == edit_id:
                edit["visible"] = visible
                break
        self._run_js(f"window.setStagedItemVisibility('{edit_id}', {str(visible).lower()});")

    # -------------------------------------------------------------------------
    # Dataset Loading & Fast Header-Only Inspection
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # File Dialogs (with a useful sidebar)
    # -------------------------------------------------------------------------
    def _default_sidebar_urls(self):
        """Sidebar locations shared by every file dialog.

        The Qt file dialog keeps its own bookmark list and does not read
        Dolphin's Places, so seed it with the folders used most in this app.
        """
        home = os.path.expanduser("~")
        cands = [
            home,
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation),
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation),
            "/mnt/omen/Data",
            os.path.dirname(os.path.abspath(self.potree_dir)),
        ]
        seen, urls = set(), []
        for p in cands:
            if p and os.path.isdir(p) and p not in seen:
                seen.add(p)
                urls.append(QUrl.fromLocalFile(p))
        return urls

    def _pick_file(self, caption: str, start_dir: str = "", filter_str: str = "All Files (*)") -> str:
        dlg = QFileDialog(self, caption, start_dir, filter_str)
        dlg.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        urls = self._default_sidebar_urls()
        if urls:
            dlg.setSidebarUrls(urls)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            files = dlg.selectedFiles()
            return files[0] if files else ""
        return ""

    def _save_file(self, caption: str, start_file: str, filter_str: str = "") -> str:
        dlg = QFileDialog(self, caption, start_file, filter_str)
        dlg.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        urls = self._default_sidebar_urls()
        if urls:
            dlg.setSidebarUrls(urls)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            files = dlg.selectedFiles()
            return files[0] if files else ""
        return ""

    def open_potree_file(self):
        """Open an existing Potree metadata.json / cloud.js, or a COPC (.copc.laz).

        Potree dispatches on the entry file name: *.copc.laz goes to the COPC loader
        and streams by byte range, so it opens without conversion.
        """
        file_path = self._pick_file(
            "Select Potree Converted File or COPC",
            "",
            "Potree Files (metadata.json cloud.js *.copc.laz);;LAZ/LAS (*.laz *.las);;All Files (*)"
        )
        if not file_path:
            return
        self.open_potree_path(file_path)

    def open_potree_path(self, file_path: str):
        """Programmatic half of open_potree_file — usable from CLI auto-load too."""
        folder = os.path.dirname(os.path.abspath(file_path))
        entry_file = os.path.basename(file_path)
        self.current_potree_dir = folder
        if self.manager:
            self.manager.potree_dir = folder

        alias = "cloud_" + str(abs(hash(folder)) % 100000)
        self.server.register_pointcloud_dir(alias, folder)

        url = f"http://127.0.0.1:{self.server.port}/clouds/{alias}/{entry_file}"
        # Name the layer for what was opened: the folder for a converted cloud (its
        # entry file is an implementation detail), the file name for COPC/LAS.
        display_name = (os.path.basename(folder)
                        if entry_file.lower() in ("metadata.json", "cloud.js")
                        else entry_file)
        self.load_url_in_potree(url, display_name)

        if hasattr(self, "lbl_layer_cloud_name"):
            self.lbl_layer_cloud_name.setText(entry_file)
        if hasattr(self, "lbl_layer_cloud_pts"):
            self.lbl_layer_cloud_pts.setText("Streaming LOD")

        # Bind the master LAS/LAZ this converted folder was built from, so tree
        # queries hit the same z-system the viewer shows. Order: the user's
        # remembered pick for this folder, the source recorded by the converter
        # in metadata.json ("name" = master stem), folder-name conventions
        # (_converted, octree-depth tags like _d10), any LAS in the parent
        # directory; ask and remember the pick if none of those exist.
        parent_dir = os.path.dirname(folder)
        found_las = False
        candidates = []
        saved_pick = self.settings_store.get(f"master_las::{folder}")
        if saved_pick:
            candidates.append(saved_pick)
        meta_path = os.path.join(folder, "metadata.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    src_name = json.load(f).get("name", "")
                if src_name:
                    candidates.append(os.path.join(parent_dir, src_name + ".laz"))
                    candidates.append(os.path.join(parent_dir, src_name + ".las"))
            except Exception:
                pass
        base_name = os.path.basename(folder).replace("_converted", "")
        base_name = re.sub(r"_d\d+$", "", base_name)   # octree-depth suffix, e.g. _d10
        candidates += [os.path.join(parent_dir, base_name + ext) for ext in (".laz", ".las")]
        for cand in candidates:
            if cand and os.path.exists(cand):
                self.load_las_header_only(cand)
                found_las = True
                break
        if not found_las:
            # No explicit pairing: ask the user and remember the pick for this
            # folder, so the binding survives restarts and folder renames of
            # the master.
            picked = self._pick_file(
                f"Select the master LAS/LAZ for {os.path.basename(folder)}",
                parent_dir,
                "LAS/LAZ Files (*.las *.laz);;All Files (*)")
            if picked:
                self.settings_store.set(f"master_las::{folder}", picked)
                self.load_las_header_only(picked)
                self.status_bar.showMessage(f"Master file bound: {os.path.basename(picked)}")
                found_las = True
        if not found_las and os.path.isdir(parent_dir):
            for ext in (".laz", ".las"):
                for fname in sorted(os.listdir(parent_dir)):
                    if fname.lower().endswith(ext):
                        cand = os.path.join(parent_dir, fname)
                        self.load_las_header_only(cand)
                        found_las = True
                        break
                if found_las:
                    break

    def open_folder_dialog(self):
        """Open a directory containing point cloud data (either converted or LAS/LAZ)."""
        folder = QFileDialog.getExistingDirectory(self, "Select Point Cloud Folder")
        if folder:
            self._load_initial_path(folder)

    def open_las_file(self):
        """Open a raw LAS/LAZ file, load header immediately, and offer streaming view."""
        las_path = self._pick_file(
            "Select LAS / LAZ Point Cloud",
            "",
            "LAS/LAZ Files (*.las *.laz);;All Files (*)"
        )
        if not las_path:
            return
        self.open_las_path(las_path)

    def open_las_path(self, las_path: str, offer_conversion: bool = True):
        """Programmatic half of open_las_file — usable from CLI auto-load too."""
        # 1. Read header metadata instantly without memory allocation
        self.load_las_header_only(las_path)

        # 2. Check if a converted sibling directory already exists
        parent_dir = os.path.dirname(las_path)
        base_name = os.path.splitext(os.path.basename(las_path))[0]
        converted_dir = os.path.join(parent_dir, base_name + "_converted")

        meta_json = os.path.join(converted_dir, "metadata.json")
        cloud_js = os.path.join(converted_dir, "cloud.js")

        if os.path.exists(meta_json) or os.path.exists(cloud_js):
            self.current_potree_dir = converted_dir
            if self.manager:
                self.manager.potree_dir = converted_dir
            entry_file = "metadata.json" if os.path.exists(meta_json) else "cloud.js"
            alias = "cloud_" + str(abs(hash(converted_dir)) % 100000)
            self.server.register_pointcloud_dir(alias, converted_dir)
            url = f"http://127.0.0.1:{self.server.port}/clouds/{alias}/{entry_file}"
            self.load_url_in_potree(url, os.path.basename(converted_dir))
        elif offer_conversion:
            reply = QMessageBox.question(
                self,
                "Convert Point Cloud for Potree?",
                f"To stream '{os.path.basename(las_path)}' with Potree's LOD octree, it needs conversion.\n\n"
                f"Would you like to convert it now using PotreeConverter?\n"
                f"(Output: {converted_dir})",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.start_conversion(las_path, converted_dir)

    def load_las_header_only(self, las_path: str):
        """Instantly read metadata header without loading heavy point arrays into RAM."""
        try:
            info = self.manager.load_las_header(las_path)
            self.current_las_path = las_path
            itc_status = "Yes (Available)" if info.get("has_itc") else "Will be added on commit"
            self.lbl_cloud_info.setText(
                f"<b>File:</b> {info['filename']}<br>"
                f"<b>Points:</b> {info['total_points']:,} (Streaming LOD)<br>"
                f"<b>ITC Field:</b> {itc_status}<br>"
                f"<b>Mode:</b> Optimistic Visual Staging"
            )
            if hasattr(self, "lbl_layer_cloud_name"):
                self.lbl_layer_cloud_name.setText(info['filename'])
            if hasattr(self, "lbl_layer_cloud_pts"):
                self.lbl_layer_cloud_pts.setText(f"{info['total_points']:,} points (Streaming LOD)")
            dims = info.get("dimensions") or info.get("extra_dimensions") or []
            if dims:
                self._sync_color_combos(None, available_attrs=dims)
            self.status_bar.showMessage(f"Dataset '{info['filename']}' ({info['total_points']:,} points) ready for visual staging.")
        except Exception as e:
            print(f"[MainWindow] Error loading LAS header: {e}")

    def load_las_for_editing(self, las_path: str):
        """Full in-memory load for smaller clouds."""
        try:
            info = self.manager.load_las(las_path)
            self.current_las_path = las_path
            self.lbl_cloud_info.setText(
                f"<b>File:</b> {info['filename']}<br>"
                f"<b>Points:</b> {info['total_points']:,}<br>"
                f"<b>Classes:</b> {info.get('classes', [])}<br>"
                f"<b>Has ITC Field:</b> {'Yes' if info.get('has_itc') else 'No (auto-created)'}"
            )
            if hasattr(self, "lbl_layer_cloud_name"):
                self.lbl_layer_cloud_name.setText(info['filename'])
            if hasattr(self, "lbl_layer_cloud_pts"):
                self.lbl_layer_cloud_pts.setText(f"{info['total_points']:,} points (In-Memory)")
            dims = info.get("dimensions") or info.get("extra_dimensions") or []
            if dims:
                self._sync_color_combos(None, available_attrs=dims)
            self.status_bar.showMessage(f"Loaded '{info['filename']}' ({info['total_points']:,} points) in memory.")
        except Exception as e:
            QMessageBox.critical(self, "Error Loading LAS", str(e))

    def convert_las_action(self):
        """Prompt user to select a LAS/LAZ file and convert it into a Potree LOD octree."""
        las_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select LAS / LAZ Point Cloud to Convert",
            "",
            "LAS/LAZ Files (*.las *.laz);;All Files (*)"
        )
        if not las_path:
            return

        parent_dir = os.path.dirname(las_path)
        base_name = os.path.splitext(os.path.basename(las_path))[0]
        default_out = os.path.join(parent_dir, base_name + "_converted")

        out_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Destination Directory for Converted Potree Dataset",
            default_out
        )
        if not out_dir:
            return

        # If user picked a directory that already contains metadata.json, warn them
        if os.path.exists(os.path.join(out_dir, "metadata.json")) or os.path.exists(os.path.join(out_dir, "cloud.js")):
            reply = QMessageBox.question(
                self,
                "Overwrite Existing Potree Dataset?",
                f"Directory '{os.path.basename(out_dir)}' already contains a Potree dataset.\n\n"
                f"Are you sure you want to re-convert and overwrite it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self.start_conversion(las_path, out_dir)

    def start_conversion(self, input_las: str, out_dir: str):
        """Launch the Potree LOD octree converter with live log streaming and statistics."""
        converter_bin = find_potree_converter(self.potree_dir)
        if not converter_bin:
            QMessageBox.critical(
                self,
                "PotreeConverter Not Found",
                "Could not find the PotreeConverter executable.\n\n"
                "Please verify that PotreeConverter is built in 'tools/PC-2.1.5/build/PotreeConverter' "
                "or installed in your system PATH."
            )
            return

        # Pre-load header and initiate automatic background indexing for input LAS immediately
        try:
            self.load_las_header_only(input_las)
        except Exception as e:
            print(f"[MainWindow] Pre-conversion header read note: {e}")

        self.status_bar.showMessage(f"Converting '{os.path.basename(input_las)}' with PotreeConverter...")

        self.conversion_dialog = ConversionDialog(
            converter_bin=converter_bin,
            input_las=input_las,
            out_dir=out_dir,
            parent=self,
            auto_start=True
        )
        self.conversion_dialog.conversion_completed.connect(self._on_conversion_completed)
        self.conversion_dialog.show()

    def _on_conversion_completed(self, out_dir: str, stats: dict):
        """Handle successful conversion: register HTTP route and stream in Potree."""
        try:
            # Ensure the source LAS is set as current_las_path and indexing is active/ready
            if self.conversion_dialog and self.conversion_dialog.input_las:
                input_las = self.conversion_dialog.input_las
                if self.current_las_path != input_las:
                    self.load_las_header_only(input_las)

            alias = "cloud_" + str(abs(hash(out_dir)) % 100000)
            self.server.register_pointcloud_dir(alias, out_dir)
            entry_file = "metadata.json" if os.path.exists(os.path.join(out_dir, "metadata.json")) else "cloud.js"
            url = f"http://127.0.0.1:{self.server.port}/clouds/{alias}/{entry_file}"
            self.load_url_in_potree(url, os.path.basename(out_dir))

            pts_str = stats.get("#points", "")
            if pts_str:
                self.status_bar.showMessage(f"Successfully converted and loaded '{os.path.basename(out_dir)}' ({pts_str} points).", 8000)
            else:
                self.status_bar.showMessage(f"Successfully converted and loaded '{os.path.basename(out_dir)}'.", 8000)
        except Exception as e:
            print(f"[MainWindow] Error loading converted cloud: {e}")

    def _sync_style_widgets(self, info):
        """Point the shape, sizing-mode and size controls at the active layer's style.

        Each cloud now carries its own material style, so switching layers has to show
        that layer's values instead of whatever the controls were last moved to.
        """
        for name, key in (("ribbon_combo_shape", "pointType"),
                          ("ribbon_combo_sizing", "pointSizeType")):
            combo = getattr(self, name, None)
            value = info.get(key)
            if combo is None or value is None:
                continue
            idx = combo.findData(str(value))
            if idx >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)
                combo.setEnabled(True)
        size = info.get("pointSize")
        if size is None:
            return
        fval = max(0.10, min(4.00, float(size)))
        ival = int(round(fval * 10))
        for name in ("ribbon_spin_size", "ribbon_slider_size", "slider_layer_size"):
            widget = getattr(self, name, None)
            if widget is None:
                continue
            widget.blockSignals(True)
            widget.setValue(ival if isinstance(widget, QSlider) else fval)
            widget.blockSignals(False)
            widget.setEnabled(True)
        if hasattr(self, "lbl_layer_size_val"):
            self.lbl_layer_size_val.setText(f"{fval:.2f}")

    @staticmethod
    def _format_attribute_label(attr: str) -> str:
        mapping = {
            "rgba": "RGB / True Color",
            "rgb": "RGB / True Color",
            "color": "RGB / True Color",
            "classification": "ASPRS Classification",
            "elevation": "Elevation Gradient",
            "intensity": "Intensity",
            "intensity gradient": "Intensity Gradient",
            "itc": "ITC (Tree ID)",
            "tree_id": "Tree ID",
            "treeID": "Tree ID",
            "tree_score": "Tree Score",
            "semantic_pred": "Semantic Prediction",
            "return number": "Return Number",
            "number of returns": "Number of Returns",
            "point source id": "Point Source ID",
            "source id": "Point Source ID",
            "gps-time": "GPS Time",
            "gps_time": "GPS Time",
            "scan angle": "Scan Angle",
            "user data": "User Data",
            "classification flags": "Classification Flags",
        }
        return mapping.get(str(attr), str(attr).replace("_", " ").replace("-", " ").title())

    def _sync_color_combos(self, attr: Optional[str], available_attrs: Optional[list] = None):
        """Point both Color By dropdowns at one attribute and populate with all available scalar fields."""
        if not attr and not available_attrs:
            return

        PRIORITY_ORDER = [
            "rgba", "color", "rgb",
            "classification",
            "elevation",
            "intensity",
            "intensity gradient",
            "itc", "tree_id", "treeID",
            "tree_score",
            "semantic_pred",
            "return number",
            "number of returns",
            "point source id",
            "source id",
            "gps-time",
            "scan angle",
            "user data",
            "classification flags",
        ]

        if available_attrs:
            def attr_sort_key(name):
                try:
                    return (0, PRIORITY_ORDER.index(name))
                except ValueError:
                    return (1, str(name))

            sorted_attrs = sorted(set(available_attrs), key=attr_sort_key)

            # 1. Update Layers tab & Ribbon combos
            for combo in (getattr(self, "combo_layer_color", None),
                          getattr(self, "ribbon_combo_color", None)):
                if combo is None:
                    continue
                combo.blockSignals(True)
                current_selected = attr or combo.currentData() or "rgba"
                combo.clear()
                for a in sorted_attrs:
                    combo.addItem(self._format_attribute_label(str(a)), str(a))
                idx = combo.findData(str(current_selected))
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                elif combo.count() > 0:
                    combo.setCurrentIndex(0)
                combo.blockSignals(False)
                combo.setEnabled(True)

            # 2. Update Forestry tab (individual tree view) combo with discovered scalar fields
            combo_tree = getattr(self, "combo_tree_color", None)
            if combo_tree is not None:
                combo_tree.blockSignals(True)
                cur_tree_sel = combo_tree.currentData() or "auto"
                combo_tree.clear()
                combo_tree.addItem("Match Main Layer (Auto)", "auto")
                combo_tree.addItem("Tree / ITC (Segmented Stand)", "itc")
                combo_tree.addItem("Natural Foliage Green", "green")
                combo_tree.addItem("Viridis Heatmap", "viridis")
                for a in sorted_attrs:
                    if str(a) == "itc":
                        continue
                    combo_tree.addItem(self._format_attribute_label(str(a)), str(a))
                t_idx = combo_tree.findData(str(cur_tree_sel))
                if t_idx >= 0:
                    combo_tree.setCurrentIndex(t_idx)
                else:
                    combo_tree.setCurrentIndex(0)
                combo_tree.blockSignals(False)
                combo_tree.setEnabled(True)
        elif attr:
            for combo in (getattr(self, "combo_layer_color", None),
                          getattr(self, "ribbon_combo_color", None),
                          getattr(self, "combo_tree_color", None)):
                if combo is None:
                    continue
                idx = combo.findData(str(attr))
                if idx < 0:
                    combo.addItem(self._format_attribute_label(str(attr)), str(attr))
                    idx = combo.count() - 1
                if combo is not getattr(self, "combo_tree_color", None):
                    combo.blockSignals(True)
                    combo.setCurrentIndex(idx)
                    combo.blockSignals(False)
                combo.setEnabled(True)

    def refresh_layers(self, tries: int = 0):
        """Ask the viewer for its clouds and rebuild the layer rows.

        The list lives in JavaScript (viewer.scene.pointclouds is potree's own layer
        manager), so the rows are pulled, not tracked twice.  Loading is asynchronous
        and a big cloud takes a while to register, so an empty answer is retried rather
        than leaving the panel showing nothing.
        """
        def handle(result):
            if not isinstance(result, list):
                return
            if not result and tries < 12:
                QTimer.singleShot(1500, lambda: self.refresh_layers(tries + 1))
                return
            self._render_layer_rows(result)

        self._run_js("window.listLayers ? window.listLayers() : [];", handle)

    def _render_layer_rows(self, layers: list):
        """Rebuild the rows, marking the active one bold and selecting it."""
        self._layer_sync_guard = True
        try:
            self.layer_list.clear()
            active_row = -1
            active_info = None
            for row, info in enumerate(layers):
                name = str(info.get("name"))
                item = QListWidgetItem(name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if info.get("visible")
                                   else Qt.CheckState.Unchecked)
                item.setData(Qt.ItemDataRole.UserRole, name)
                if info.get("active"):
                    active_row, active_info = row, info
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.layer_list.addItem(item)
            if active_row < 0 and layers:
                active_row, active_info = 0, layers[0]
                self.layer_list.item(0).setFont(self._bold_font(self.layer_list.item(0)))
            if active_row >= 0:
                self.layer_list.setCurrentRow(active_row)
            if active_info:
                self.lbl_layer_cloud_name.setText(f"Active layer: {active_info.get('name')}")
                self.lbl_layer_cloud_pts.setText(f"{int(active_info.get('points') or 0):,} points")
                self._sync_color_combos(active_info.get("colorAttribute"), available_attrs=active_info.get("attributes"))
                self._sync_style_widgets(active_info)
            else:
                self.lbl_layer_cloud_name.setText("No point cloud loaded")
                self.lbl_layer_cloud_pts.setText("- points")
                # Nothing to recolour: grey both dropdowns instead of pretending.
                for combo in (getattr(self, "combo_layer_color", None),
                              getattr(self, "ribbon_combo_color", None)):
                    if combo is not None:
                        combo.setEnabled(False)
        finally:
            self._layer_sync_guard = False

    @staticmethod
    def _bold_font(item):
        font = item.font()
        font.setBold(True)
        return font

    def _on_layer_row_selected(self):
        """Clicking a row makes that cloud the active layer in the viewer."""
        if self._layer_sync_guard:
            return
        item = self.layer_list.currentItem()
        if not item:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        self._run_js(
            f"window.setActiveLayer({json.dumps(name)});",
            lambda res: self._render_layer_rows(res if isinstance(res, list) else []))

    def _on_layer_row_checked(self, item):
        """The row checkbox is pc.visible, potree's own per-layer visibility flag."""
        if self._layer_sync_guard:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        visible = item.checkState() == Qt.CheckState.Checked
        self._run_js(
            f"window.setLayerVisible({json.dumps(name)}, {str(bool(visible)).lower()});")

    def _on_layer_remove_clicked(self):
        """Drop one cloud from the scene list and the graph, leaving the others alone."""
        item = self.layer_list.currentItem()
        if not item:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        self._run_js(
            f"window.removeLayer({json.dumps(name)});",
            lambda res: self._render_layer_rows(res if isinstance(res, list) else []))

    def load_url_in_potree(self, url: str, display_name: str = ""):
        """Load a cloud in the viewer as another layer.

        display_name is what the layer row shows.  It has to be passed in: the server
        mounts every cloud under a hashed alias (cloud_25662), so the URL carries
        nothing a person could read, and the cloud's own metadata name is generic.
        """
        self.current_potree_url = url
        js = f"window.loadPointCloudUrl({json.dumps(url)}, {json.dumps(display_name or '')});"
        self._run_js(js)
        # The load resolves on its own schedule; refresh_layers retries until the new
        # cloud shows up, which is what makes it a layer instead of a replacement.
        QTimer.singleShot(600, lambda: self.refresh_layers(0))

    # -------------------------------------------------------------------------
    # Selection & Optimistic Visual Staging
    # -------------------------------------------------------------------------
    def on_selection_changed(self, data: dict):
        # Polygon/box tools finish in JS (activeTool back to 'nav'); mirror that
        # in the ribbon button states. Pick Point stays active until toggled.
        if self._active_tool in ("polygon", "box"):
            self._active_tool = "nav"
            self._sync_tool_buttons()
        if data.get("mode") == "staged":
            vcount = data.get("vertex_count", 0)
            self.lbl_sel_count.setText(f"Selection: Polygon ({vcount} vertices)")
            self.lbl_sel_classes.setText("Classes: (Pending Commit)")
            self.lbl_sel_itc.setText("ITC IDs: (Pending Commit)")
            if hasattr(self, "lbl_layer_sel_info"):
                self.lbl_layer_sel_info.setText(f"Polygon ({vcount} vertices)")
            if hasattr(self, "chk_layer_sel"):
                self.chk_layer_sel.blockSignals(True)
                self.chk_layer_sel.setChecked(True)
                self.chk_layer_sel.blockSignals(False)
            if hasattr(self, "ribbon_chk_sel"):
                self.ribbon_chk_sel.blockSignals(True)
                self.ribbon_chk_sel.setChecked(True)
                self.ribbon_chk_sel.blockSignals(False)
            self.status_bar.showMessage(f"Delineated polygon with {vcount} vertices. Ready to stage Tree ID or Noise.")
            return

        count = data.get("count", 0)
        classes = data.get("classes", [])
        itc_vals = data.get("itc_values", [])

        self.lbl_sel_count.setText(f"Selected: {count:,} points")
        self.lbl_sel_classes.setText(f"Classes: {classes if classes else '-'}")
        self.lbl_sel_itc.setText(f"ITC IDs: {itc_vals[:6] if itc_vals else '-'}")
        if hasattr(self, "lbl_layer_sel_info"):
            self.lbl_layer_sel_info.setText(f"Active: {count:,} points")
        if hasattr(self, "chk_layer_sel"):
            self.chk_layer_sel.blockSignals(True)
            self.chk_layer_sel.setChecked(True)
            self.chk_layer_sel.blockSignals(False)
        if hasattr(self, "ribbon_chk_sel"):
            self.ribbon_chk_sel.blockSignals(True)
            self.ribbon_chk_sel.setChecked(True)
            self.ribbon_chk_sel.blockSignals(False)
        self.status_bar.showMessage(f"Selected {count:,} points.")

    def on_point_picked(self, data: dict):
        staged_hit = data.get("staged_hit")
        x = data.get("x", 0.0)
        y = data.get("y", 0.0)
        z = data.get("z", 0.0)
        cls_val = data.get("classification")
        itc_val = data.get("itc")
        intensity_val = data.get("intensity")

        if staged_hit:
            self.status_bar.showMessage(
                f"Picked Point: ({x:.2f}, {y:.2f}, {z:.2f}) | "
                f"Inside Staged Region: {staged_hit.get('label')} ({staged_hit.get('action')})"
            )
        else:
            self.status_bar.showMessage(
                f"Picked Point: ({x:.2f}, {y:.2f}, {z:.2f}) | "
                f"Class: {cls_val} | ITC: {itc_val}"
            )

        if hasattr(self, "lbl_layer_pick_coords"):
            self.lbl_layer_pick_coords.setText(f"XYZ: {x:.2f}, {y:.2f}, {z:.2f}")
        if hasattr(self, "lbl_layer_pick_attr"):
            cls_str = f"Class: {cls_val}" if cls_val is not None else "Class: -"
            tree_str = f"Tree ID: {itc_val}" if itc_val is not None else (f"Inside: {staged_hit.get('label')}" if staged_hit else "Tree ID: -")
            int_str = f"Int: {intensity_val}" if intensity_val is not None else "Int: -"
            parts = [cls_str, tree_str, int_str]
            extra_attrs = data.get("attributes", {})
            if isinstance(extra_attrs, dict):
                for k, v in extra_attrs.items():
                    if k in ("classification", "intensity", "itc", "tree_id", "treeID", "x", "y", "z"):
                        continue
                    v_str = f"{v:.3f}" if isinstance(v, float) else str(v)
                    k_label = self._format_attribute_label(k)
                    parts.append(f"{k_label}: {v_str}")
            self.lbl_layer_pick_attr.setText(" | ".join(parts[:5]))
        if hasattr(self, "chk_layer_pick"):
            self.chk_layer_pick.blockSignals(True)
            self.chk_layer_pick.setChecked(True)
            self.chk_layer_pick.blockSignals(False)
        if hasattr(self, "ribbon_chk_pick"):
            self.ribbon_chk_pick.blockSignals(True)
            self.ribbon_chk_pick.setChecked(True)
            self.ribbon_chk_pick.blockSignals(False)

        # Feed the picked tree into the Tree View card (Forestry tab) and quick action in Layers tab
        if itc_val is not None and int(itc_val) > 0:
            itc_int = int(itc_val)
            self._last_picked_itc = itc_int
            self._last_picked_x = float(x)
            self._last_picked_y = float(y)
            if hasattr(self, "btn_layer_view_tree"):
                self.btn_layer_view_tree.setText(f"View Tree #{itc_int} in Forestry Tab ->")
                self.btn_layer_view_tree.setVisible(True)
            if hasattr(self, "spin_tree_view"):
                self.spin_tree_view.blockSignals(True)
                self.spin_tree_view.setValue(itc_int)
                self.spin_tree_view.blockSignals(False)
            if hasattr(self, "spin_ai_tree"):
                self.spin_ai_tree.blockSignals(True)
                self.spin_ai_tree.setValue(itc_int)
                self.spin_ai_tree.blockSignals(False)
            if hasattr(self, "lbl_tree_view_status"):
                self.lbl_tree_view_status.setText(
                    f"Tree #{itc_int} picked — click View to render every point of this tree."
                )
        else:
            if hasattr(self, "btn_layer_view_tree"):
                self.btn_layer_view_tree.setVisible(False)
            if staged_hit and hasattr(self, "lbl_tree_view_status"):
                self.lbl_tree_view_status.setText(
                    f"Picked point is inside staged region “{staged_hit.get('label')}” (not committed yet)."
                )

    def on_tree_view_requested(self, data: dict):
        """Triggered when user clicks 'View Tree' from the WebGL HUD inspector or API."""
        self._run_js("if (document.getElementById('inspectCard')) document.getElementById('inspectCard').style.display = 'none';")
        itc = data.get("itc")
        if itc is not None:
            try:
                itc_int = int(itc)
                self._last_picked_itc = itc_int
                if data.get("x") is not None:
                    self._last_picked_x = float(data["x"])
                if data.get("y") is not None:
                    self._last_picked_y = float(data["y"])
                # HUD "View Tree" uses the JS defaults: single tree, no stand radius
                self._tree_view_state = {
                    "itc": itc_int, "radius": float(data.get("radius", 0.0) or 0.0),
                    "mode": str(data.get("neighbor_mode", "trees_only") or "trees_only"),
                    "ground": 0.0,
                    "x": getattr(self, "_last_picked_x", None),
                    "y": getattr(self, "_last_picked_y", None),
                }
                if hasattr(self, "spin_tree_view"):
                    self.spin_tree_view.blockSignals(True)
                    self.spin_tree_view.setValue(itc_int)
                    self.spin_tree_view.blockSignals(False)
                if hasattr(self, "spin_ai_tree"):
                    self.spin_ai_tree.blockSignals(True)
                    self.spin_ai_tree.setValue(itc_int)
                    self.spin_ai_tree.blockSignals(False)
                if hasattr(self, "lbl_tree_view_status"):
                    self.lbl_tree_view_status.setText(f"Viewing Tree #{itc_int} (full density)...")
                self.status_bar.showMessage(f"Viewing Tree #{itc_int} in Forestry tab.")
            except (TypeError, ValueError):
                pass
        if hasattr(self, "sidebar_tabs"):
            self.sidebar_tabs.setCurrentIndex(1)

    def close_tree_view_action(self):
        """Close full-density tree overlay and return to the Layers tab."""
        self._run_js("if (window.closeTreeOverlay) window.closeTreeOverlay(); else if (window.hideTreeOverlay) window.hideTreeOverlay();")
        self._tree_view_state = None
        if hasattr(self, "btn_extract_tree"):
            self.btn_extract_tree.setEnabled(False)
        if hasattr(self, "btn_confirm_segmentation"):
            self.btn_confirm_segmentation.setEnabled(False)
            self.btn_confirm_segmentation.setText("Confirm & Stage Split Edits")
        if hasattr(self, "grp_stand_filter"):
            self.grp_stand_filter.setVisible(False)
            self.list_stand_trees.clear()
        if hasattr(self, "bar_stand_filter"):
            self.bar_stand_filter.setVisible(True)
            self._set_stand_filter_buttons_enabled(False)
        if hasattr(self, "btn_tree_filter_mode"):
            self.btn_tree_filter_mode.blockSignals(True)
            self.btn_tree_filter_mode.setChecked(False)
            self.btn_tree_filter_mode.setText("Filter Mode")
            self.btn_tree_filter_mode.setStyleSheet("font-size: 11px; padding: 4px; font-weight: 600;")
            self.btn_tree_filter_mode.blockSignals(False)
        if hasattr(self, "sidebar_tabs"):
            self.sidebar_tabs.setCurrentIndex(0)

    def _on_tree_color_changed(self):
        if hasattr(self, "combo_tree_color"):
            mode = self.combo_tree_color.currentData()
            self._run_js(f"if (window.setTreeColorMode) window.setTreeColorMode('{mode}');")

    def _on_tree_size_changed(self, ival: int):
        fval = round(ival / 10.0, 2)
        if hasattr(self, "lbl_tree_size_val"):
            self.lbl_tree_size_val.setText(f"{fval:.2f}")
        self._run_js(f"if (window.setTreePointSize) window.setTreePointSize({fval:.2f});")

    def _on_tree_spin_changed(self, val: int):
        if hasattr(self, "spin_ai_tree") and self.spin_ai_tree.value() != val:
            self.spin_ai_tree.blockSignals(True)
            self.spin_ai_tree.setValue(val)
            self.spin_ai_tree.blockSignals(False)
        if val > 0 and hasattr(self, "manager"):
            if not hasattr(self, "_tree_prefetch_timer"):
                self._tree_prefetch_timer = QTimer(self)
                self._tree_prefetch_timer.setSingleShot(True)
                self._tree_prefetch_timer.setInterval(200)
                self._tree_prefetch_timer.timeout.connect(self._do_prefetch_tree)
            self._pending_prefetch_itc = val
            self._tree_prefetch_timer.start()

    def _do_prefetch_tree(self):
        itc = getattr(self, "_pending_prefetch_itc", None)
        if itc and itc > 0 and hasattr(self, "manager"):
            self.manager.prefetch_tree(itc)

    def _on_tree_neighbors_toggled(self, checked: bool):
        if hasattr(self, "spin_tree_radius"):
            self.spin_tree_radius.setEnabled(checked)
        if hasattr(self, "combo_neighbor_mode"):
            self.combo_neighbor_mode.setEnabled(checked)
        if hasattr(self, "chk_ai_neighbors") and self.chk_ai_neighbors.isChecked() != checked:
            self.chk_ai_neighbors.blockSignals(True)
            self.chk_ai_neighbors.setChecked(checked)
            self.chk_ai_neighbors.blockSignals(False)
        if hasattr(self, "spin_ai_radius"):
            self.spin_ai_radius.setEnabled(checked)

    def _toggle_tree_measure(self, on: bool):
        """Ribbon Point-tab Measure toggle: click-to-click distances on the tree."""
        self._run_js(f"if (window.setTreeMeasure) window.setTreeMeasure({'true' if on else 'false'});")
        self.status_bar.showMessage(
            "Measure: click points on the tree (double-click to clear)" if on else "Measure: off")

    def view_tree_action(self):
        """Render the full point set of one tree (queried from the LAS master)
        as an isolated cloud and switch to the Forestry tab."""
        itc = int(self.spin_tree_view.value())
        if itc <= 0:
            self.status_bar.showMessage("Set a Tree ID (ITC) first, or Ctrl+Click a tree point.")
            return
        if not self.manager.las_path and self.manager.las_data is None:
            self.status_bar.showMessage("No LAS/LAZ source file is bound to query full tree points from.")
            self.lbl_tree_view_status.setText("No LAS/LAZ source file bound. Open a LAS/LAZ or converted dataset with sibling LAS.")
            return

        # Automatically switch to Forestry tab
        if hasattr(self, "sidebar_tabs"):
            self.sidebar_tabs.setCurrentIndex(1)

        radius = float(self.spin_tree_radius.value()) if (hasattr(self, "chk_tree_neighbors") and self.chk_tree_neighbors.isChecked()) else 0.0
        n_mode = self.combo_neighbor_mode.currentData() if hasattr(self, "combo_neighbor_mode") else "all"
        g_rad = float(self.spin_tree_ground.value()) if (hasattr(self, "chk_tree_ground") and self.chk_tree_ground.isChecked()) else 0.0

        hint_x = f"{self._last_picked_x}" if (hasattr(self, "_last_picked_itc") and self._last_picked_itc == itc and hasattr(self, "_last_picked_x")) else "null"
        hint_y = f"{self._last_picked_y}" if (hasattr(self, "_last_picked_itc") and self._last_picked_itc == itc and hasattr(self, "_last_picked_y")) else "null"

        # Remember exactly what is being displayed so "Extract View as LAS/LAZ"
        # re-queries the identical full-density point set from the master file
        self._tree_view_state = {
            "itc": itc, "radius": radius, "mode": n_mode, "ground": g_rad,
            "x": float(self._last_picked_x) if hint_x != "null" else None,
            "y": float(self._last_picked_y) if hint_y != "null" else None,
        }

        is_indexing = getattr(self.manager, "_is_indexing", False)
        fallback_str = " via streaming fallback (background indexing in progress)" if is_indexing else ""

        if radius > 0:
            self.lbl_tree_view_status.setText(f"Querying Tree #{itc} and neighbors ({radius:.0f}m stand){fallback_str}...")
            self.status_bar.showMessage(f"Viewing Tree #{itc} + neighbors within {radius:.0f}m stand{fallback_str}...", 6000)
        else:
            self.lbl_tree_view_status.setText(f"Querying all points of Tree #{itc}{fallback_str}...")
            self.status_bar.showMessage(f"Viewing Tree #{itc} — querying full-density points{fallback_str}...", 6000)

        self._run_js(f"window.viewTreeById({itc}, {hint_x}, {hint_y}, {radius:.1f}, '{n_mode}', {g_rad:.1f});")

    def segment_tree_action(self):
        """Segment the displayed tree/stand with ForestFormer3D (separate
        torch-env subprocess) and rebuild the overlay coloured per detected
        tree instance — e.g. split a merged two-stem ITC tree for inspection."""
        itc = int(self.spin_tree_view.value())
        if itc <= 0:
            self.status_bar.showMessage("Set a Tree ID (ITC) first, or Ctrl+Click a tree point.")
            return
        if not self.manager.las_path and self.manager.las_data is None:
            self.status_bar.showMessage("No LAS/LAZ source file is bound to query full tree points from.")
            return
        if hasattr(self, "sidebar_tabs"):
            self.sidebar_tabs.setCurrentIndex(1)
        if hasattr(self, "btn_tab_ai") and hasattr(self, "ribbon_stack"):
            self.btn_tab_ai.setChecked(True)
            self.ribbon_stack.setCurrentIndex(5)

        radius = float(self.spin_tree_radius.value()) if (hasattr(self, "chk_tree_neighbors") and self.chk_tree_neighbors.isChecked()) else 0.0
        n_mode = self.combo_neighbor_mode.currentData() if hasattr(self, "combo_neighbor_mode") else "all"
        g_rad = float(self.spin_tree_ground.value()) if (hasattr(self, "chk_tree_ground") and self.chk_tree_ground.isChecked()) else 0.0
        hint_x = f"{self._last_picked_x}" if (hasattr(self, "_last_picked_itc") and self._last_picked_itc == itc and hasattr(self, "_last_picked_x")) else "null"
        hint_y = f"{self._last_picked_y}" if (hasattr(self, "_last_picked_itc") and self._last_picked_itc == itc and hasattr(self, "_last_picked_y")) else "null"

        # Check if Stand Filter is active with specific checked trees
        keep_ids = None
        if hasattr(self, "list_stand_trees") and self.list_stand_trees.count() > 0:
            checked = [self.list_stand_trees.item(i).data(Qt.ItemDataRole.UserRole)
                       for i in range(self.list_stand_trees.count())
                       if self.list_stand_trees.item(i).checkState() == Qt.CheckState.Checked]
            if checked and len(checked) < self.list_stand_trees.count():
                keep_ids = checked

        self._tree_view_state = {
            "itc": itc, "radius": radius, "mode": n_mode, "ground": g_rad,
            "x": float(self._last_picked_x) if hint_x != "null" else None,
            "y": float(self._last_picked_y) if hint_y != "null" else None,
            "keep_ids": keep_ids,
        }
        status_target = f"Tree #{itc}" if (keep_ids is None or itc in keep_ids) else f"selected stand points near Tree #{itc}"
        self.lbl_tree_view_status.setText(f"ForestFormer3D segmenting {status_target}... (first run loads the model, ~15 s)")
        self.status_bar.showMessage(f"ForestFormer3D segmentation running on {status_target} (~10 s)...", 15000)

        # Open activity console so the user sees live steps streaming
        if hasattr(self, "console_panel") and not self.console_panel.isVisible():
            self.toggle_console_panel()

        self.log_console("AI", f"Initiating ForestFormer3D AI segmentation for {status_target} (radius={radius:.0f}m, mode={n_mode})...", level="info")
        if keep_ids:
            self.log_console("STAND", f"Stand Filter active: isolating {len(keep_ids)} selected IDs: {keep_ids}", level="info")

        keep_json = json.dumps(keep_ids) if keep_ids is not None else "null"
        self._run_js(f"window.segmentTreeById({itc}, {hint_x}, {hint_y}, {radius:.1f}, '{n_mode}', {g_rad:.1f}, {keep_json});")

    def extract_tree_view_action(self):
        """Write the full-density tree/stand currently displayed in the Forestry tab
        to a new LAS/LAZ file. Points are extracted straight from the master
        LAS/LAZ, and an active Neighbor Stand Filter (checked trees) restricts output."""
        state = getattr(self, "_tree_view_state", None)
        if not state:
            self.status_bar.showMessage("Open a tree in full density first (View Tree / Ctrl+Click).")
            return
        if not self.manager.las_path and self.manager.las_data is None:
            self.status_bar.showMessage("No LAS/LAZ master file bound for extraction.")
            return

        # Restrict to the stand filter selection when trees are listed
        keep_ids = None
        if hasattr(self, "list_stand_trees") and self.list_stand_trees.count() > 0:
            checked = [self.list_stand_trees.item(i).data(Qt.ItemDataRole.UserRole)
                       for i in range(self.list_stand_trees.count())
                       if self.list_stand_trees.item(i).checkState() == Qt.CheckState.Checked]
            keep_ids = checked or None

        itc = int(state.get("itc", 0))
        radius = float(state.get("radius", 0.0))
        suggested = os.path.join(os.path.expanduser("~"),
                                 f"tree_{itc}_stand_{radius:.0f}m.copc.laz" if radius > 0 else f"tree_{itc}.copc.laz")
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Extract Displayed Points as COPC / LAS / LAZ", suggested,
            "COPC LAZ Files (*.copc.laz);;LAZ Files (*.laz);;LAS Files (*.las)")
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".copc.laz" if selected_filter.startswith("COPC") else (
                ".las" if selected_filter.startswith("LAS") else ".laz")

        self.status_bar.showMessage(f"Extracting Tree #{itc} full-density points from master...")
        QApplication.processEvents()
        try:
            count = self.manager.export_tree_points(
                path, itc,
                x=state.get("x"), y=state.get("y"),
                radius=radius, neighbor_mode=state.get("mode", "all"),
                keep_tree_ids=keep_ids,
                ground_radius=float(state.get("ground", 0.0) or 0.0))
        except Exception as exc:
            QMessageBox.warning(self, "Extract Tree Stand", f"Extraction failed: {exc}")
            self.status_bar.showMessage(f"Extraction failed: {exc}", 6000)
            return
        if count > 0:
            self.status_bar.showMessage(f"Extracted {count:,} points to {path}", 8000)
        else:
            QMessageBox.information(self, "Extract Tree Stand",
                                    "No points found for the current tree view / filter.")

    def confirm_segmentation_action(self):
        """Review and confirm ForestFormer3D tree split, staging or committing edits to the point cloud."""
        data = getattr(self, "_last_segmented_data", None)
        if not data or not data.get("ff3d_las"):
            QMessageBox.warning(self, "No Segmentation Data", "Please run 'Segment (ForestFormer3D)' on a tree first.")
            return

        itc = int(data.get("itc", 0))
        n_trees = int(data.get("ff3d_trees", 0))
        seg_las = str(data.get("ff3d_las", ""))
        max_itc = self.manager.get_max_itc()

        keep_ids = data.get("keep_tree_ids")
        is_target_included = True
        if keep_ids is not None:
            is_target_included = (itc in [int(x) for x in keep_ids])

        dlg = ConfirmSplitDialog(self, itc, n_trees, seg_las, max_itc, is_target_included=is_target_included)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        id_map = dlg.get_id_mapping()

        # Ensure local edits directory exists and copy canonical split file
        potree_dir = getattr(self, "current_potree_dir", None) or self.potree_dir
        base_dir = os.path.dirname(os.path.abspath(potree_dir)) if potree_dir else os.getcwd()
        edits_dir = os.path.join(base_dir, "edits", "trees")
        os.makedirs(edits_dir, exist_ok=True)
        import time
        if is_target_included:
            canonical_las = os.path.join(edits_dir, f"tree_{itc}_split.las")
        else:
            canonical_las = os.path.join(edits_dir, f"stand_near_{itc}_split_{int(time.time())}.las")

        if seg_las and os.path.exists(seg_las) and os.path.abspath(seg_las) != os.path.abspath(canonical_las):
            try:
                import shutil
                shutil.copy2(seg_las, canonical_las)
                seg_las = canonical_las
            except Exception as ce:
                print(f"[MainWindow] Error copying split LAS to edits dir: {ce}")

        # Stage the edit in PointCloudManager
        edit_label = f"FF3D Split Tree #{itc} -> {list(id_map.values())}" if is_target_included else f"FF3D New Trees -> {list(id_map.values())}"
        staged_edit = {
            "type": "ff3d_split",
            "action": "ff3d_split",
            "target_itc": itc if is_target_included else None,
            "segmented_las": seg_las,
            "id_mapping": id_map,
            "label": edit_label,
            "color": "#06b6d4",
            "is_target_included": is_target_included,
        }
        # 1. Instant Octree Commit (In-place patch)
        if dlg.action_choice == "octree":
            if not potree_dir or not os.path.isdir(potree_dir):
                QMessageBox.warning(self, "No Potree Octree Loaded",
                                    "No Potree converted octree folder is currently loaded to patch.\n"
                                    "Staged the edit for LAS commit instead.")
                self.manager.add_staged_edit(staged_edit)
                self.refresh_staged_edits_table()
                return

            self.status_bar.showMessage(f"Patching Potree 2.0 octree in place for {edit_label}...", 10000)
            self.log_console("OCTREE", f"Starting in-place commit to Potree octree for {edit_label}...", level="info")
            QApplication.processEvents()
            try:
                res = self.manager.patch_potree_octree(potree_dir, [staged_edit])
                pts_changed = res.get("changed_points", 0)

                # 1. Automatically log to SQLite database for history and undo/revert
                db_id = self.settings_store.log_edit(
                    edit_type="ff3d_split",
                    potree_dir=potree_dir,
                    las_path=self.manager.las_path,
                    target_itc=itc if is_target_included else None,
                    assigned_itcs=list(id_map.values()),
                    id_mapping=id_map,
                    segmented_las_path=seg_las,
                    points_affected=pts_changed,
                    details=staged_edit
                )

                # 2. Automatically stage the edit in PointCloudManager
                staged_edit["committed_octree"] = True
                staged_edit["db_edit_id"] = db_id
                staged_edit["action"] = "ff3d_split [Octree Patched]"
                staged_edit["color"] = "#0284c7"
                staged_edit["label"] = f"{edit_label} [Octree Patched]"
                self.manager.add_staged_edit(staged_edit)
                self.refresh_staged_edits_table()

                # 3. Register tree edits in database index and manager routing
                try:
                    import laspy
                    import numpy as np
                    s_las = laspy.read(seg_las)
                    extra_d = [d.name for d in s_las.point_format.dimensions]
                    if "ff3d_tree_id" in extra_d:
                        c_arr = np.asarray(s_las.ff3d_tree_id, dtype=np.int32)
                    elif "tree_id" in extra_d:
                        c_arr = np.asarray(s_las.tree_id, dtype=np.int32)
                    else:
                        c_arr = np.zeros(len(s_las), dtype=np.int32)

                    for c_idx, assigned_id in id_map.items():
                        c_int = int(c_idx)
                        a_int = int(assigned_id)
                        pt_cnt = int((c_arr == c_int).sum())
                        parent_val = itc if is_target_included else a_int
                        self.settings_store.register_tree_edit(a_int, parent_val, c_int, db_id, seg_las, pt_cnt)
                        self.manager.register_tree_edit(a_int, parent_val, c_int, db_id, seg_las, pt_cnt, id_map)
                except Exception as reg_err:
                    print(f"[MainWindow] Error registering tree edits: {reg_err}")

                # 4. Invalidate caches so next query fetches the new single tree
                if is_target_included:
                    self.manager.clear_tree_cache(itc)
                    self._run_js(f"if (window.invalidateTreeCache) window.invalidateTreeCache({itc});")
                for assigned_id in id_map.values():
                    self.manager.clear_tree_cache(int(assigned_id))
                    self._run_js(f"if (window.invalidateTreeCache) window.invalidateTreeCache({int(assigned_id)});")

                self.status_bar.showMessage(f"Patched Potree octree ({pts_changed:,} points) and logged to database!", 8000)
                self.log_console("OCTREE", f"✓ Patched octree.bin: {pts_changed:,} points modified across ({', '.join(res.get('touched_attributes', []))}).", level="success")
                self.log_console("DB", f"✓ Logged transaction #{db_id} to settings.db history.", level="success")
                self.log_console("VIEWPORT", f"✓ Reloaded 3D octree point cloud.", level="success")
                self._run_js("window.reloadCurrentPointCloud();")
                # Refresh the tree view
                if is_target_included:
                    self._run_js(f"window.viewTreeById({itc});")
                elif id_map:
                    first_id = list(id_map.values())[0]
                    self._run_js(f"window.viewTreeById({first_id});")

                if hasattr(self, "btn_confirm_segmentation"):
                    self.btn_confirm_segmentation.setEnabled(False)

                source_desc = f"• Target Tree: #{itc}\n" if is_target_included else "• Source: Stand Selection (Unassigned Points)\n"
                refresh_desc = f"The Forestry tree view has refreshed to show only Tree #{itc}." if is_target_included else f"The Forestry tree view has refreshed to show new Tree #{list(id_map.values())[0]}."

                QMessageBox.information(
                    self,
                    "Octree In-Place Commit Complete",
                    f"Successfully committed ForestFormer3D split to the active Potree Octree!\n\n"
                    f"{source_desc}"
                    f"• Assigned Tree IDs: {list(id_map.values())}\n"
                    f"• Octree Points Updated: {pts_changed:,}\n"
                    f"• Octree Folder: {os.path.basename(potree_dir)}\n"
                    f"• Database History ID: #{db_id}\n\n"
                    "The edit has been automatically staged and logged into settings.db.\n"
                    f"{refresh_desc}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Octree Patch Failed", f"Failed to patch Potree octree in place:\n\n{e}")
                self.status_bar.showMessage(f"Octree patch error: {e}", 6000)

        # 2. Commit to Both (Streaming LAS/LAZ + In-place Octree Patch)
        elif dlg.action_choice in ("both", "commit"):
            self.log_console("LAS", f"Committing staged edits to LAS/LAZ and updating octree...", level="info")
            self.manager.add_staged_edit(staged_edit)
            self.refresh_staged_edits_table()
            self.commit_staged_edits_action()

        # 3. Stage Edits Only
        else:
            self.manager.add_staged_edit(staged_edit)
            self.refresh_staged_edits_table()
            self.log_console("STAGE", f"✓ Staged {edit_label} for subsequent LAS commit.", level="success")
            # Register in-memory so Forestry view displays the single tree while staged
            try:
                import laspy
                import numpy as np
                s_las = laspy.read(seg_las)
                extra_d = [d.name for d in s_las.point_format.dimensions]
                if "ff3d_tree_id" in extra_d:
                    c_arr = np.asarray(s_las.ff3d_tree_id, dtype=np.int32)
                elif "tree_id" in extra_d:
                    c_arr = np.asarray(s_las.tree_id, dtype=np.int32)
                else:
                    c_arr = np.zeros(len(s_las), dtype=np.int32)

                for c_idx, assigned_id in id_map.items():
                    c_int = int(c_idx)
                    a_int = int(assigned_id)
                    pt_cnt = int((c_arr == c_int).sum())
                    parent_val = itc if is_target_included else a_int
                    self.manager.register_tree_edit(a_int, parent_val, c_int, 0, seg_las, pt_cnt, id_map)
            except Exception:
                pass
            if is_target_included:
                self.manager.clear_tree_cache(itc)
                self._run_js(f"if (window.invalidateTreeCache) window.invalidateTreeCache({itc});")
            for assigned_id in id_map.values():
                self.manager.clear_tree_cache(int(assigned_id))
                self._run_js(f"if (window.invalidateTreeCache) window.invalidateTreeCache({int(assigned_id)});")
            if is_target_included:
                self._run_js(f"window.viewTreeById({itc});")
            elif id_map:
                first_id = list(id_map.values())[0]
                self._run_js(f"window.viewTreeById({first_id});")

            if hasattr(self, "btn_confirm_segmentation"):
                self.btn_confirm_segmentation.setEnabled(False)
            self.status_bar.showMessage(f"Staged {edit_label}", 8000)
            title_stage = f"ForestFormer3D split for Tree #{itc}" if is_target_included else "ForestFormer3D segmentation for selected stand points"
            QMessageBox.information(
                self,
                "Edits Staged",
                f"{title_stage} has been staged successfully!\n\n"
                f"Assigned Tree IDs: {list(id_map.values())}\n\n"
                "You can see it in the 'Staged Trees & Edits' list on the Layers tab.\n"
                "When ready, click 'Commit & Save to LAS/LAZ...' (or Ctrl+Shift+S) to write the updated point cloud."
            )

    def on_tree_loaded(self, data: dict):
        itc = data.get("itc")
        t_cnt = data.get("target_count", 0)
        n_cnt = data.get("neighbor_count", 0)
        rad = data.get("radius", 0.0)
        is_indexing = getattr(self.manager, "_is_indexing", False)
        fallback_notice = " via streaming fallback (indexing in background)" if is_indexing else ""

        ff3d_trees = int(data.get("ff3d_trees", 0))
        ff3d_las = data.get("ff3d_las")
        is_segmented = bool(data.get("is_segmented")) or (ff3d_trees > 0)

        if is_segmented and ff3d_trees > 0:
            self._last_segmented_data = data
            if hasattr(self, "btn_confirm_segmentation"):
                self.btn_confirm_segmentation.setEnabled(True)
                self.btn_confirm_segmentation.setText(f"Confirm & Stage Split ({ff3d_trees} Trees)")
            if hasattr(self, "btn_tab_ai") and hasattr(self, "ribbon_stack"):
                self.btn_tab_ai.setChecked(True)
                self.ribbon_stack.setCurrentIndex(5)
            if hasattr(self, "combo_tree_color"):
                t_idx = self.combo_tree_color.findData("itc")
                if t_idx >= 0:
                    self.combo_tree_color.blockSignals(True)
                    self.combo_tree_color.setCurrentIndex(t_idx)
                    self.combo_tree_color.blockSignals(False)
            if hasattr(self, "lbl_tree_view_status"):
                self.lbl_tree_view_status.setText(
                    f"ForestFormer3D: {ff3d_trees} trees detected for Tree #{itc} · Click 'Confirm & Stage Split' to apply"
                )
        elif hasattr(self, "btn_confirm_segmentation") and not is_segmented:
            self.btn_confirm_segmentation.setEnabled(False)
            self.btn_confirm_segmentation.setText("Confirm & Stage Split Edits")

        # A full-density tree/stand is on screen: the Extract action can re-query it
        if itc is not None:
            self._tree_view_state = {
                "itc": int(itc), "radius": float(rad or 0.0),
                "mode": getattr(self, "_tree_view_state", {}).get("mode", "all"),
                "ground": float(getattr(self, "_tree_view_state", {}).get("ground", 0.0) or 0.0),
                "x": getattr(self, "_last_picked_x", None) if getattr(self, "_last_picked_itc", None) == int(itc) else None,
                "y": getattr(self, "_last_picked_y", None) if getattr(self, "_last_picked_itc", None) == int(itc) else None,
            }
        if hasattr(self, "btn_extract_tree"):
            self.btn_extract_tree.setEnabled(True)

        if hasattr(self, "lbl_tree_view_status"):
            if n_cnt > 0:
                self.lbl_tree_view_status.setText(
                    f"Tree #{itc}: {t_cnt:,} target pts + {n_cnt:,} neighbor pts ({rad:.0f}m stand){fallback_notice}"
                )
                self.status_bar.showMessage(
                    f"Viewing Tree #{itc} ({t_cnt:,} pts) + {n_cnt:,} neighbor pts within {rad:.0f}m stand{fallback_notice}.", 6000
                )
            else:
                self.lbl_tree_view_status.setText(
                    f"Tree #{itc}: {t_cnt:,} target points{fallback_notice}"
                )
                self.status_bar.showMessage(
                    f"Viewing Tree #{itc} ({t_cnt:,} full-density points){fallback_notice}.", 6000
                )
        if hasattr(self, "grp_stand_filter") and n_cnt <= 0:
            self.grp_stand_filter.setVisible(False)
            self.list_stand_trees.clear()
        if hasattr(self, "bar_stand_filter") and n_cnt <= 0:
            # Keep the Stand Filter buttons present on the Filter page, just not highlighted
            self.bar_stand_filter.setVisible(True)
            self._set_stand_filter_buttons_enabled(False)

    def _on_tree_stand_loaded(self, data: dict):
        target_itc = data.get("target_itc")
        n_cnt = data.get("neighbor_count", 0)
        t_cnt = data.get("target_count", 0)
        rad = data.get("radius", 0.0)
        trees = data.get("trees", [])

        if not hasattr(self, "grp_stand_filter"):
            return

        if (n_cnt <= 0 and len(trees) <= 1) or not trees:
            self.grp_stand_filter.setVisible(False)
            self.list_stand_trees.clear()
            if hasattr(self, "bar_stand_filter"):
                # Buttons stay visible on the Filter page, dimmed until neighbors exist
                self.bar_stand_filter.setVisible(True)
                self._set_stand_filter_buttons_enabled(False)
            return

        num_neighbor_trees = len([t for t in trees if not t.get("isTarget") and t.get("id", 0) > 0])
        stand_label = f"🌲 Stand: Tree #{target_itc} ({num_neighbor_trees} neighbors, {rad:.0f}m)" if n_cnt > 0 else f"🌲 Stand: {len(trees)} trees detected"
        if hasattr(self, "bar_stand_filter"):
            self.lbl_top_stand_info.setText(stand_label)
            self.bar_stand_filter.setVisible(True)
            self._set_stand_filter_buttons_enabled(True)

        self.grp_stand_filter.setVisible(True)
        info_label = f"Stand: {num_neighbor_trees} neighbor trees ({rad:.0f}m radius). Check trees to keep:" if n_cnt > 0 else f"Stand: {len(trees)} trees detected. Check trees to keep:"
        self.lbl_stand_filter_info.setText(info_label)

        self.list_stand_trees.blockSignals(True)
        self.list_stand_trees.clear()
        for t in trees:
            tid = t.get("id", 0)
            cnt = t.get("count", 0)
            is_target = t.get("isTarget", False)
            if is_target:
                label = f"★ Target Tree #{tid} ({cnt:,} pts)"
            elif tid > 0:
                label = f"Tree #{tid} ({cnt:,} pts)"
            else:
                label = f"Canopy / Ground ({cnt:,} pts)"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, tid)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.list_stand_trees.addItem(item)
        self.list_stand_trees.blockSignals(False)

        if hasattr(self, "btn_tree_filter_mode"):
            self.btn_tree_filter_mode.blockSignals(True)
            self.btn_tree_filter_mode.setChecked(False)
            self.btn_tree_filter_mode.setText("Filter Mode")
            self.btn_tree_filter_mode.blockSignals(False)

    def _on_tree_filter_mode_toggled(self, checked: bool):
        if hasattr(self, "btn_tree_filter_mode"):
            if checked:
                self.btn_tree_filter_mode.setText("Filter Mode ON")
                self.status_bar.showMessage("Filter mode active: Ctrl+Click or click neighbor trees in the 3D viewport to toggle selection.", 4000)
            else:
                self.btn_tree_filter_mode.setText("Filter Mode")
                self.status_bar.showMessage("Filter mode deactivated (Ctrl+Click picks single points).", 3000)
        self._run_js(f"if (window.setTreeFilterMode) window.setTreeFilterMode({'true' if checked else 'false'});")

    def _on_tree_filter_mode_toggled_from_hud(self, data: dict):
        mode = bool(data.get("filter_mode", False))
        if hasattr(self, "btn_tree_filter_mode"):
            self.btn_tree_filter_mode.blockSignals(True)
            self.btn_tree_filter_mode.setChecked(mode)
            self.btn_tree_filter_mode.setText("Filter Mode ON" if mode else "Filter Mode")
            self.btn_tree_filter_mode.blockSignals(False)

    def _on_tree_filter_selection_changed(self, data: dict):
        selected_ids = set(data.get("selected_ids", []))
        if hasattr(self, "list_stand_trees"):
            self.list_stand_trees.blockSignals(True)
            for i in range(self.list_stand_trees.count()):
                item = self.list_stand_trees.item(i)
                tid = item.data(Qt.ItemDataRole.UserRole)
                item.setCheckState(Qt.CheckState.Checked if tid in selected_ids else Qt.CheckState.Unchecked)
            self.list_stand_trees.blockSignals(False)
        t_id = data.get("toggled_id")
        is_sel = data.get("is_selected")
        if t_id is not None:
            lbl = "Canopy / Ground" if t_id == 0 else f"Tree #{t_id}"
            state_str = "selected" if is_sel else "deselected"
            self.status_bar.showMessage(f"{lbl} {state_str} for filter ({len(selected_ids)} total selected).", 3000)

    def _on_stand_tree_item_changed(self, item: QListWidgetItem):
        if not hasattr(self, "list_stand_trees"):
            return
        checked_ids = []
        for i in range(self.list_stand_trees.count()):
            it = self.list_stand_trees.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                checked_ids.append(it.data(Qt.ItemDataRole.UserRole))
        self._run_js(f"if (window.setTreeFilterSelection) window.setTreeFilterSelection({json.dumps(checked_ids)});")

    def _on_filter_select_all(self):
        if not hasattr(self, "list_stand_trees"):
            return
        self.list_stand_trees.blockSignals(True)
        all_ids = []
        for i in range(self.list_stand_trees.count()):
            it = self.list_stand_trees.item(i)
            it.setCheckState(Qt.CheckState.Checked)
            all_ids.append(it.data(Qt.ItemDataRole.UserRole))
        self.list_stand_trees.blockSignals(False)
        self._run_js(f"if (window.setTreeFilterSelection) window.setTreeFilterSelection({json.dumps(all_ids)});")
        self.status_bar.showMessage("All stand trees selected for filter.", 2500)

    def _on_filter_select_target_only(self):
        if not hasattr(self, "list_stand_trees"):
            return
        target_itc = int(self.spin_tree_view.value()) if hasattr(self, "spin_tree_view") else 1
        self.list_stand_trees.blockSignals(True)
        target_ids = []
        for i in range(self.list_stand_trees.count()):
            it = self.list_stand_trees.item(i)
            tid = it.data(Qt.ItemDataRole.UserRole)
            if tid == target_itc:
                it.setCheckState(Qt.CheckState.Checked)
                target_ids.append(tid)
            else:
                it.setCheckState(Qt.CheckState.Unchecked)
        self.list_stand_trees.blockSignals(False)
        self._run_js(f"if (window.setTreeFilterSelection) window.setTreeFilterSelection({json.dumps(target_ids)});")
        self.status_bar.showMessage(f"Only Target Tree #{target_itc} selected for filter.", 2500)

    def _apply_tree_filter(self):
        if not hasattr(self, "list_stand_trees"):
            return
        checked_ids = []
        for i in range(self.list_stand_trees.count()):
            it = self.list_stand_trees.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                checked_ids.append(it.data(Qt.ItemDataRole.UserRole))
        if not checked_ids:
            QMessageBox.information(self, "Stand Filter", "Please select at least one tree to keep.")
            return
        self._run_js(f"if (window.applyTreeFilter) window.applyTreeFilter({json.dumps(checked_ids)});")
        self.status_bar.showMessage(f"Applied stand filter: keeping {len(checked_ids)} tree(s).", 4000)

    def _reset_tree_filter(self):
        self._run_js("if (window.resetTreeFilter) window.resetTreeFilter();")
        if hasattr(self, "list_stand_trees"):
            self.list_stand_trees.blockSignals(True)
            for i in range(self.list_stand_trees.count()):
                self.list_stand_trees.item(i).setCheckState(Qt.CheckState.Checked)
            self.list_stand_trees.blockSignals(False)
        self.status_bar.showMessage("Stand filter reset — all stand trees restored.", 4000)

    def on_edit_applied(self, data: dict):
        count = data.get("modified_count", 0)
        action = data.get("action")
        self.lbl_save_status.setText("Status: <span style='color:#f59e0b;'>Unsaved Changes</span>")
        self.status_bar.showMessage(f"Applied '{action}' to {count:,} points.")

    def on_edit_staged(self, data: dict):
        self.refresh_staged_edits_table()

    def on_edit_removed(self, data: dict):
        self.refresh_staged_edits_table()

    def on_edits_cleared(self, data: dict):
        self.refresh_staged_edits_table()

    def apply_assign_itc(self):
        itc_val = self.spin_itc.value()

        sel = self.manager.current_selection
        if sel and sel.get("coords") and len(sel.get("coords")) >= 3:
            coords = sel.get("coords", [])
            points_3d = sel.get("points_3d", [])
            edit = {
                "type": "polygon",
                "coords": coords,
                "points_3d": points_3d,
                "action": "assign_itc",
                "value": itc_val,
                "label": f"Tree #{itc_val}",
                "color": "#10b981"
            }
            staged = self.manager.add_staged_edit(edit)
            self._run_js(f"window.addStagedVisualPolygon({json.dumps(staged)});")
            self._run_js("window.clearSelection(false);")
            self.spin_itc.setValue(itc_val + 1)
            self.refresh_staged_edits_table()
            self.status_bar.showMessage(f"Staged Tree #{itc_val} ({len(coords)} vertices).")
            return

        if self.manager.las_data and len(self.manager.selected_indices) > 0:
            count = self.manager.assign_itc(itc_val)
            self.on_edit_applied({"action": f"assign_itc({itc_val})", "modified_count": count})
            QMessageBox.information(self, "Success", f"Assigned Tree ID (ITC) {itc_val} to {count:,} points.")
            return

        QMessageBox.information(
            self,
            "No Active Polygon",
            "Please use the Polygon Select tool in the top bar to trace a tree crown first."
        )

    def apply_mark_noise(self):
        sel = self.manager.current_selection
        if sel and sel.get("coords") and len(sel.get("coords")) >= 3:
            coords = sel.get("coords", [])
            points_3d = sel.get("points_3d", [])
            edit = {
                "type": "polygon",
                "coords": coords,
                "points_3d": points_3d,
                "action": "mark_noise",
                "value": 9999,
                "label": "Noise (7)",
                "color": "#ef4444"
            }
            staged = self.manager.add_staged_edit(edit)
            self._run_js(f"window.addStagedVisualPolygon({json.dumps(staged)});")
            self._run_js("window.clearSelection(false);")
            self.refresh_staged_edits_table()
            self.status_bar.showMessage(f"Staged Noise region ({len(coords)} vertices).")
            return

        if self.manager.las_data and len(self.manager.selected_indices) > 0:
            count = self.manager.mark_as_noise(noise_itc=9999, class_val=7)
            self.on_edit_applied({"action": "mark_noise(7, 9999)", "modified_count": count})
            QMessageBox.information(self, "Success", f"Converted {count:,} points to Noise (Class 7).")
            return

        QMessageBox.information(
            self,
            "No Active Polygon",
            "Please use the Polygon Select tool in the top bar to trace a region first."
        )

    def apply_set_classification(self):
        class_val = self.combo_class.currentData()
        class_name = self.combo_class.currentText()

        sel = self.manager.current_selection
        if sel and sel.get("coords") and len(sel.get("coords")) >= 3:
            coords = sel.get("coords", [])
            points_3d = sel.get("points_3d", [])
            edit = {
                "type": "polygon",
                "coords": coords,
                "points_3d": points_3d,
                "action": "set_classification",
                "value": class_val,
                "label": f"Class {class_val}",
                "color": "#3b82f6"
            }
            staged = self.manager.add_staged_edit(edit)
            self._run_js(f"window.addStagedVisualPolygon({json.dumps(staged)});")
            self._run_js("window.clearSelection(false);")
            self.refresh_staged_edits_table()
            self.status_bar.showMessage(f"Staged Class {class_val} ({class_name}) for {len(coords)} vertices.")
            return

        if self.manager.las_data and len(self.manager.selected_indices) > 0:
            count = self.manager.set_classification(class_val)
            self.on_edit_applied({"action": f"set_class({class_val})", "modified_count": count})
            QMessageBox.information(self, "Success", f"Updated classification to {class_val} for {count:,} points.")
            return

        QMessageBox.information(
            self,
            "No Active Polygon",
            "Please use the Polygon Select tool in the top bar to trace a region first."
        )

    def apply_delete_selection(self):
        if len(self.manager.selected_indices) == 0:
            QMessageBox.information(self, "No Selection", "No points are currently selected in memory.")
            return

        reply = QMessageBox.question(
            self,
            "Delete Points",
            f"Are you sure you want to delete {len(self.manager.selected_indices):,} points from this cloud?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            count = self.manager.delete_selected_points()
            self.on_edit_applied({"action": "delete_points", "modified_count": count})
            self.lbl_sel_count.setText("Selection: None")
            self._run_js("window.clearSelection(false);")

    def clear_selection(self):
        self.manager.current_selection = None
        self._run_js("window.clearSelection(true);")
        self.lbl_sel_count.setText("Selection: None")
        self.lbl_sel_classes.setText("Classes: -")
        self.lbl_sel_itc.setText("ITC IDs: -")
        if hasattr(self, "lbl_layer_sel_info"):
            self.lbl_layer_sel_info.setText("Selection: None")

    # -------------------------------------------------------------------------
    # Staged Edits Table & Removal Actions
    # -------------------------------------------------------------------------
    def refresh_staged_edits_table(self):
        staged = self.manager.get_staged_edits()
        count = len(staged)

        self.lbl_staged_count.setText(f"Staged Edits: {count} (Ready to Commit)")
        if count > 0:
            self.lbl_save_status.setText(f"Status: <span style='color:#f59e0b;'>{count} Uncommitted Edits</span>")
        else:
            self.lbl_save_status.setText("Status: <span style='color:#10b981;'>Ready</span>")

        self.table_staged.setRowCount(count)
        for row, edit in enumerate(staged):
            edit_id = edit.get("id")
            label_text = edit.get("label", f"Edit #{row+1}")
            action_text = f"{edit.get('action')} ({edit.get('value')})" if "value" in edit else edit.get("action")
            coords_len = len(edit.get("coords", []))
            if edit.get("committed_octree"):
                type_text = "Octree Patched"
            elif edit.get("type") == "polygon":
                type_text = f"Poly ({coords_len} pts)"
            else:
                type_text = edit.get("type")

            item_lbl = QTableWidgetItem(label_text)
            item_act = QTableWidgetItem(action_text)
            item_type = QTableWidgetItem(type_text)

            if edit.get("committed_octree"):
                item_lbl.setForeground(QColor("#38bdf8"))
                item_act.setForeground(QColor("#38bdf8"))
                item_type.setForeground(QColor("#10b981"))

            self.table_staged.setItem(row, 0, item_lbl)
            self.table_staged.setItem(row, 1, item_act)
            self.table_staged.setItem(row, 2, item_type)

            btn_del = QPushButton("X")
            btn_del.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: none;
                    color: #ef4444;
                    font-weight: bold;
                    font-size: 11px;
                }
                QPushButton:hover {
                    color: #f87171;
                    background-color: rgba(239, 68, 68, 0.2);
                    border-radius: 3px;
                }
            """)
            btn_del.clicked.connect(lambda checked, eid=edit_id: self.remove_single_staged_edit(eid))
            self.table_staged.setCellWidget(row, 3, btn_del)

        # 2. Update Layer Manager Tab's Staged Overlays Table & Counter
        if hasattr(self, "lbl_layer_staged_count"):
            self.lbl_layer_staged_count.setText(f"{count} staged edits (ready to commit)")

        if hasattr(self, "table_layer_staged"):
            self.table_layer_staged.setRowCount(count)
            for row, edit in enumerate(staged):
                edit_id = edit.get("id")
                label_text = edit.get("label", f"Edit #{row+1}")
                action_text = f"{edit.get('action')} ({edit.get('value')})" if "value" in edit else edit.get("action")
                is_visible = edit.get("visible", True)
                color_hex = edit.get("color", "#10b981")

                # Column 0: Visibility Checkbox
                chk_container = QWidget()
                chk_lay = QHBoxLayout(chk_container)
                chk_lay.setContentsMargins(0, 0, 0, 0)
                chk_lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
                chk = QCheckBox()
                chk.setChecked(is_visible)
                chk.toggled.connect(lambda state, eid=edit_id: self.toggle_single_staged_visibility(eid, state))
                chk_lay.addWidget(chk)
                self.table_layer_staged.setCellWidget(row, 0, chk_container)

                # Column 1: Color Swatch + Label
                pix = QPixmap(10, 10)
                pix.fill(QColor(color_hex))
                item_label = QTableWidgetItem(QIcon(pix), label_text)
                if edit.get("committed_octree"):
                    item_label.setForeground(QColor("#38bdf8"))
                self.table_layer_staged.setItem(row, 1, item_label)

                # Column 2: Action
                item_act = QTableWidgetItem(action_text)
                self.table_layer_staged.setItem(row, 2, item_act)

                # Column 3: Delete Button
                btn_del_lm = QPushButton("X")
                btn_del_lm.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;
                        border: none;
                        color: #ef4444;
                        font-weight: bold;
                        font-size: 11px;
                    }
                    QPushButton:hover {
                        color: #f87171;
                        background-color: rgba(239, 68, 68, 0.2);
                        border-radius: 3px;
                    }
                """)
                btn_del_lm.clicked.connect(lambda checked, eid=edit_id: self.remove_single_staged_edit(eid))
                self.table_layer_staged.setCellWidget(row, 3, btn_del_lm)

    def remove_single_staged_edit(self, edit_id: str):
        edit = next((e for e in self.manager.staged_edits if e.get("id") == edit_id), None)
        if edit and edit.get("committed_octree"):
            potree_dir = getattr(self, "current_potree_dir", None)
            reply = QMessageBox.question(
                self,
                "Revert Octree Edit?",
                f"This edit was already committed to the active Potree Octree in-place:\n\n"
                f"• {edit.get('label', edit_id)}\n\n"
                "Would you like to revert this edit in the Potree Octree now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.Yes:
                if potree_dir and os.path.isdir(potree_dir):
                    try:
                        self.manager.revert_potree_octree(potree_dir, edit)
                        db_id = edit.get("db_edit_id")
                        target_itc = edit.get("target_itc")
                        if db_id:
                            self.settings_store.mark_edit_reverted(db_id)
                        if target_itc:
                            self.settings_store.revert_tree_edits_by_parent_itc(target_itc)
                            self.manager.clear_tree_cache(target_itc)
                            self._run_js(f"if (window.invalidateTreeCache) window.invalidateTreeCache({target_itc});")
                            for a_id in edit.get("id_mapping", {}).values():
                                self.manager.clear_tree_cache(int(a_id))
                                self._run_js(f"if (window.invalidateTreeCache) window.invalidateTreeCache({int(a_id)});")
                            self.manager.sync_tree_edits_from_db(self.settings_store)
                            self._run_js(f"window.viewTreeById({target_itc});")
                        self._run_js("window.reloadCurrentPointCloud();")
                        self.status_bar.showMessage(f"Reverted octree edit for Tree #{target_itc}", 6000)
                    except Exception as e:
                        QMessageBox.critical(self, "Revert Failed", f"Failed to revert octree edit:\n\n{e}")
                        return

        self.manager.remove_staged_edit(edit_id)
        self._run_js(f"window.removeStagedVisualPolygon('{edit_id}');")
        self.refresh_staged_edits_table()
        self.status_bar.showMessage(f"Removed staged edit: {edit_id}")

    def clear_all_staged_edits(self):
        self.manager.clear_staged_edits()
        self._run_js("window.clearAllStagedVisuals();")
        self.refresh_staged_edits_table()
        self.status_bar.showMessage("Cleared all staged edits.")

    # -------------------------------------------------------------------------
    # High-Performance Deferred Commit Operation
    # -------------------------------------------------------------------------
    def commit_staged_edits_action(self):
        """Execute high-performance deferred commit in a background thread."""
        if not self.manager.staged_edits:
            QMessageBox.information(
                self,
                "No Staged Edits",
                "There are no staged edits to commit.\n\n"
                "Trace tree crowns or noise regions with the polygon tool, "
                "then click 'Stage Tree ID' or 'Stage Noise'."
            )
            return

        if not self.current_las_path or not os.path.exists(self.current_las_path):
            file_path = self._pick_file(
                "Select Source LAS/LAZ File to Commit Edits Into",
                "",
                "LAS/LAZ Files (*.las *.laz);;All Files (*)"
            )
            if not file_path:
                return
            self.current_las_path = file_path

        base, ext = os.path.splitext(self.current_las_path)
        default_out = f"{base}_edited{ext}"
        save_path = self._save_file(
            "Save Committed Point Cloud",
            default_out,
            "LAZ Files (*.laz);;LAS Files (*.las);;All Files (*)"
        )
        if not save_path:
            return

        total_pts = self.manager.header_info.get("total_points", 100) if self.manager.header_info else 100
        self.commit_dialog = QProgressDialog("Committing staged edits...", "Cancel", 0, total_pts, self)
        self.commit_dialog.setWindowTitle("Committing Point Cloud Edits")
        self.commit_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.commit_dialog.setAutoClose(True)
        self.commit_dialog.setAutoReset(False)
        self.commit_dialog.setValue(0)
        self.commit_dialog.show()

        self.commit_worker = CommitWorker(self.manager, self.current_las_path, save_path)
        self.commit_worker.progress.connect(self._on_commit_progress)
        self.commit_worker.finished.connect(self._on_commit_finished)
        self.commit_worker.error.connect(self._on_commit_error)
        self.commit_worker.start()

    def _on_commit_progress(self, processed: int, total: int, msg: str):
        if self.commit_dialog:
            self.commit_dialog.setValue(processed)
            self.commit_dialog.setLabelText(msg)
        self.status_bar.showMessage(msg)

    def _on_commit_finished(self, result: dict):
        if self.commit_dialog:
            self.commit_dialog.close()

        count = result.get("edits_count", 0)
        pts = result.get("total_points", 0)
        mod_pts = result.get("modified_points", 0)
        sec = result.get("elapsed_seconds", 0.0)
        out = result.get("output_path", "")

        msg = (
            f"Successfully committed {count} staged edits across {pts:,} points in {sec:.1f}s!\n\n"
            f"Modified points: {mod_pts:,}\n"
            f"Saved output: {out}\n\n"
            f"Would you like to keep the 3D visual outlines visible in the viewport?"
        )
        reply = QMessageBox.question(
            self,
            "Commit Complete",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.No:
            self.clear_all_staged_edits()
        else:
            self.refresh_staged_edits_table()

        # Auto-patch the active Potree 2.0 octree so it stays synchronized with the committed LAS file!
        potree_dir = getattr(self, "current_potree_dir", None)
        committed_edits = result.get("committed_edits", [])
        if potree_dir and os.path.isdir(potree_dir) and committed_edits:
            try:
                self.status_bar.showMessage("Syncing committed edits to active Potree 2.0 octree in place...", 6000)
                oct_res = self.manager.patch_potree_octree(potree_dir, committed_edits)
                print(f"[MainWindow] Auto-patched Potree octree {potree_dir}: {oct_res}")
                # Ensure committed edits are logged to database
                for e in committed_edits:
                    if not e.get("db_edit_id"):
                        try:
                            db_id = self.settings_store.log_edit(
                                edit_type=e.get("type", "edit"),
                                potree_dir=potree_dir,
                                las_path=out,
                                target_itc=e.get("target_itc"),
                                assigned_itcs=list(e.get("id_mapping", {}).values()) if e.get("id_mapping") else None,
                                id_mapping=e.get("id_mapping"),
                                segmented_las_path=e.get("segmented_las"),
                                points_affected=oct_res.get("changed_points", 0),
                                details=e
                            )
                            e["db_edit_id"] = db_id
                        except Exception as log_err:
                            print(f"[MainWindow] Error logging edit: {log_err}")

                    if e.get("type") == "ff3d_split" and e.get("segmented_las"):
                        try:
                            import laspy
                            import numpy as np
                            s_las = laspy.read(e["segmented_las"])
                            extra_d = [d.name for d in s_las.point_format.dimensions]
                            c_arr = np.asarray(s_las.ff3d_tree_id if "ff3d_tree_id" in extra_d else (s_las.tree_id if "tree_id" in extra_d else np.zeros(len(s_las))), dtype=np.int32)
                            is_inc = e.get("is_target_included", e.get("target_itc") is not None)
                            raw_target = e.get("target_itc")
                            p_itc = int(raw_target) if (raw_target is not None and is_inc) else 0
                            id_map = e.get("id_mapping", {})
                            for c_idx, assigned_id in id_map.items():
                                c_int = int(c_idx)
                                a_int = int(assigned_id)
                                pt_cnt = int((c_arr == c_int).sum())
                                parent_val = p_itc if is_inc else a_int
                                self.settings_store.register_tree_edit(a_int, parent_val, c_int, e.get("db_edit_id", 0), e["segmented_las"], pt_cnt)
                                self.manager.register_tree_edit(a_int, parent_val, c_int, e.get("db_edit_id", 0), e["segmented_las"], pt_cnt, id_map)
                        except Exception as tree_err:
                            print(f"[MainWindow] Error registering tree edit on commit: {tree_err}")
                self._run_js("window.reloadCurrentPointCloud();")
            except Exception as pe:
                print(f"[MainWindow] Warning: Failed to patch Potree octree after LAS commit: {pe}")

        self.lbl_save_status.setText(f"Status: <span style='color:#10b981;'>Committed to {os.path.basename(out)}</span>")
        self.status_bar.showMessage(f"Committed {count} edits in {sec:.1f}s. Saved: {os.path.basename(out)}")

    def _on_commit_error(self, err_msg: str):
        if self.commit_dialog:
            self.commit_dialog.close()
        QMessageBox.critical(self, "Commit Failed", f"An error occurred while committing staged edits:\n\n{err_msg}")
        self.status_bar.showMessage(f"Commit error: {err_msg}")

    def save_modified_las(self):
        """In-memory save fallback."""
        if not self.manager.las_data:
            self.commit_staged_edits_action()
            return

        default_name = os.path.splitext(self.current_las_path or "edited_cloud.las")[0] + "_edited.las"
        save_path = self._save_file(
            "Save Modified Point Cloud",
            default_name,
            "LAS Files (*.las);;LAZ Files (*.laz)"
        )
        if not save_path:
            return

        try:
            self.manager.save_las(save_path)
            self.lbl_save_status.setText("Status: <span style='color:#10b981;'>All changes saved</span>")
            QMessageBox.information(self, "Saved", f"Point cloud successfully saved to:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    # -------------------------------------------------------------------------
    # Render & Visual Settings Actions
    # -------------------------------------------------------------------------
    # Viewer JS Bridge (queued until the page is live)
    # -------------------------------------------------------------------------
    def _on_page_load_finished(self, ok: bool):
        self._page_ready = bool(ok)
        if ok and self._js_queue:
            for script, callback in self._js_queue:
                if callback is None:
                    self.web_view.page().runJavaScript(script)
                else:
                    self.web_view.page().runJavaScript(script, callback)
            self._js_queue = []
        if ok and not self._settings_applied:
            self._settings_applied = True
            self._apply_loaded_settings()

    # -------------------------------------------------------------------------
    # Settings Persistence (SQLite)
    # -------------------------------------------------------------------------
    def _collect_settings(self) -> dict:
        """Snapshot every persisted viewer setting from the ribbon widgets."""
        return {
            "render_mode": self.ribbon_combo_render_mode.currentData(),
            "shape": self.ribbon_combo_shape.currentData(),
            "sizing": self.ribbon_combo_sizing.currentData(),
            "point_size": self.ribbon_spin_size.value(),
            "splat": self.ribbon_combo_splat.currentData(),
            "budget": self.ribbon_combo_budget.currentData(),
            "node": self.ribbon_combo_node.currentData(),
            "color": self.ribbon_combo_color.currentData(),
            "bg": self.ribbon_combo_bg.currentData(),
            "edl_enabled": self.ribbon_chk_edl.isChecked(),
            "edl_radius": self.ribbon_spin_radius.value(),
            "edl_strength": self.ribbon_spin_strength.value(),
            "edl_opacity": self.ribbon_spin_opacity.value(),
        }

    def _schedule_save_settings(self, *_):
        self._settings_timer.start()

    def _save_settings(self):
        try:
            self.settings_store.set_many(self._collect_settings())
            self.settings_store.set("geometry",
                                    bytes(self.saveGeometry().toHex().data()).decode())
            self.settings_store.set("splitter", self.splitter.sizes())
        except Exception as e:  # persistence must never break editing
            print(f"[Settings] save failed: {e}")

    def _apply_loaded_settings(self):
        """Restore the persisted settings after the page is live.

        Order matters: the render-mode preset writes the shape/size/budget
        defaults in JS, so it goes first and the individual settings win on top.
        """
        s = self.settings_store

        def set_combo(widget, key):
            val = s.get(key)
            if val is None:
                return
            idx = widget.findData(val)
            if idx >= 0:
                widget.blockSignals(True)
                widget.setCurrentIndex(idx)
                widget.blockSignals(False)

        def set_spin(widget, key):
            val = s.get(key)
            if val is None:
                return
            widget.blockSignals(True)
            widget.setValue(float(val))
            widget.blockSignals(False)

        if s.get("render_mode") is not None:
            set_combo(self.ribbon_combo_render_mode, "render_mode")
            self._on_ribbon_render_mode_changed()
        if s.get("shape") is not None:
            set_combo(self.ribbon_combo_shape, "shape")
            self._on_ribbon_shape_changed()
        if s.get("sizing") is not None:
            set_combo(self.ribbon_combo_sizing, "sizing")
            self._on_ribbon_sizing_changed()
        if s.get("point_size") is not None:
            set_spin(self.ribbon_spin_size, "point_size")
            self._on_ribbon_size_spin_changed(self.ribbon_spin_size.value())
        if s.get("splat") is not None:
            set_combo(self.ribbon_combo_splat, "splat")
            self._on_ribbon_splat_changed()
        if s.get("budget") is not None:
            set_combo(self.ribbon_combo_budget, "budget")
            self._on_ribbon_budget_changed()
        if s.get("node") is not None:
            set_combo(self.ribbon_combo_node, "node")
            self.set_min_node_size(self.ribbon_combo_node.currentData())
        if s.get("color") is not None:
            set_combo(self.ribbon_combo_color, "color")
            self._on_ribbon_color_changed()
        if s.get("bg") is not None:
            set_combo(self.ribbon_combo_bg, "bg")
            self._on_ribbon_bg_changed()
        edl = s.get("edl_enabled")
        if edl is not None:
            self.ribbon_chk_edl.blockSignals(True)
            self.ribbon_chk_edl.setChecked(bool(edl))
            self.ribbon_chk_edl.blockSignals(False)
            self.set_edl_enabled(bool(edl))
        for widget, key, setter in (
                (self.ribbon_spin_radius, "edl_radius", self.set_edl_radius),
                (self.ribbon_spin_strength, "edl_strength", self.set_edl_strength),
                (self.ribbon_spin_opacity, "edl_opacity", self.set_edl_opacity)):
            if s.get(key) is not None:
                set_spin(widget, key)
                setter(widget.value())

        geometry = s.get("geometry")
        if geometry:
            self.restoreGeometry(QByteArray.fromHex(geometry.encode()))
        sizes = s.get("splitter")
        if isinstance(sizes, list) and sizes:
            self.splitter.setSizes(sizes)

    def _wire_settings_persistence(self):
        """Every ribbon setting change schedules a debounced save."""
        for combo in (self.ribbon_combo_shape, self.ribbon_combo_sizing,
                      self.ribbon_combo_splat, self.ribbon_combo_budget,
                      self.ribbon_combo_node, self.ribbon_combo_color,
                      self.ribbon_combo_bg, self.ribbon_combo_render_mode):
            combo.currentIndexChanged.connect(self._schedule_save_settings)
        for spin in (self.ribbon_spin_size, self.ribbon_spin_radius,
                     self.ribbon_spin_strength, self.ribbon_spin_opacity):
            spin.valueChanged.connect(self._schedule_save_settings)
        self.ribbon_slider_size.valueChanged.connect(self._schedule_save_settings)
        self.ribbon_chk_edl.toggled.connect(self._schedule_save_settings)

    def closeEvent(self, event):
        # Final flush so a change made <450 ms before closing is not lost.
        self._save_settings()
        self.settings_store.close()
        super().closeEvent(event)

    def _run_js(self, script: str, callback=None):
        """Run JS in the viewer page, queueing it while the page is still loading.

        web/app.js is a type=module script, so its window.* API surface only
        exists after loadFinished; calling the setters — and the reader calls
        that pass callbacks — earlier silently drops the requested value.
        """
        if self._page_ready:
            if callback is None:
                self.web_view.page().runJavaScript(script)
            else:
                self.web_view.page().runJavaScript(script, callback)
        else:
            self._js_queue.append((script, callback))

    def set_point_shape(self, shape: str):
        self._run_js(f"window.setPointShape('{shape}');")

    def set_point_size_type(self, stype: str):
        self._run_js(f"window.setPointSizeType('{stype}');")

    def set_point_size(self, size: float):
        self._run_js(f"window.setPointSize({size:.2f});")

    def set_splat_quality(self, quality: str):
        self._run_js(f"window.setSplatQuality('{quality}');")

    def set_edl_enabled(self, enabled: bool):
        val = "true" if enabled else "false"
        self._run_js(f"window.setEDLEnabled({val});")

    def set_edl_radius(self, radius: float):
        self._run_js(f"window.setEDLRadius({radius:.1f});")

    def set_edl_strength(self, strength: float):
        self._run_js(f"window.setEDLStrength({strength:.1f});")

    def set_point_budget(self, budget: int):
        self._run_js(f"window.setPointBudget({budget});")

    def set_color_mode(self, mode: str):
        self._run_js(f"window.setColorAttribute('{mode}');")

    def toggle_in_viewport_settings(self):
        self._run_js("window.toggleSettingsMenu();")

    def set_edl_opacity(self, opacity: float):
        self._run_js(f"window.setEDLOpacity({opacity:.2f});")

    def set_min_node_size(self, size: int):
        self._run_js(f"window.setMinNodeSize({size});")

    def set_background(self, bg: str):
        self._run_js(f"window.setBackground('{bg}');")

    # -------------------------------------------------------------------------
    # Top Tabbed Ribbon Bar
    # -------------------------------------------------------------------------
    def create_ribbon_bar(self) -> QWidget:
        ribbon_container = QWidget()
        ribbon_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        ribbon_container.setFixedHeight(134)
        v_main = QVBoxLayout(ribbon_container)
        v_main.setContentsMargins(0, 6, 0, 6)
        v_main.setSpacing(0)

        # Top Row: Category Tabs (File, Point, Settings)
        category_bar = QFrame()
        category_bar.setObjectName("ribbonCategoryBar")
        category_bar.setFixedHeight(30)
        h_tabs = QHBoxLayout(category_bar)
        h_tabs.setContentsMargins(8, 0, 8, 0)
        h_tabs.setSpacing(2)

        self.ribbon_tab_group = QButtonGroup(self)
        self.ribbon_tab_group.setExclusive(True)

        self.btn_tab_file = QPushButton("File")
        self.btn_tab_file.setCheckable(True)
        self.btn_tab_file.setProperty("class", "ribbonCategoryTab")

        self.btn_tab_point = QPushButton("Point")
        self.btn_tab_point.setCheckable(True)
        self.btn_tab_point.setChecked(True)
        self.btn_tab_point.setProperty("class", "ribbonCategoryTab")

        self.btn_tab_layers = QPushButton("Layers")
        self.btn_tab_layers.setCheckable(True)
        self.btn_tab_layers.setProperty("class", "ribbonCategoryTab")

        self.btn_tab_settings = QPushButton("Settings")
        self.btn_tab_settings.setCheckable(True)
        self.btn_tab_settings.setProperty("class", "ribbonCategoryTab")

        # Stand Filter category (active only while the Forestry sidebar tab is open)
        self.btn_tab_filter = QPushButton("Filter")
        self.btn_tab_filter.setCheckable(True)
        self.btn_tab_filter.setProperty("class", "ribbonCategoryTab")
        self.btn_tab_filter.setVisible(False)  # Shown only while the Forestry sidebar tab is open

        # AI category (active only while the Forestry sidebar tab is open, like Filter)
        self.btn_tab_ai = QPushButton("AI")
        self.btn_tab_ai.setCheckable(True)
        self.btn_tab_ai.setProperty("class", "ribbonCategoryTab")
        self.btn_tab_ai.setVisible(False)  # Shown only while the Forestry sidebar tab is open

        self.ribbon_tab_group.addButton(self.btn_tab_file, 0)
        self.ribbon_tab_group.addButton(self.btn_tab_point, 1)
        self.ribbon_tab_group.addButton(self.btn_tab_layers, 2)
        self.ribbon_tab_group.addButton(self.btn_tab_settings, 3)
        self.ribbon_tab_group.addButton(self.btn_tab_filter, 4)
        self.ribbon_tab_group.addButton(self.btn_tab_ai, 5)

        h_tabs.addWidget(self.btn_tab_file)
        h_tabs.addWidget(self.btn_tab_point)
        h_tabs.addWidget(self.btn_tab_layers)
        h_tabs.addWidget(self.btn_tab_settings)
        h_tabs.addWidget(self.btn_tab_filter)
        h_tabs.addWidget(self.btn_tab_ai)
        h_tabs.addStretch()

        v_main.addWidget(category_bar)

        # Bottom Row: QStackedWidget switching pages based on active tab
        self.ribbon_stack = QStackedWidget()
        self.ribbon_stack.setObjectName("ribbonStack")
        self.ribbon_stack.setFixedHeight(92)

        page_file = self._create_ribbon_file_page()
        page_point = self._create_ribbon_point_page()
        page_layers = self._create_ribbon_layers_page()
        page_settings = self._create_ribbon_settings_page()
        page_filter = self._create_ribbon_filter_page()
        page_ai = self._create_ribbon_ai_page()

        self.ribbon_stack.addWidget(page_file)       # Index 0
        self.ribbon_stack.addWidget(page_point)      # Index 1
        self.ribbon_stack.addWidget(page_layers)     # Index 2
        self.ribbon_stack.addWidget(page_settings)   # Index 3
        self.ribbon_stack.addWidget(page_filter)     # Index 4 (Stand Filter)
        self.ribbon_stack.addWidget(page_ai)         # Index 5 (AI Segmentation Toolset)

        self.ribbon_tab_group.idClicked.connect(self._on_ribbon_category_clicked)
        self.ribbon_stack.setCurrentIndex(1)

        v_main.addWidget(self.ribbon_stack)
        return ribbon_container

    # Category 1: File Page
    def _create_ribbon_file_page(self) -> QWidget:
        page = QFrame()
        page.setFixedHeight(92)
        h_layout = QHBoxLayout(page)
        h_layout.setContentsMargins(8, 6, 8, 6)
        h_layout.setSpacing(8)

        # Open Group
        grp_open = QFrame()
        grp_open.setProperty("class", "ribbonGroup")
        v_open = QVBoxLayout(grp_open)
        v_open.setContentsMargins(10, 7, 10, 7)
        v_open.setSpacing(3)

        h_open_btns = QHBoxLayout()
        h_open_btns.setSpacing(4)
        btn_open_potree = QPushButton("Open Potree...")
        btn_open_potree.setProperty("class", "ribbonToolBtn")
        btn_open_potree.clicked.connect(self.open_potree_file)

        btn_open_las = QPushButton("Open LAS/LAZ...")
        btn_open_las.setProperty("class", "ribbonToolBtn")
        btn_open_las.clicked.connect(self.open_las_file)

        btn_convert_las = QPushButton("Convert LAS...")
        btn_convert_las.setProperty("class", "ribbonToolBtn")
        btn_convert_las.setToolTip("Convert a LAS/LAZ file into Potree octree with real-time logs and stats")
        btn_convert_las.clicked.connect(self.convert_las_action)

        btn_open_folder = QPushButton("Open Folder...")
        btn_open_folder.setProperty("class", "ribbonToolBtn")
        btn_open_folder.clicked.connect(self.open_folder_dialog)

        h_open_btns.addWidget(btn_open_potree)
        h_open_btns.addWidget(btn_open_las)
        h_open_btns.addWidget(btn_convert_las)
        h_open_btns.addWidget(btn_open_folder)
        v_open.addLayout(h_open_btns)

        lbl_open = QLabel("OPEN")
        lbl_open.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_open.setProperty("class", "ribbonTitle")
        v_open.addWidget(lbl_open)
        h_layout.addWidget(grp_open)

        # Save & Commit Group
        grp_save = QFrame()
        grp_save.setProperty("class", "ribbonGroup")
        v_save = QVBoxLayout(grp_save)
        v_save.setContentsMargins(10, 7, 10, 7)
        v_save.setSpacing(3)

        h_save_btns = QHBoxLayout()
        h_save_btns.setSpacing(4)

        btn_commit = QPushButton("Commit & Save to LAS/LAZ...")
        btn_commit.setProperty("class", "ribbonToolBtn")
        btn_commit.setObjectName("btnSuccess")
        btn_commit.clicked.connect(self.commit_staged_edits_action)

        btn_save = QPushButton("Save (RAM)...")
        btn_save.setProperty("class", "ribbonToolBtn")
        btn_save.clicked.connect(self.save_modified_las)

        h_save_btns.addWidget(btn_commit)
        h_save_btns.addWidget(btn_save)
        v_save.addLayout(h_save_btns)

        lbl_save = QLabel("SAVE & COMMIT")
        lbl_save.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_save.setProperty("class", "ribbonTitle")
        v_save.addWidget(lbl_save)
        h_layout.addWidget(grp_save)

        h_layout.addStretch()
        return page

    # Category 2: Point Page
    def _create_ribbon_point_page(self) -> QWidget:
        page = QFrame()
        page.setFixedHeight(92)
        h_layout = QHBoxLayout(page)
        h_layout.setContentsMargins(8, 6, 8, 6)
        h_layout.setSpacing(8)

        # 1. Point Settings Group
        grp_point = QFrame()
        grp_point.setProperty("class", "ribbonGroup")
        v_pt = QVBoxLayout(grp_point)
        v_pt.setContentsMargins(10, 7, 10, 7)
        v_pt.setSpacing(3)

        h_pt_controls = QHBoxLayout()
        h_pt_controls.setSpacing(8)

        # Shape
        v_shape = QVBoxLayout()
        v_shape.setSpacing(1)
        lbl_shape = QLabel("Shape:")
        lbl_shape.setProperty("class", "ribbonLabel")
        self.ribbon_combo_shape = QComboBox()
        self.ribbon_combo_shape.setProperty("class", "ribbonCombo")
        self.ribbon_combo_shape.addItem("Square", "SQUARE")
        self.ribbon_combo_shape.addItem("Circle", "CIRCLE")
        self.ribbon_combo_shape.addItem("Paraboloid (Splat)", "PARABOLOID")
        self.ribbon_combo_shape.currentIndexChanged.connect(self._on_ribbon_shape_changed)
        v_shape.addWidget(lbl_shape)
        v_shape.addWidget(self.ribbon_combo_shape)
        h_pt_controls.addLayout(v_shape)

        # Sizing
        v_sizing = QVBoxLayout()
        v_sizing.setSpacing(1)
        lbl_sizing = QLabel("Sizing:")
        lbl_sizing.setProperty("class", "ribbonLabel")
        self.ribbon_combo_sizing = QComboBox()
        self.ribbon_combo_sizing.setProperty("class", "ribbonCombo")
        self.ribbon_combo_sizing.addItem("Adaptive", "ADAPTIVE")
        self.ribbon_combo_sizing.addItem("Fixed", "FIXED")
        self.ribbon_combo_sizing.addItem("Attenuated", "ATTENUATED")
        self.ribbon_combo_sizing.currentIndexChanged.connect(self._on_ribbon_sizing_changed)
        v_sizing.addWidget(lbl_sizing)
        v_sizing.addWidget(self.ribbon_combo_sizing)
        h_pt_controls.addLayout(v_sizing)

        # Size Slider + SpinBox
        v_size = QVBoxLayout()
        v_size.setSpacing(1)
        lbl_size = QLabel("Size:")
        lbl_size.setProperty("class", "ribbonLabel")

        h_size = QHBoxLayout()
        h_size.setSpacing(4)
        self.ribbon_slider_size = QSlider(Qt.Orientation.Horizontal)
        self.ribbon_slider_size.setRange(1, 40)
        self.ribbon_slider_size.setValue(10)
        self.ribbon_slider_size.setFixedWidth(80)
        self.ribbon_slider_size.valueChanged.connect(self._on_ribbon_size_slider_changed)

        self.ribbon_spin_size = QDoubleSpinBox()
        self.ribbon_spin_size.setProperty("class", "ribbonSpin")
        self.ribbon_spin_size.setRange(0.10, 4.00)
        self.ribbon_spin_size.setSingleStep(0.05)
        self.ribbon_spin_size.setValue(1.00)
        self.ribbon_spin_size.setFixedWidth(60)
        self.ribbon_spin_size.valueChanged.connect(self._on_ribbon_size_spin_changed)

        h_size.addWidget(self.ribbon_slider_size)
        h_size.addWidget(self.ribbon_spin_size)

        v_size.addWidget(lbl_size)
        v_size.addLayout(h_size)
        h_pt_controls.addLayout(v_size)

        v_pt.addLayout(h_pt_controls)
        lbl_title_pt = QLabel("POINT SETTINGS")
        lbl_title_pt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title_pt.setProperty("class", "ribbonTitle")
        v_pt.addWidget(lbl_title_pt)
        h_layout.addWidget(grp_point)

        # 1b. Color By -- the chooser that used to live only in the sidebar's Point
        # Cloud Layer card (and, before this, on the Settings page of the ribbon).
        grp_color = QFrame()
        grp_color.setProperty("class", "ribbonGroup")
        v_color = QVBoxLayout(grp_color)
        v_color.setContentsMargins(10, 7, 10, 7)
        v_color.setSpacing(3)
        h_color_controls = QHBoxLayout()
        h_color_controls.setSpacing(8)
        v_mode = QVBoxLayout()
        v_mode.setSpacing(1)
        lbl_mode = QLabel("Color By:")
        lbl_mode.setProperty("class", "ribbonLabel")
        self.ribbon_combo_color = QComboBox()
        self.ribbon_combo_color.setProperty("class", "ribbonCombo")
        self.ribbon_combo_color.addItem("RGB / True Color", "rgba")
        self.ribbon_combo_color.addItem("ASPRS Classification", "classification")
        self.ribbon_combo_color.addItem("Elevation", "elevation")
        self.ribbon_combo_color.addItem("Intensity", "intensity")
        self.ribbon_combo_color.addItem("Tree / ITC", "itc")
        self.ribbon_combo_color.currentIndexChanged.connect(self._on_ribbon_color_changed)
        v_mode.addWidget(lbl_mode)
        v_mode.addWidget(self.ribbon_combo_color)
        h_color_controls.addLayout(v_mode)
        v_color.addLayout(h_color_controls)
        lbl_title_color = QLabel("COLOR BY")
        lbl_title_color.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title_color.setProperty("class", "ribbonTitle")
        v_color.addWidget(lbl_title_color)
        h_layout.addWidget(grp_color)

        # 2. LOD & Quality Group
        grp_lod = QFrame()
        grp_lod.setProperty("class", "ribbonGroup")
        v_lod = QVBoxLayout(grp_lod)
        v_lod.setContentsMargins(10, 7, 10, 7)
        v_lod.setSpacing(3)

        h_lod_controls = QHBoxLayout()
        h_lod_controls.setSpacing(8)

        # Splat Quality
        v_splat = QVBoxLayout()
        v_splat.setSpacing(1)
        lbl_splat = QLabel("Quality:")
        lbl_splat.setProperty("class", "ribbonLabel")
        self.ribbon_combo_splat = QComboBox()
        self.ribbon_combo_splat.setProperty("class", "ribbonCombo")
        self.ribbon_combo_splat.addItem("Standard", "standard")
        self.ribbon_combo_splat.addItem("High Quality (HQ)", "hq")
        self.ribbon_combo_splat.currentIndexChanged.connect(self._on_ribbon_splat_changed)
        v_splat.addWidget(lbl_splat)
        v_splat.addWidget(self.ribbon_combo_splat)
        h_lod_controls.addLayout(v_splat)

        # Point Budget
        v_budget = QVBoxLayout()
        v_budget.setSpacing(1)
        lbl_budget = QLabel("Budget:")
        lbl_budget.setProperty("class", "ribbonLabel")
        self.ribbon_combo_budget = QComboBox()
        self.ribbon_combo_budget.setProperty("class", "ribbonCombo")
        self.ribbon_combo_budget.addItem("1 Million", 1000000)
        self.ribbon_combo_budget.addItem("2.5 Million", 2500000)
        self.ribbon_combo_budget.addItem("5 Million", 5000000)
        self.ribbon_combo_budget.addItem("10 Million", 10000000)
        self.ribbon_combo_budget.addItem("15 Million", 15000000)
        self.ribbon_combo_budget.setCurrentIndex(1)
        self.ribbon_combo_budget.currentIndexChanged.connect(self._on_ribbon_budget_changed)
        v_budget.addWidget(lbl_budget)
        v_budget.addWidget(self.ribbon_combo_budget)
        h_lod_controls.addLayout(v_budget)

        # Min Node Size
        v_node = QVBoxLayout()
        v_node.setSpacing(1)
        lbl_node = QLabel("Min Node:")
        lbl_node.setProperty("class", "ribbonLabel")
        self.ribbon_combo_node = QComboBox()
        self.ribbon_combo_node.setProperty("class", "ribbonCombo")
        self.ribbon_combo_node.addItem("5 px (Dense)", 5)
        self.ribbon_combo_node.addItem("10 px (Default)", 10)
        self.ribbon_combo_node.addItem("15 px", 15)
        self.ribbon_combo_node.addItem("20 px (Fast)", 20)
        self.ribbon_combo_node.setCurrentIndex(1)
        self.ribbon_combo_node.currentIndexChanged.connect(lambda: self.set_min_node_size(self.ribbon_combo_node.currentData()))
        v_node.addWidget(lbl_node)
        v_node.addWidget(self.ribbon_combo_node)
        h_lod_controls.addLayout(v_node)

        v_lod.addLayout(h_lod_controls)
        lbl_title_lod = QLabel("LOD & QUALITY")
        lbl_title_lod.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title_lod.setProperty("class", "ribbonTitle")
        v_lod.addWidget(lbl_title_lod)
        h_layout.addWidget(grp_lod)

        # 3. Viewport Selection & Staging Tools Group
        grp_tools = QFrame()
        grp_tools.setProperty("class", "ribbonGroup")
        v_tools = QVBoxLayout(grp_tools)
        v_tools.setContentsMargins(10, 7, 10, 7)
        v_tools.setSpacing(3)

        h_tools = QHBoxLayout()
        h_tools.setSpacing(4)

        btn_orbit = QPushButton("Navigate")
        btn_orbit.setProperty("class", "ribbonToolBtn")
        btn_orbit.setCheckable(True)
        btn_orbit.setChecked(True)
        btn_orbit.clicked.connect(lambda checked=False: self.activate_web_tool("nav"))
        self._tool_buttons["nav"] = btn_orbit

        btn_box = QPushButton("Box Select")
        btn_box.setProperty("class", "ribbonToolBtn")
        btn_box.setCheckable(True)
        btn_box.clicked.connect(lambda checked=False: self.activate_web_tool("box"))
        self._tool_buttons["box"] = btn_box

        btn_pick = QPushButton("Pick Point")
        btn_pick.setToolTip("Pick Point (Ctrl+Click on any point to inspect and highlight)")
        btn_pick.setProperty("class", "ribbonToolBtn")
        btn_pick.setCheckable(True)
        btn_pick.clicked.connect(lambda checked=False: self.activate_web_tool("inspect"))
        self._tool_buttons["inspect"] = btn_pick

        btn_commit_ribbon = QPushButton("Commit Edits")
        btn_commit_ribbon.setProperty("class", "ribbonToolBtn")
        btn_commit_ribbon.setObjectName("btnSuccess")
        btn_commit_ribbon.clicked.connect(self.commit_staged_edits_action)

        btn_clear = QPushButton("Clear")
        btn_clear.setProperty("class", "ribbonToolBtn")
        btn_clear.setObjectName("btnDanger")
        btn_clear.clicked.connect(lambda: self.activate_web_tool("clear"))

        h_tools.addWidget(btn_orbit)
        h_tools.addWidget(btn_box)
        h_tools.addWidget(btn_pick)
        h_tools.addWidget(btn_commit_ribbon)
        h_tools.addWidget(btn_clear)

        v_tools.addLayout(h_tools)

        # Point-to-point distance measuring on the displayed tree (Forestry
        # overlay): markers + 3D/horizontal distance labels; double-click clears.
        self.chk_ribbon_measure = QCheckBox("Measure")
        self.chk_ribbon_measure.setStyleSheet("color: #e2e8f0; font-size: 11px; font-weight: 600;")
        self.chk_ribbon_measure.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_ribbon_measure.toggled.connect(self._toggle_tree_measure)
        v_tools.addWidget(self.chk_ribbon_measure, alignment=Qt.AlignmentFlag.AlignCenter)

        lbl_title_tools = QLabel("SELECTION & EDITING")
        lbl_title_tools.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title_tools.setProperty("class", "ribbonTitle")
        v_tools.addWidget(lbl_title_tools)
        h_layout.addWidget(grp_tools)

        h_layout.addStretch()
        return page

    # Category 3: Layers Page
    def _create_ribbon_layers_page(self) -> QWidget:
        page = QFrame()
        page.setFixedHeight(92)
        h_layout = QHBoxLayout(page)
        h_layout.setContentsMargins(8, 6, 8, 6)
        h_layout.setSpacing(8)

        # 1. Quick Layer Visibility Toggles Group
        grp_vis = QFrame()
        grp_vis.setProperty("class", "ribbonGroup")
        v_vis = QVBoxLayout(grp_vis)
        v_vis.setContentsMargins(10, 7, 10, 7)
        v_vis.setSpacing(2)

        g_checks = QGridLayout()
        g_checks.setHorizontalSpacing(10)
        g_checks.setVerticalSpacing(2)

        self.ribbon_chk_pc = QCheckBox("Point Cloud")
        self.ribbon_chk_pc.setChecked(True)
        self.ribbon_chk_pc.toggled.connect(self.toggle_pointcloud_layer)
        g_checks.addWidget(self.ribbon_chk_pc, 0, 0)

        self.ribbon_chk_staged = QCheckBox("Staged Overlays")
        self.ribbon_chk_staged.setChecked(True)
        self.ribbon_chk_staged.toggled.connect(self.toggle_staged_layer)
        g_checks.addWidget(self.ribbon_chk_staged, 0, 1)

        self.ribbon_chk_pick = QCheckBox("Pick Reticle")
        self.ribbon_chk_pick.setChecked(True)
        self.ribbon_chk_pick.toggled.connect(self.toggle_pick_layer)
        g_checks.addWidget(self.ribbon_chk_pick, 1, 0)

        self.ribbon_chk_sel = QCheckBox("Boundary")
        self.ribbon_chk_sel.setChecked(True)
        self.ribbon_chk_sel.toggled.connect(self.toggle_sel_layer)
        g_checks.addWidget(self.ribbon_chk_sel, 1, 1)

        v_vis.addLayout(g_checks)
        lbl_title_vis = QLabel("LAYER VISIBILITY")
        lbl_title_vis.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title_vis.setProperty("class", "ribbonTitle")
        v_vis.addWidget(lbl_title_vis)
        h_layout.addWidget(grp_vis)

        # 2. Master Visibility Actions Group
        grp_master = QFrame()
        grp_master.setProperty("class", "ribbonGroup")
        v_master = QVBoxLayout(grp_master)
        v_master.setContentsMargins(10, 7, 10, 7)
        v_master.setSpacing(3)

        h_master_btns = QHBoxLayout()
        h_master_btns.setSpacing(4)

        btn_show = QPushButton("Show All")
        btn_show.setProperty("class", "ribbonToolBtn")
        btn_show.clicked.connect(self.show_all_layers)

        btn_hide = QPushButton("Hide All")
        btn_hide.setProperty("class", "ribbonToolBtn")
        btn_hide.clicked.connect(self.hide_all_layers)

        h_master_btns.addWidget(btn_show)
        h_master_btns.addWidget(btn_hide)
        v_master.addLayout(h_master_btns)

        lbl_title_master = QLabel("MASTER CONTROLS")
        lbl_title_master.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title_master.setProperty("class", "ribbonTitle")
        v_master.addWidget(lbl_title_master)
        h_layout.addWidget(grp_master)

        # 3. Sidebar Focus Group
        grp_sidebar = QFrame()
        grp_sidebar.setProperty("class", "ribbonGroup")
        v_sidebar = QVBoxLayout(grp_sidebar)
        v_sidebar.setContentsMargins(10, 7, 10, 7)
        v_sidebar.setSpacing(3)

        btn_focus_lm = QPushButton("Focus Layer Manager")
        btn_focus_lm.setProperty("class", "ribbonToolBtn")
        btn_focus_lm.clicked.connect(lambda: self.sidebar_tabs.setCurrentIndex(0))
        v_sidebar.addWidget(btn_focus_lm)

        lbl_title_sidebar = QLabel("SIDEBAR")
        lbl_title_sidebar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title_sidebar.setProperty("class", "ribbonTitle")
        v_sidebar.addWidget(lbl_title_sidebar)
        h_layout.addWidget(grp_sidebar)

        h_layout.addStretch()
        return page

    # Category 4: Settings Page
    def _create_ribbon_settings_page(self) -> QWidget:
        page = QFrame()
        page.setFixedHeight(92)
        h_layout = QHBoxLayout(page)
        h_layout.setContentsMargins(8, 6, 8, 6)
        h_layout.setSpacing(8)

        # 1. Lighting (EDL) Group
        grp_edl = QFrame()
        grp_edl.setProperty("class", "ribbonGroup")
        v_edl = QVBoxLayout(grp_edl)
        v_edl.setContentsMargins(10, 7, 10, 7)
        v_edl.setSpacing(3)

        h_edl_controls = QHBoxLayout()
        h_edl_controls.setSpacing(8)

        self.ribbon_chk_edl = QCheckBox("EDL")
        self.ribbon_chk_edl.setChecked(True)
        self.ribbon_chk_edl.toggled.connect(self._on_ribbon_edl_toggled)
        h_edl_controls.addWidget(self.ribbon_chk_edl)

        # Radius
        v_rad = QVBoxLayout()
        v_rad.setSpacing(1)
        lbl_rad = QLabel("Radius:")
        lbl_rad.setProperty("class", "ribbonLabel")
        self.ribbon_spin_radius = QDoubleSpinBox()
        self.ribbon_spin_radius.setProperty("class", "ribbonSpin")
        self.ribbon_spin_radius.setRange(0.5, 4.0)
        self.ribbon_spin_radius.setSingleStep(0.1)
        self.ribbon_spin_radius.setValue(1.4)
        self.ribbon_spin_radius.setSuffix(" px")
        self.ribbon_spin_radius.setFixedWidth(65)
        self.ribbon_spin_radius.valueChanged.connect(self.set_edl_radius)
        v_rad.addWidget(lbl_rad)
        v_rad.addWidget(self.ribbon_spin_radius)
        h_edl_controls.addLayout(v_rad)

        # Strength
        v_str = QVBoxLayout()
        v_str.setSpacing(1)
        lbl_str = QLabel("Strength:")
        lbl_str.setProperty("class", "ribbonLabel")
        self.ribbon_spin_strength = QDoubleSpinBox()
        self.ribbon_spin_strength.setProperty("class", "ribbonSpin")
        self.ribbon_spin_strength.setRange(0.1, 4.0)
        self.ribbon_spin_strength.setSingleStep(0.1)
        self.ribbon_spin_strength.setValue(1.0)
        self.ribbon_spin_strength.setFixedWidth(55)
        self.ribbon_spin_strength.valueChanged.connect(self.set_edl_strength)
        v_str.addWidget(lbl_str)
        v_str.addWidget(self.ribbon_spin_strength)
        h_edl_controls.addLayout(v_str)

        # Opacity
        v_op = QVBoxLayout()
        v_op.setSpacing(1)
        lbl_op = QLabel("Opacity:")
        lbl_op.setProperty("class", "ribbonLabel")
        self.ribbon_spin_opacity = QDoubleSpinBox()
        self.ribbon_spin_opacity.setProperty("class", "ribbonSpin")
        self.ribbon_spin_opacity.setRange(0.1, 1.0)
        self.ribbon_spin_opacity.setSingleStep(0.05)
        self.ribbon_spin_opacity.setValue(1.0)
        self.ribbon_spin_opacity.setFixedWidth(55)
        self.ribbon_spin_opacity.valueChanged.connect(self.set_edl_opacity)
        v_op.addWidget(lbl_op)
        v_op.addWidget(self.ribbon_spin_opacity)
        h_edl_controls.addLayout(v_op)

        v_edl.addLayout(h_edl_controls)
        lbl_title_edl = QLabel("LIGHTING & EDL")
        lbl_title_edl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title_edl.setProperty("class", "ribbonTitle")
        v_edl.addWidget(lbl_title_edl)
        h_layout.addWidget(grp_edl)

        # 2. Display & Coloring Group
        grp_display = QFrame()
        grp_display.setProperty("class", "ribbonGroup")
        v_display = QVBoxLayout(grp_display)
        v_display.setContentsMargins(10, 7, 10, 7)
        v_display.setSpacing(3)

        h_disp_controls = QHBoxLayout()
        h_disp_controls.setSpacing(8)

        # Background
        v_bg = QVBoxLayout()
        v_bg.setSpacing(1)
        lbl_bg = QLabel("Background:")
        lbl_bg.setProperty("class", "ribbonLabel")
        self.ribbon_combo_bg = QComboBox()
        self.ribbon_combo_bg.setProperty("class", "ribbonCombo")
        self.ribbon_combo_bg.addItem("Black", "black")
        self.ribbon_combo_bg.addItem("Gradient", "gradient")
        self.ribbon_combo_bg.addItem("Skybox", "skybox")
        self.ribbon_combo_bg.addItem("White", "white")
        self.ribbon_combo_bg.currentIndexChanged.connect(self._on_ribbon_bg_changed)
        v_bg.addWidget(lbl_bg)
        v_bg.addWidget(self.ribbon_combo_bg)
        # Render Mode: one switch for the whole scene's look.  Performance keeps the
        # shipped point settings, Quality goes to round splats, smaller size, HQ splat
        # and the top budget; Balanced sits between them.
        v_render_mode = QVBoxLayout()
        v_render_mode.setSpacing(1)
        lbl_render_mode = QLabel("Render Mode:")
        lbl_render_mode.setProperty("class", "ribbonLabel")
        self.ribbon_combo_render_mode = QComboBox()
        self.ribbon_combo_render_mode.setProperty("class", "ribbonCombo")
        for label, value in (("Performance", "performance"),
                             ("Balanced", "balanced"),
                             ("Quality", "quality")):
            self.ribbon_combo_render_mode.addItem(label, value)
        self.ribbon_combo_render_mode.currentIndexChanged.connect(
            self._on_ribbon_render_mode_changed)
        v_render_mode.addWidget(lbl_render_mode)
        v_render_mode.addWidget(self.ribbon_combo_render_mode)
        h_disp_controls.addLayout(v_render_mode)

        h_disp_controls.addLayout(v_bg)

        v_display.addLayout(h_disp_controls)
        lbl_title_disp = QLabel("DISPLAY & COLORS")
        lbl_title_disp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title_disp.setProperty("class", "ribbonTitle")
        v_display.addWidget(lbl_title_disp)
        h_layout.addWidget(grp_display)

        h_layout.addStretch()
        return page

    # Category 5: Stand Filter Page (shown under the ribbon "Filter" menu)
    def _create_ribbon_filter_page(self) -> QWidget:
        page = QFrame()
        page.setFixedHeight(92)
        h_layout = QHBoxLayout(page)
        h_layout.setContentsMargins(8, 6, 8, 6)
        h_layout.setSpacing(8)

        # Stand Filter Group (moved from the ribbon category row top bar)
        grp_filter = QFrame()
        grp_filter.setProperty("class", "ribbonGroup")
        grp_filter.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        v_filter_grp = QVBoxLayout(grp_filter)
        v_filter_grp.setContentsMargins(10, 7, 10, 7)
        v_filter_grp.setSpacing(3)

        self.bar_stand_filter = QFrame()
        h_sfb = QHBoxLayout(self.bar_stand_filter)
        h_sfb.setContentsMargins(0, 0, 0, 0)
        h_sfb.setSpacing(6)

        # Info label kept alive for status updates, but not shown in the ribbon
        self.lbl_top_stand_info = QLabel("🌲 Stand: Tree #- (0 neighbors)")

        # Mode toggle (moved here from the Forestry sidebar group)
        self.btn_tree_filter_mode = QPushButton("Filter Mode")
        self.btn_tree_filter_mode.setCheckable(True)
        self.btn_tree_filter_mode.setProperty("class", "ribbonToolBtn")
        self.btn_tree_filter_mode.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_tree_filter_mode.toggled.connect(self._on_tree_filter_mode_toggled)
        h_sfb.addWidget(self.btn_tree_filter_mode)

        self.btn_top_filter = QPushButton("Filter")
        self.btn_top_filter.setProperty("class", "ribbonToolBtn")
        self.btn_top_filter.setObjectName("btnSuccess")
        self.btn_top_filter.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_top_filter.clicked.connect(self._apply_tree_filter)
        h_sfb.addWidget(self.btn_top_filter)

        self.btn_top_reset = QPushButton("Reset")
        self.btn_top_reset.setProperty("class", "ribbonToolBtn")
        self.btn_top_reset.setObjectName("btnSecondary")
        self.btn_top_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_top_reset.clicked.connect(self._reset_tree_filter)
        h_sfb.addWidget(self.btn_top_reset)

        # Buttons are always shown on the Filter page; they start disabled (not
        # highlighted) and light up once a tree stand with neighbors is loaded.
        self.bar_stand_filter.setVisible(True)
        self._set_stand_filter_buttons_enabled(False)
        v_filter_grp.addWidget(self.bar_stand_filter)
        v_filter_grp.addStretch()

        lbl_filter = QLabel("STAND FILTER")
        lbl_filter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_filter.setProperty("class", "ribbonTitle")
        v_filter_grp.addWidget(lbl_filter)

        h_layout.addWidget(grp_filter)
        h_layout.addStretch()
        return page

    # Category 6: AI Page (shown under the ribbon "AI" menu)
    def _create_ribbon_ai_page(self) -> QWidget:
        page = QFrame()
        page.setFixedHeight(92)
        h_layout = QHBoxLayout(page)
        h_layout.setContentsMargins(8, 6, 8, 6)
        h_layout.setSpacing(8)

        # 1. AI Segmentation Toolset Group
        grp_ai = QFrame()
        grp_ai.setProperty("class", "ribbonGroup")
        grp_ai.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        v_ai = QVBoxLayout(grp_ai)
        v_ai.setContentsMargins(10, 7, 10, 7)
        v_ai.setSpacing(3)

        h_ai = QHBoxLayout()
        h_ai.setSpacing(6)

        # Tree ID selector
        h_tid = QHBoxLayout()
        h_tid.setSpacing(4)
        lbl_tid = QLabel("Tree ID:")
        lbl_tid.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.spin_ai_tree = QSpinBox()
        self.spin_ai_tree.setRange(1, 999999)
        self.spin_ai_tree.setValue(1)
        self.spin_ai_tree.setStyleSheet("font-size: 11px; padding: 2px 4px; min-width: 60px;")
        self.spin_ai_tree.setCursor(Qt.CursorShape.PointingHandCursor)
        self.spin_ai_tree.valueChanged.connect(self._on_ai_tree_spin_changed)
        h_tid.addWidget(lbl_tid)
        h_tid.addWidget(self.spin_ai_tree)
        h_ai.addLayout(h_tid)

        btn_ai_view = QPushButton("View Tree")
        btn_ai_view.setProperty("class", "ribbonToolBtn")
        btn_ai_view.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_ai_view.setToolTip("View full-density point cloud for the selected tree")
        btn_ai_view.clicked.connect(self.view_tree_action)
        h_ai.addWidget(btn_ai_view)

        self.btn_segment_tree = QPushButton("Segment (ForestFormer3D)")
        self.btn_segment_tree.setProperty("class", "ribbonToolBtn")
        self.btn_segment_tree.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_segment_tree.setStyleSheet("font-weight: 600;")
        self.btn_segment_tree.setToolTip("Run ForestFormer3D deep learning instance segmentation to split merged trees")
        self.btn_segment_tree.clicked.connect(self.segment_tree_action)
        h_ai.addWidget(self.btn_segment_tree)

        self.btn_confirm_segmentation = QPushButton("Confirm & Stage Split Edits")
        self.btn_confirm_segmentation.setProperty("class", "ribbonToolBtn")
        self.btn_confirm_segmentation.setObjectName("btnSuccess")
        self.btn_confirm_segmentation.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_confirm_segmentation.setEnabled(False)
        self.btn_confirm_segmentation.setStyleSheet("""
            QPushButton#btnSuccess {
                background-color: #059669;
                border: 1px solid #10b981;
                color: #ffffff;
                font-weight: 700;
                padding: 4px 10px;
                border-radius: 4px;
            }
            QPushButton#btnSuccess:hover {
                background-color: #10b981;
            }
            QPushButton#btnSuccess:disabled {
                background-color: #1e293b;
                border-color: #334155;
                color: #64748b;
            }
        """)
        self.btn_confirm_segmentation.clicked.connect(self.confirm_segmentation_action)
        h_ai.addWidget(self.btn_confirm_segmentation)

        v_ai.addLayout(h_ai)
        v_ai.addStretch()

        lbl_ai_title = QLabel("AI SEGMENTATION TOOLSET")
        lbl_ai_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_ai_title.setProperty("class", "ribbonTitle")
        v_ai.addWidget(lbl_ai_title)
        h_layout.addWidget(grp_ai)

        # 2. Stand & Neighbors Options Group
        grp_stand = QFrame()
        grp_stand.setProperty("class", "ribbonGroup")
        grp_stand.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        v_stand = QVBoxLayout(grp_stand)
        v_stand.setContentsMargins(10, 7, 10, 7)
        v_stand.setSpacing(3)

        h_stand = QHBoxLayout()
        h_stand.setSpacing(6)

        self.chk_ai_neighbors = QCheckBox("Neighbors")
        self.chk_ai_neighbors.setStyleSheet("color: #e2e8f0; font-size: 11px; font-weight: 500;")
        self.chk_ai_neighbors.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_ai_neighbors.toggled.connect(self._on_ai_neighbors_toggled)
        h_stand.addWidget(self.chk_ai_neighbors)

        self.spin_ai_radius = QDoubleSpinBox()
        self.spin_ai_radius.setRange(2.0, 100.0)
        self.spin_ai_radius.setValue(5.0)
        self.spin_ai_radius.setSingleStep(5.0)
        self.spin_ai_radius.setSuffix(" m")
        self.spin_ai_radius.setStyleSheet("font-size: 11px; padding: 2px 4px; max-width: 60px;")
        self.spin_ai_radius.setEnabled(False)
        self.spin_ai_radius.valueChanged.connect(self._on_ai_radius_changed)
        h_stand.addWidget(self.spin_ai_radius)

        v_stand.addLayout(h_stand)
        v_stand.addStretch()

        lbl_stand_title = QLabel("STAND CONTEXT")
        lbl_stand_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_stand_title.setProperty("class", "ribbonTitle")
        v_stand.addWidget(lbl_stand_title)
        h_layout.addWidget(grp_stand)

        # 3. History & Audit Group
        grp_hist = QFrame()
        grp_hist.setProperty("class", "ribbonGroup")
        grp_hist.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        v_hist = QVBoxLayout(grp_hist)
        v_hist.setContentsMargins(10, 7, 10, 7)
        v_hist.setSpacing(3)

        h_hist = QHBoxLayout()
        h_hist.setSpacing(4)
        btn_hist = QPushButton("Edit History (Ctrl+H)...")
        btn_hist.setProperty("class", "ribbonToolBtn")
        btn_hist.setObjectName("btnSecondary")
        btn_hist.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_hist.setToolTip("Open database log of edits and restore previous versions")
        btn_hist.clicked.connect(self.open_edit_history_dialog)
        h_hist.addWidget(btn_hist)
        v_hist.addLayout(h_hist)
        v_hist.addStretch()

        lbl_hist_title = QLabel("HISTORY & REVERT")
        lbl_hist_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_hist_title.setProperty("class", "ribbonTitle")
        v_hist.addWidget(lbl_hist_title)
        h_layout.addWidget(grp_hist)

        h_layout.addStretch()
        return page

    def _on_ai_tree_spin_changed(self, val: int):
        if hasattr(self, "spin_tree_view") and self.spin_tree_view.value() != val:
            self.spin_tree_view.blockSignals(True)
            self.spin_tree_view.setValue(val)
            self.spin_tree_view.blockSignals(False)
        self._on_tree_spin_changed(val)

    def _on_ai_neighbors_toggled(self, checked: bool):
        if hasattr(self, "spin_ai_radius"):
            self.spin_ai_radius.setEnabled(checked)
        if hasattr(self, "chk_tree_neighbors") and self.chk_tree_neighbors.isChecked() != checked:
            self.chk_tree_neighbors.blockSignals(True)
            self.chk_tree_neighbors.setChecked(checked)
            self.chk_tree_neighbors.blockSignals(False)
            self._on_tree_neighbors_toggled(checked)

    def _on_ai_radius_changed(self, val: float):
        if hasattr(self, "spin_tree_radius") and abs(self.spin_tree_radius.value() - val) > 1e-4:
            self.spin_tree_radius.blockSignals(True)
            self.spin_tree_radius.setValue(val)
            self.spin_tree_radius.blockSignals(False)

    def _on_tree_radius_changed(self, val: float):
        if hasattr(self, "spin_ai_radius") and abs(self.spin_ai_radius.value() - val) > 1e-4:
            self.spin_ai_radius.blockSignals(True)
            self.spin_ai_radius.setValue(val)
            self.spin_ai_radius.blockSignals(False)

    # Ribbon Event Handlers
    def _on_ribbon_category_clicked(self, idx: int):
        self.ribbon_stack.setCurrentIndex(idx)
        if idx == 2:  # Layers tab
            if hasattr(self, "sidebar_tabs"):
                self.sidebar_tabs.setCurrentIndex(0)
        elif idx in (4, 5):  # Filter tab (4) or AI tab (5)
            if hasattr(self, "sidebar_tabs"):
                self.sidebar_tabs.setCurrentIndex(1)

    def _on_ribbon_shape_changed(self):
        shape = self.ribbon_combo_shape.currentData()
        self.set_point_shape(shape)

    def _on_ribbon_sizing_changed(self):
        stype = self.ribbon_combo_sizing.currentData()
        self.set_point_size_type(stype)

    def _on_ribbon_size_slider_changed(self, ival: int):
        fval = round(ival / 10.0, 2)
        self.ribbon_spin_size.blockSignals(True)
        self.ribbon_spin_size.setValue(fval)
        self.ribbon_spin_size.blockSignals(False)
        if hasattr(self, "slider_layer_size"):
            self.slider_layer_size.blockSignals(True)
            self.slider_layer_size.setValue(ival)
            self.slider_layer_size.blockSignals(False)
        if hasattr(self, "lbl_layer_size_val"):
            self.lbl_layer_size_val.setText(f"{fval:.2f}")
        self.set_point_size(fval)

    def _on_ribbon_size_spin_changed(self, fval: float):
        ival = int(round(fval * 10))
        self.ribbon_slider_size.blockSignals(True)
        self.ribbon_slider_size.setValue(ival)
        self.ribbon_slider_size.blockSignals(False)
        if hasattr(self, "slider_layer_size"):
            self.slider_layer_size.blockSignals(True)
            self.slider_layer_size.setValue(ival)
            self.slider_layer_size.blockSignals(False)
        if hasattr(self, "lbl_layer_size_val"):
            self.lbl_layer_size_val.setText(f"{fval:.2f}")
        self.set_point_size(fval)

    def _on_ribbon_splat_changed(self):
        quality = self.ribbon_combo_splat.currentData()
        self.set_splat_quality(quality)

    def _on_ribbon_budget_changed(self):
        budget = self.ribbon_combo_budget.currentData()
        self.set_point_budget(budget)

    def _on_ribbon_edl_toggled(self, checked: bool):
        self.ribbon_spin_radius.setEnabled(checked)
        self.ribbon_spin_strength.setEnabled(checked)
        self.set_edl_enabled(checked)

    def _on_ribbon_color_changed(self):
        attr = self.ribbon_combo_color.currentData()
        self._sync_color_combos(attr)
        self.set_color_mode(attr)

    def _sync_viewer_render_widgets(self, info):
        """Mirror the scene-wide half of a render mode into the Point tab.

        Budget, minimum node size, splat quality and EDL belong to the viewer rather
        than to a layer, so the per-layer sync cannot reach them.  Without this the
        Settings combo says Quality while the Point tab still reads 2.5 M / 10 px.
        """
        for name, key in (("ribbon_combo_budget", "pointBudget"),
                          ("ribbon_combo_node", "minNodeSize"),
                          ("ribbon_combo_splat", "splatQuality")):
            combo = getattr(self, name, None)
            if combo is None or info.get(key) is None:
                continue
            idx = combo.findData(info[key])
            if idx >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)
        chk = getattr(self, "ribbon_chk_edl", None)
        if chk is not None and info.get("edlEnabled") is not None:
            chk.blockSignals(True)
            chk.setChecked(bool(info["edlEnabled"]))
            chk.blockSignals(False)

    def _on_ribbon_render_mode_changed(self):
        """Performance / Balanced / Quality, applied to every layer at once.

        The viewer hands back the layer list, so the Point tab's shape, sizing and size
        widgets land on the new mode's values; a second read pulls the scene-wide half
        (budget, node size, splat quality, EDL) so nothing in the ribbon is left stale.
        """
        mode = self.ribbon_combo_render_mode.currentData()

        def apply_layers(layers):
            self._render_layer_rows(layers if isinstance(layers, list) else [])
            self._run_js(
                "window.getRenderSettings ? window.getRenderSettings() : null;",
                lambda settings: self._sync_viewer_render_widgets(
                    settings if isinstance(settings, dict) else {}))

        self._run_js(
            f"window.setRenderMode({json.dumps(mode)});", apply_layers)

    def _on_ribbon_bg_changed(self):
        bg = self.ribbon_combo_bg.currentData()
        self.set_background(bg)

    def activate_web_tool(self, tool_name: str):
        if tool_name == "clear":
            self.clear_selection()
            return
        # Re-clicking the active lasso/box/pick tool turns it off, back to navigation.
        if self._active_tool == tool_name and tool_name in ("polygon", "box", "inspect"):
            tool_name = "nav"
        self._active_tool = tool_name
        self._run_js(f"window.setActiveTool('{tool_name}');")
        self._sync_tool_buttons()

    def _sync_tool_buttons(self):
        """Keep the ribbon tool buttons' checked state in sync with the viewer mode."""
        for name, btn in self._tool_buttons.items():
            btn.blockSignals(True)
            btn.setChecked(name == self._active_tool)
            btn.blockSignals(False)
