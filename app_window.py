"""
Main Application Window for Potree Hybrid Viewer.
Combines:
- QWebEngineView hosting the Potree WebGL streaming renderer.
- Native PyQt6 sidebar with LAS/LAZ editing controls, ITC assignment, noise tagging, and point cloud export.
"""

import os
import sys
import subprocess
import threading
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox, QProgressBar, QSplitter,
    QFrame, QComboBox, QCheckBox, QSpinBox, QGroupBox, QStatusBar,
    QScrollArea
)
from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QIcon, QColor
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings

from pointcloud_manager import PointCloudManager
from server import HybridServer


class BridgeSignaler(QObject):
    selection_changed = pyqtSignal(dict)
    point_picked = pyqtSignal(dict)
    edit_applied = pyqtSignal(dict)


class MainWindow(QMainWindow):
    def __init__(self, server: HybridServer, manager: PointCloudManager, potree_dir: str):
        super().__init__()
        self.server = server
        self.manager = manager
        self.potree_dir = potree_dir

        self.current_las_path: Optional[str] = None
        self.current_potree_url: Optional[str] = None

        self.signaler = BridgeSignaler()
        self.signaler.selection_changed.connect(self.on_selection_changed)
        self.signaler.point_picked.connect(self.on_point_picked)
        self.signaler.edit_applied.connect(self.on_edit_applied)

        # Wire server events to Qt signals (thread-safe UI updates)
        self.server.on_event(self._on_server_event)

        self.init_ui()

    def _on_server_event(self, event_name: str, data: dict):
        if event_name == "selection_changed":
            self.signaler.selection_changed.emit(data)
        elif event_name == "point_picked":
            self.signaler.point_picked.emit(data)
        elif event_name == "edit_applied":
            self.signaler.edit_applied.emit(data)

    def init_ui(self):
        self.setWindowTitle("Potree Hybrid Editor (LOD Streaming + Point Cloud Editing)")
        self.resize(1440, 900)

        # Dark theme styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e24;
                color: #e2e8f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #334155;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 14px;
                background-color: #18181f;
                color: #93c5fd;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #2563eb;
                color: #ffffff;
                border: 1px solid #1d4ed8;
                border-radius: 5px;
                padding: 6px 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #3b82f6;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
            QPushButton#btnDanger {
                background-color: #dc2626;
                border-color: #b91c1c;
            }
            QPushButton#btnDanger:hover {
                background-color: #ef4444;
            }
            QPushButton#btnSecondary {
                background-color: #334155;
                border-color: #475569;
            }
            QPushButton#btnSecondary:hover {
                background-color: #475569;
            }
            QComboBox, QSpinBox {
                background-color: #0f172a;
                color: #f8fafc;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 5px 8px;
            }
            QLabel {
                color: #cbd5e1;
            }
            QStatusBar {
                background-color: #0f172a;
                color: #94a3b8;
                border-top: 1px solid #334155;
            }
        """)

        # Central Splitter: Left Sidebar + Central WebEngine Viewport
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        sidebar_widget = self.create_sidebar()
        splitter.addWidget(sidebar_widget)

        # WebEngine Viewport
        self.web_view = QWebEngineView()
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)

        viewer_url = f"http://127.0.0.1:{self.server.port}/"
        self.web_view.load(QUrl(viewer_url))
        splitter.addWidget(self.web_view)

        # Set initial splitter proportions: Sidebar 340px, Viewport remainder
        splitter.setSizes([340, 1100])

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Load a point cloud to begin.")

    def create_sidebar(self) -> QWidget:
        container = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # 1. Dataset Management Group
        grp_data = QGroupBox("Point Cloud Dataset")
        v_data = QVBoxLayout(grp_data)

        self.lbl_cloud_info = QLabel("No point cloud loaded.")
        self.lbl_cloud_info.setWordWrap(True)
        v_data.addWidget(self.lbl_cloud_info)

        btn_open_potree = QPushButton("📂 Open Potree Converted...")
        btn_open_potree.clicked.connect(self.open_potree_file)
        v_data.addWidget(btn_open_potree)

        btn_open_las = QPushButton("🌲 Open LAS / LAZ (with Auto-Convert)...")
        btn_open_las.setObjectName("btnSecondary")
        btn_open_las.clicked.connect(self.open_las_file)
        v_data.addWidget(btn_open_las)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        v_data.addWidget(self.progress_bar)

        layout.addWidget(grp_data)

        # 2. Selection & Editing Group
        grp_edit = QGroupBox("Selection & Editing (LAS Bridge)")
        v_edit = QVBoxLayout(grp_edit)

        self.lbl_sel_count = QLabel("Selected: 0 points")
        self.lbl_sel_count.setStyleSheet("font-weight: bold; color: #60a5fa;")
        v_edit.addWidget(self.lbl_sel_count)

        self.lbl_sel_classes = QLabel("Classes: -")
        v_edit.addWidget(self.lbl_sel_classes)

        self.lbl_sel_itc = QLabel("ITC IDs: -")
        v_edit.addWidget(self.lbl_sel_itc)

        # Action: Assign Tree ID (ITC)
        h_itc = QHBoxLayout()
        h_itc.addWidget(QLabel("Tree ID (ITC):"))
        self.spin_itc = QSpinBox()
        self.spin_itc.setRange(1, 999999)
        self.spin_itc.setValue(1)
        h_itc.addWidget(self.spin_itc)
        v_edit.addLayout(h_itc)

        btn_assign_itc = QPushButton("🏷️ Assign Tree ID to Selection")
        btn_assign_itc.clicked.connect(self.apply_assign_itc)
        v_edit.addWidget(btn_assign_itc)

        # Action: Mark as Noise
        btn_noise = QPushButton("⚠️ Convert Selection to Noise (Class 7)")
        btn_noise.setObjectName("btnDanger")
        btn_noise.clicked.connect(self.apply_mark_noise)
        v_edit.addWidget(btn_noise)

        # Action: Set Classification
        h_cls = QHBoxLayout()
        h_cls.addWidget(QLabel("ASPRS Class:"))
        self.combo_class = QComboBox()
        self.combo_class.addItem("0: Unclassified", 0)
        self.combo_class.addItem("1: Unassigned", 1)
        self.combo_class.addItem("2: Ground", 2)
        self.combo_class.addItem("3: Low Vegetation", 3)
        self.combo_class.addItem("4: Medium Vegetation", 4)
        self.combo_class.addItem("5: High Vegetation", 5)
        self.combo_class.addItem("7: Noise", 7)
        h_cls.addWidget(self.combo_class)
        v_edit.addLayout(h_cls)

        btn_set_class = QPushButton("Set Classification")
        btn_set_class.setObjectName("btnSecondary")
        btn_set_class.clicked.connect(self.apply_set_classification)
        v_edit.addWidget(btn_set_class)

        # Action: Delete Selection
        btn_del = QPushButton("🗑️ Delete Selected Points")
        btn_del.setObjectName("btnDanger")
        btn_del.clicked.connect(self.apply_delete_selection)
        v_edit.addWidget(btn_del)

        # Action: Clear Selection
        btn_clear = QPushButton("✕ Clear Selection")
        btn_clear.setObjectName("btnSecondary")
        btn_clear.clicked.connect(self.clear_selection)
        v_edit.addWidget(btn_clear)

        layout.addWidget(grp_edit)

        # 3. Save & Export Group
        grp_save = QGroupBox("Save & Export")
        v_save = QVBoxLayout(grp_save)

        self.lbl_save_status = QLabel("Status: No unsaved edits")
        v_save.addWidget(self.lbl_save_status)

        btn_save = QPushButton("💾 Save Modified LAS...")
        btn_save.clicked.connect(self.save_modified_las)
        v_save.addWidget(btn_save)

        layout.addWidget(grp_save)

        # 4. Potree Viewer Display Settings
        grp_disp = QGroupBox("Potree Display Controls")
        v_disp = QVBoxLayout(grp_disp)

        h_budget = QHBoxLayout()
        h_budget.addWidget(QLabel("Point Budget:"))
        self.combo_budget = QComboBox()
        self.combo_budget.addItem("1 Million", 1000000)
        self.combo_budget.addItem("2.5 Million", 2500000)
        self.combo_budget.addItem("5 Million", 5000000)
        self.combo_budget.addItem("10 Million", 10000000)
        self.combo_budget.setCurrentIndex(1)
        self.combo_budget.currentIndexChanged.connect(self.on_budget_changed)
        h_budget.addWidget(self.combo_budget)
        v_disp.addLayout(h_budget)

        h_color = QHBoxLayout()
        h_color.addWidget(QLabel("Color By:"))
        self.combo_color = QComboBox()
        self.combo_color.addItem("RGB / True Color", "rgba")
        self.combo_color.addItem("Classification", "classification")
        self.combo_color.addItem("Elevation", "elevation")
        self.combo_color.addItem("Intensity", "intensity")
        self.combo_color.addItem("Tree / ITC", "itc")
        self.combo_color.currentIndexChanged.connect(self.on_color_mode_changed)
        h_color.addWidget(self.combo_color)
        v_disp.addLayout(h_color)

        self.chk_edl = QCheckBox("Eye-Dome Lighting (EDL)")
        self.chk_edl.setChecked(True)
        self.chk_edl.toggled.connect(self.on_edl_toggled)
        v_disp.addWidget(self.chk_edl)

        layout.addWidget(grp_disp)
        layout.addStretch()

        scroll.setWidget(content)
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)
        return container

    # -------------------------------------------------------------------------
    # Dataset Loading & Auto-Conversion
    # -------------------------------------------------------------------------
    def open_potree_file(self):
        """Open an existing Potree metadata.json or cloud.js file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Potree Converted File",
            "",
            "Potree Files (metadata.json cloud.js);;All Files (*)"
        )
        if not file_path:
            return

        folder = os.path.dirname(os.path.abspath(file_path))
        entry_file = os.path.basename(file_path)

        alias = "cloud_" + str(abs(hash(folder)) % 100000)
        self.server.register_pointcloud_dir(alias, folder)

        url = f"http://127.0.0.1:{self.server.port}/clouds/{alias}/{entry_file}"
        self.load_url_in_potree(url)

        # Look for sibling or original LAS file
        base_name = os.path.basename(folder).replace("_converted", "")
        parent_dir = os.path.dirname(folder)
        for ext in [".las", ".laz"]:
            candidate = os.path.join(parent_dir, base_name + ext)
            if os.path.exists(candidate):
                self.load_las_for_editing(candidate)
                break

    def open_las_file(self):
        """Open a raw LAS/LAZ file, check for converted version or convert on the fly."""
        las_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select LAS / LAZ Point Cloud",
            "",
            "LAS/LAZ Files (*.las *.laz);;All Files (*)"
        )
        if not las_path:
            return

        # 1. Load into Python manager for editing
        self.load_las_for_editing(las_path)

        # 2. Check if a converted sibling directory already exists
        parent_dir = os.path.dirname(las_path)
        base_name = os.path.splitext(os.path.basename(las_path))[0]
        converted_dir = os.path.join(parent_dir, base_name + "_converted")

        meta_json = os.path.join(converted_dir, "metadata.json")
        cloud_js = os.path.join(converted_dir, "cloud.js")

        if os.path.exists(meta_json) or os.path.exists(cloud_js):
            entry_file = "metadata.json" if os.path.exists(meta_json) else "cloud.js"
            alias = "cloud_" + str(abs(hash(converted_dir)) % 100000)
            self.server.register_pointcloud_dir(alias, converted_dir)
            url = f"http://127.0.0.1:{self.server.port}/clouds/{alias}/{entry_file}"
            self.load_url_in_potree(url)
        else:
            # Offer auto-conversion
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

    def load_las_for_editing(self, las_path: str):
        try:
            info = self.manager.load_las(las_path)
            self.current_las_path = las_path
            self.lbl_cloud_info.setText(
                f"<b>File:</b> {info['filename']}<br>"
                f"<b>Points:</b> {info['total_points']:,}<br>"
                f"<b>Classes:</b> {info['classes']}<br>"
                f"<b>Has ITC Field:</b> {'Yes' if info['has_itc'] else 'No (auto-created)'}"
            )
            self.status_bar.showMessage(f"Loaded '{info['filename']}' ({info['total_points']:,} points) for editing.")
        except Exception as e:
            QMessageBox.critical(self, "Error Loading LAS", str(e))

    def start_conversion(self, input_las: str, out_dir: str):
        converter_exe = os.path.join(self.potree_dir, "libs", "PotreeConverter2", "PotreeConverter.exe")
        if not os.path.exists(converter_exe):
            QMessageBox.critical(self, "Converter Not Found", f"Could not find PotreeConverter at:\n{converter_exe}")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # indeterminate spinner
        self.status_bar.showMessage("Converting point cloud with PotreeConverter...")

        def run():
            cmd = [converter_exe, input_las, "-o", out_dir]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                alias = "cloud_" + str(abs(hash(out_dir)) % 100000)
                self.server.register_pointcloud_dir(alias, out_dir)
                entry_file = "metadata.json" if os.path.exists(os.path.join(out_dir, "metadata.json")) else "cloud.js"
                url = f"http://127.0.0.1:{self.server.port}/clouds/{alias}/{entry_file}"
                self.load_url_in_potree(url)
            else:
                print("Conversion error:", proc.stderr)

            self.progress_bar.setVisible(False)
            self.status_bar.showMessage("Conversion finished.")

        threading.Thread(target=run, daemon=True).start()

    def load_url_in_potree(self, url: str):
        self.current_potree_url = url
        js = f"window.loadPointCloudUrl('{url}');"
        self.web_view.page().runJavaScript(js)

    # -------------------------------------------------------------------------
    # Selection & Edit Actions
    # -------------------------------------------------------------------------
    def on_selection_changed(self, data: dict):
        count = data.get("count", 0)
        classes = data.get("classes", [])
        itc_vals = data.get("itc_values", [])

        self.lbl_sel_count.setText(f"Selected: {count:,} points")
        self.lbl_sel_classes.setText(f"Classes: {classes if classes else '-'}")
        self.lbl_sel_itc.setText(f"ITC IDs: {itc_vals[:6] if itc_vals else '-'}")
        self.status_bar.showMessage(f"Selected {count:,} points.")

    def on_point_picked(self, data: dict):
        self.status_bar.showMessage(
            f"Picked Point: ({data['x']:.2f}, {data['y']:.2f}, {data['z']:.2f}) | "
            f"Class: {data.get('classification')} | ITC: {data.get('itc')}"
        )

    def on_edit_applied(self, data: dict):
        count = data.get("modified_count", 0)
        action = data.get("action")
        self.lbl_save_status.setText("Status: <span style='color:#f59e0b;'>● Unsaved Changes</span>")
        self.status_bar.showMessage(f"Applied '{action}' to {count:,} points.")

    def apply_assign_itc(self):
        itc_val = self.spin_itc.value()
        count = self.manager.assign_itc(itc_val)
        if count == 0:
            QMessageBox.information(self, "No Selection", "No points are currently selected. Draw a polygon first.")
            return

        self.on_edit_applied({"action": f"assign_itc({itc_val})", "modified_count": count})
        QMessageBox.information(self, "Success", f"Assigned Tree ID (ITC) {itc_val} to {count:,} points.")

    def apply_mark_noise(self):
        count = self.manager.mark_as_noise(noise_itc=9999, class_val=7)
        if count == 0:
            QMessageBox.information(self, "No Selection", "No points are currently selected.")
            return

        self.on_edit_applied({"action": "mark_noise", "modified_count": count})
        QMessageBox.information(self, "Success", f"Marked {count:,} points as Noise (Class 7).")

    def apply_set_classification(self):
        cls_val = self.combo_class.currentData()
        count = self.manager.set_classification(cls_val)
        if count == 0:
            QMessageBox.information(self, "No Selection", "No points are currently selected.")
            return

        self.on_edit_applied({"action": f"class_{cls_val}", "modified_count": count})

    def apply_delete_selection(self):
        reply = QMessageBox.warning(
            self,
            "Confirm Delete",
            f"Delete {len(self.manager.selected_indices):,} selected points from memory?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            count = self.manager.delete_selected_points()
            self.on_edit_applied({"action": "delete", "modified_count": count})
            self.lbl_sel_count.setText("Selected: 0 points")

    def clear_selection(self):
        self.web_view.page().runJavaScript("window.clearSelection();")
        self.manager.selected_indices = []
        self.lbl_sel_count.setText("Selected: 0 points")
        self.lbl_sel_classes.setText("Classes: -")
        self.lbl_sel_itc.setText("ITC IDs: -")

    def save_modified_las(self):
        if not self.manager.las_data:
            QMessageBox.warning(self, "Warning", "No point cloud loaded to save.")
            return

        default_name = os.path.splitext(self.current_las_path or "edited_cloud.las")[0] + "_edited.las"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Modified Point Cloud",
            default_name,
            "LAS Files (*.las);;LAZ Files (*.laz)"
        )
        if not save_path:
            return

        try:
            self.manager.save_las(save_path)
            self.lbl_save_status.setText("Status: <span style='color:#10b981;'>● All changes saved</span>")
            QMessageBox.information(self, "Saved", f"Point cloud successfully saved to:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    # -------------------------------------------------------------------------
    # Viewport Display Controls
    # -------------------------------------------------------------------------
    def on_budget_changed(self):
        budget = self.combo_budget.currentData()
        self.web_view.page().runJavaScript(f"window.setPointBudget({budget});")

    def on_color_mode_changed(self):
        attr = self.combo_color.currentData()
        self.web_view.page().runJavaScript(f"window.setColorAttribute('{attr}');")

    def on_edl_toggled(self, checked: bool):
        val = "true" if checked else "false"
        self.web_view.page().runJavaScript(f"window.setEDLEnabled({val});")
