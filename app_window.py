"""
Main Application Window for Potree Hybrid Viewer.
Combines:
- QWebEngineView hosting the Potree WebGL streaming renderer.
- Native PyQt6 sidebar with LAS/LAZ editing controls, ITC assignment, noise tagging, and point cloud export.
- Top Settings menu and in-viewport HUD controls for Point Type, Point Size, Splat Quality, and Eye-Dome Lighting (EDL).
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
from PyQt6.QtGui import QFont, QIcon, QColor, QAction, QActionGroup
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
        self.resize(1440, 920)

        # Build application menu bar
        self.init_menu_bar()

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
                font-size: 13px;
            }
            QMenu::item {
                padding: 6px 20px 6px 24px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #2563eb;
                color: #ffffff;
            }
            QMenu::separator {
                height: 1px;
                background-color: #334155;
                margin: 4px 6px;
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

        # Proportions: Sidebar ~330px, Viewport remainder
        splitter.setSizes([330, 1110])

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Load a point cloud to begin.")

    # -------------------------------------------------------------------------
    # Menu Bar Setup (Settings, File)
    # -------------------------------------------------------------------------
    def init_menu_bar(self):
        menubar = self.menuBar()

        # File Menu
        menu_file = menubar.addMenu("&File")

        act_open_potree = QAction("📂 Open Potree Converted...", self)
        act_open_potree.triggered.connect(self.open_potree_file)
        menu_file.addAction(act_open_potree)

        act_open_las = QAction("🌲 Open LAS / LAZ Point Cloud...", self)
        act_open_las.triggered.connect(self.open_las_file)
        menu_file.addAction(act_open_las)

        menu_file.addSeparator()

        act_save_las = QAction("💾 Save Modified LAS...", self)
        act_save_las.triggered.connect(self.save_modified_las)
        menu_file.addAction(act_save_las)

        menu_file.addSeparator()

        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self.close)
        menu_file.addAction(act_exit)

        # Settings Menu (Point Type, Size, Splat Quality, EDL, Budget, Colors)
        menu_settings = menubar.addMenu("&Settings")

        # 1. Point Type (Shape)
        menu_point_type = menu_settings.addMenu("Point Type (Shape)")
        self.grp_menu_point_type = QActionGroup(self)
        self.grp_menu_point_type.setExclusive(True)
        for name, key in [("Square", "SQUARE"), ("Circle", "CIRCLE"), ("Paraboloid (Splat)", "PARABOLOID")]:
            act = QAction(name, self, checkable=True)
            if key == "SQUARE":
                act.setChecked(True)
            act.triggered.connect(lambda checked, k=key: self.set_point_shape(k))
            self.grp_menu_point_type.addAction(act)
            menu_point_type.addAction(act)

        # 2. Point Sizing
        menu_point_sizing = menu_settings.addMenu("Point Sizing")
        self.grp_menu_point_sizing = QActionGroup(self)
        self.grp_menu_point_sizing.setExclusive(True)
        for name, key in [("Adaptive", "ADAPTIVE"), ("Fixed", "FIXED"), ("Attenuated", "ATTENUATED")]:
            act = QAction(name, self, checkable=True)
            if key == "ADAPTIVE":
                act.setChecked(True)
            act.triggered.connect(lambda checked, k=key: self.set_point_size_type(k))
            self.grp_menu_point_sizing.addAction(act)
            menu_point_sizing.addAction(act)

        # 3. Point Size Presets
        menu_point_size = menu_settings.addMenu("Point Size")
        self.grp_menu_point_size = QActionGroup(self)
        self.grp_menu_point_size.setExclusive(True)
        for name, val in [("0.5x (Small)", 0.5), ("0.8x", 0.8), ("1.0x (Normal)", 1.0), ("1.5x", 1.5), ("2.0x (Large)", 2.0), ("3.0x (Extra Large)", 3.0)]:
            act = QAction(name, self, checkable=True)
            if val == 1.0:
                act.setChecked(True)
            act.triggered.connect(lambda checked, v=val: self.set_point_size(v))
            self.grp_menu_point_size.addAction(act)
            menu_point_size.addAction(act)

        # 4. Splat Quality
        menu_splat_quality = menu_settings.addMenu("Splat Quality")
        self.grp_menu_splat = QActionGroup(self)
        self.grp_menu_splat.setExclusive(True)
        for name, key in [("Standard Quality", "standard"), ("High Quality (HQ Splats)", "hq")]:
            act = QAction(name, self, checkable=True)
            if key == "standard":
                act.setChecked(True)
            act.triggered.connect(lambda checked, k=key: self.set_splat_quality(k))
            self.grp_menu_splat.addAction(act)
            menu_splat_quality.addAction(act)

        menu_settings.addSeparator()

        # 5. Eye-Dome Lighting (EDL)
        menu_edl = menu_settings.addMenu("Eye-Dome Lighting (EDL)")
        self.act_menu_edl = QAction("Enable Eye-Dome Lighting", self, checkable=True)
        self.act_menu_edl.setChecked(True)
        self.act_menu_edl.toggled.connect(self.set_edl_enabled)
        menu_edl.addAction(self.act_menu_edl)
        menu_edl.addSeparator()

        # EDL Radius Submenu
        menu_edl_radius = menu_edl.addMenu("EDL Radius")
        self.grp_menu_edl_radius = QActionGroup(self)
        self.grp_menu_edl_radius.setExclusive(True)
        for r_val in [0.8, 1.0, 1.4, 2.0, 3.0]:
            act = QAction(f"{r_val:.1f} px", self, checkable=True)
            if r_val == 1.4:
                act.setChecked(True)
            act.triggered.connect(lambda checked, v=r_val: self.set_edl_radius(v))
            self.grp_menu_edl_radius.addAction(act)
            menu_edl_radius.addAction(act)

        # EDL Strength Submenu
        menu_edl_str = menu_edl.addMenu("EDL Strength")
        self.grp_menu_edl_str = QActionGroup(self)
        self.grp_menu_edl_str.setExclusive(True)
        for s_val in [0.4, 0.7, 1.0, 1.5, 2.5]:
            act = QAction(f"{s_val:.1f}", self, checkable=True)
            if s_val == 1.0:
                act.setChecked(True)
            act.triggered.connect(lambda checked, v=s_val: self.set_edl_strength(v))
            self.grp_menu_edl_str.addAction(act)
            menu_edl_str.addAction(act)

        menu_settings.addSeparator()

        # 6. Point Budget
        menu_budget = menu_settings.addMenu("Point Budget")
        self.grp_menu_budget = QActionGroup(self)
        self.grp_menu_budget.setExclusive(True)
        for name, budget in [("1 Million", 1000000), ("2.5 Million", 2500000), ("5 Million", 5000000), ("10 Million", 10000000), ("15 Million", 15000000)]:
            act = QAction(name, self, checkable=True)
            if budget == 2500000:
                act.setChecked(True)
            act.triggered.connect(lambda checked, b=budget: self.set_point_budget(b))
            self.grp_menu_budget.addAction(act)
            menu_budget.addAction(act)

        # 7. Color By
        menu_color = menu_settings.addMenu("Color By")
        self.grp_menu_color = QActionGroup(self)
        self.grp_menu_color.setExclusive(True)
        for name, attr in [("RGB / True Color", "rgba"), ("Classification", "classification"), ("Elevation", "elevation"), ("Intensity", "intensity"), ("Tree / ITC", "itc")]:
            act = QAction(name, self, checkable=True)
            if attr == "rgba":
                act.setChecked(True)
            act.triggered.connect(lambda checked, a=attr: self.set_color_mode(a))
            self.grp_menu_color.addAction(act)
            menu_color.addAction(act)

        menu_settings.addSeparator()

        # 8. Open In-Viewport Settings Menu (HUD)
        act_toggle_hud_settings = QAction("⚙️ Toggle In-Viewport Settings Panel (HUD)", self)
        act_toggle_hud_settings.setShortcut("Ctrl+S")
        act_toggle_hud_settings.triggered.connect(self.toggle_in_viewport_settings)
        menu_settings.addAction(act_toggle_hud_settings)

    # -------------------------------------------------------------------------
    # Sidebar Construction (Data + Editing + Save only)
    # -------------------------------------------------------------------------
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
        self.progress_bar.setRange(0, 0)
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

        self.on_edit_applied({"action": "mark_noise(7, 9999)", "modified_count": count})
        QMessageBox.information(self, "Success", f"Converted {count:,} points to Noise (Class 7).")

    def apply_set_classification(self):
        class_val = self.combo_class.currentData()
        count = self.manager.set_classification(class_val)
        if count == 0:
            QMessageBox.information(self, "No Selection", "No points are currently selected.")
            return

        self.on_edit_applied({"action": f"set_class({class_val})", "modified_count": count})
        QMessageBox.information(self, "Success", f"Updated classification to {class_val} for {count:,} points.")

    def apply_delete_selection(self):
        if len(self.manager.selected_indices) == 0:
            QMessageBox.information(self, "No Selection", "No points are currently selected.")
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
            self.lbl_sel_count.setText("Selected: 0 points")
            self.web_view.page().runJavaScript("window.clearSelection(false);")

    def clear_selection(self):
        self.web_view.page().runJavaScript("window.clearSelection(true);")
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
    # Render & Visual Settings Actions (Invoked from top Settings Menu)
    # -------------------------------------------------------------------------
    def set_point_shape(self, shape: str):
        self.web_view.page().runJavaScript(f"window.setPointShape('{shape}');")

    def set_point_size_type(self, stype: str):
        self.web_view.page().runJavaScript(f"window.setPointSizeType('{stype}');")

    def set_point_size(self, size: float):
        self.web_view.page().runJavaScript(f"window.setPointSize({size:.2f});")

    def set_splat_quality(self, quality: str):
        self.web_view.page().runJavaScript(f"window.setSplatQuality('{quality}');")

    def set_edl_enabled(self, enabled: bool):
        val = "true" if enabled else "false"
        self.web_view.page().runJavaScript(f"window.setEDLEnabled({val});")

    def set_edl_radius(self, radius: float):
        self.web_view.page().runJavaScript(f"window.setEDLRadius({radius:.1f});")

    def set_edl_strength(self, strength: float):
        self.web_view.page().runJavaScript(f"window.setEDLStrength({strength:.1f});")

    def set_point_budget(self, budget: int):
        self.web_view.page().runJavaScript(f"window.setPointBudget({budget});")

    def set_color_mode(self, attr: str):
        self.web_view.page().runJavaScript(f"window.setColorAttribute('{attr}');")

    def toggle_in_viewport_settings(self):
        self.web_view.page().runJavaScript("window.toggleSettingsMenu();")
