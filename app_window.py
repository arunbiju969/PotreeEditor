"""
Main Application Window for Potree Hybrid Viewer.
Combines:
- QWebEngineView hosting the Potree WebGL streaming renderer.
- Native PyQt6 sidebar with LAS/LAZ editing controls, ITC assignment, noise tagging, and point cloud export.
- Dedicated Settings menu and sidebar controls for Point Type, Point Size, Splat Quality, and Eye-Dome Lighting (EDL).
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
    QFrame, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QSlider,
    QGroupBox, QStatusBar, QScrollArea
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
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #0f172a;
                color: #f8fafc;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 5px 8px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #334155;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #2563eb;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #60a5fa;
                width: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
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

        # Set initial splitter proportions: Sidebar 360px, Viewport remainder
        splitter.setSizes([360, 1080])

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Load a point cloud to begin.")

    # -------------------------------------------------------------------------
    # Menu Bar Setup
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

        # Settings Menu (Point Type, Size, Splat Quality, EDL)
        menu_settings = menubar.addMenu("&Settings")

        # Submenu: Point Type (Shape)
        menu_point_type = menu_settings.addMenu("Point Type (Shape)")
        self.grp_menu_point_type = QActionGroup(self)
        self.grp_menu_point_type.setExclusive(True)

        for name, key in [("Square", "SQUARE"), ("Circle", "CIRCLE"), ("Paraboloid (Splat)", "PARABOLOID")]:
            act = QAction(name, self, checkable=True)
            if key == "SQUARE":
                act.setChecked(True)
            act.setData(key)
            act.triggered.connect(lambda checked, k=key: self._on_menu_point_type_triggered(k))
            self.grp_menu_point_type.addAction(act)
            menu_point_type.addAction(act)

        # Submenu: Point Sizing
        menu_point_sizing = menu_settings.addMenu("Point Sizing")
        self.grp_menu_point_sizing = QActionGroup(self)
        self.grp_menu_point_sizing.setExclusive(True)

        for name, key in [("Adaptive", "ADAPTIVE"), ("Fixed", "FIXED"), ("Attenuated", "ATTENUATED")]:
            act = QAction(name, self, checkable=True)
            if key == "ADAPTIVE":
                act.setChecked(True)
            act.setData(key)
            act.triggered.connect(lambda checked, k=key: self._on_menu_size_type_triggered(k))
            self.grp_menu_point_sizing.addAction(act)
            menu_point_sizing.addAction(act)

        # Submenu: Splat Quality
        menu_splat_quality = menu_settings.addMenu("Splat Quality")
        self.grp_menu_splat = QActionGroup(self)
        self.grp_menu_splat.setExclusive(True)

        for name, key in [("Standard Quality", "standard"), ("High Quality (HQ Splats)", "hq")]:
            act = QAction(name, self, checkable=True)
            if key == "standard":
                act.setChecked(True)
            act.setData(key)
            act.triggered.connect(lambda checked, k=key: self._on_menu_splat_triggered(k))
            self.grp_menu_splat.addAction(act)
            menu_splat_quality.addAction(act)

        menu_settings.addSeparator()

        # Eye-Dome Lighting (EDL) Toggle
        self.act_menu_edl = QAction("Enable Eye-Dome Lighting (EDL)", self, checkable=True)
        self.act_menu_edl.setChecked(True)
        self.act_menu_edl.toggled.connect(self._on_menu_edl_toggled)
        menu_settings.addAction(self.act_menu_edl)

        menu_settings.addSeparator()

        # Toggle In-Viewport Settings Menu
        act_toggle_hud_settings = QAction("⚙️ Toggle In-Viewport Settings Panel", self)
        act_toggle_hud_settings.setShortcut("Ctrl+S")
        act_toggle_hud_settings.triggered.connect(self.toggle_in_viewport_settings)
        menu_settings.addAction(act_toggle_hud_settings)

    def _on_menu_point_type_triggered(self, key: str):
        idx = self.combo_point_type.findData(key)
        if idx >= 0:
            self.combo_point_type.setCurrentIndex(idx)

    def _on_menu_size_type_triggered(self, key: str):
        idx = self.combo_size_type.findData(key)
        if idx >= 0:
            self.combo_size_type.setCurrentIndex(idx)

    def _on_menu_splat_triggered(self, key: str):
        idx = self.combo_splat_quality.findData(key)
        if idx >= 0:
            self.combo_splat_quality.setCurrentIndex(idx)

    def _on_menu_edl_toggled(self, checked: bool):
        self.chk_edl.setChecked(checked)

    # -------------------------------------------------------------------------
    # Sidebar Construction
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

        # 4. Render & Visual Settings (Point Type, Size, Splat Quality, EDL)
        grp_settings = QGroupBox("⚙️ Render & Visual Settings")
        v_settings = QVBoxLayout(grp_settings)

        # Point Type (Shape)
        h_ptype = QHBoxLayout()
        h_ptype.addWidget(QLabel("Point Type:"))
        self.combo_point_type = QComboBox()
        self.combo_point_type.addItem("Square", "SQUARE")
        self.combo_point_type.addItem("Circle", "CIRCLE")
        self.combo_point_type.addItem("Paraboloid (Splat)", "PARABOLOID")
        self.combo_point_type.currentIndexChanged.connect(self.on_point_type_changed)
        h_ptype.addWidget(self.combo_point_type)
        v_settings.addLayout(h_ptype)

        # Point Sizing (Adaptive / Fixed / Attenuated)
        h_stype = QHBoxLayout()
        h_stype.addWidget(QLabel("Point Sizing:"))
        self.combo_size_type = QComboBox()
        self.combo_size_type.addItem("Adaptive", "ADAPTIVE")
        self.combo_size_type.addItem("Fixed", "FIXED")
        self.combo_size_type.addItem("Attenuated", "ATTENUATED")
        self.combo_size_type.currentIndexChanged.connect(self.on_size_type_changed)
        h_stype.addWidget(self.combo_size_type)
        v_settings.addLayout(h_stype)

        # Point Size (Slider + DoubleSpinBox)
        v_psize = QVBoxLayout()
        h_psize_hdr = QHBoxLayout()
        h_psize_hdr.addWidget(QLabel("Point Size:"))
        self.spin_point_size = QDoubleSpinBox()
        self.spin_point_size.setRange(0.1, 5.0)
        self.spin_point_size.setSingleStep(0.1)
        self.spin_point_size.setValue(1.0)
        self.spin_point_size.valueChanged.connect(self.on_point_size_spin_changed)
        h_psize_hdr.addWidget(self.spin_point_size)
        v_psize.addLayout(h_psize_hdr)

        self.slider_point_size = QSlider(Qt.Orientation.Horizontal)
        self.slider_point_size.setRange(1, 50)  # /10 => 0.1 to 5.0
        self.slider_point_size.setValue(10)
        self.slider_point_size.valueChanged.connect(self.on_point_size_slider_changed)
        v_psize.addWidget(self.slider_point_size)
        v_settings.addLayout(v_psize)

        # Splat Quality Control
        h_splat = QHBoxLayout()
        h_splat.addWidget(QLabel("Splat Quality:"))
        self.combo_splat_quality = QComboBox()
        self.combo_splat_quality.addItem("Standard", "standard")
        self.combo_splat_quality.addItem("High Quality (HQ)", "hq")
        self.combo_splat_quality.currentIndexChanged.connect(self.on_splat_quality_changed)
        h_splat.addWidget(self.combo_splat_quality)
        v_settings.addLayout(h_splat)

        # Eye-Dome Lighting (EDL) Group
        self.chk_edl = QCheckBox("Enable Eye-Dome Lighting (EDL)")
        self.chk_edl.setChecked(True)
        self.chk_edl.toggled.connect(self.on_edl_toggled)
        v_settings.addWidget(self.chk_edl)

        self.edl_container = QWidget()
        v_edl = QVBoxLayout(self.edl_container)
        v_edl.setContentsMargins(6, 4, 6, 4)
        v_edl.setSpacing(6)

        # EDL Radius (0.5 to 5.0)
        h_rad_hdr = QHBoxLayout()
        h_rad_hdr.addWidget(QLabel("EDL Radius:"))
        self.lbl_edl_radius = QLabel("1.4")
        self.lbl_edl_radius.setStyleSheet("font-weight:bold; color:#38bdf8;")
        h_rad_hdr.addWidget(self.lbl_edl_radius)
        v_edl.addLayout(h_rad_hdr)
        self.slider_edl_radius = QSlider(Qt.Orientation.Horizontal)
        self.slider_edl_radius.setRange(5, 50)  # /10 => 0.5 to 5.0
        self.slider_edl_radius.setValue(14)
        self.slider_edl_radius.valueChanged.connect(self.on_edl_radius_changed)
        v_edl.addWidget(self.slider_edl_radius)

        # EDL Strength (0.1 to 5.0)
        h_str_hdr = QHBoxLayout()
        h_str_hdr.addWidget(QLabel("EDL Strength:"))
        self.lbl_edl_strength = QLabel("1.0")
        self.lbl_edl_strength.setStyleSheet("font-weight:bold; color:#38bdf8;")
        h_str_hdr.addWidget(self.lbl_edl_strength)
        v_edl.addLayout(h_str_hdr)
        self.slider_edl_strength = QSlider(Qt.Orientation.Horizontal)
        self.slider_edl_strength.setRange(1, 50)  # /10 => 0.1 to 5.0
        self.slider_edl_strength.setValue(10)
        self.slider_edl_strength.valueChanged.connect(self.on_edl_strength_changed)
        v_edl.addWidget(self.slider_edl_strength)

        # EDL Opacity (0.1 to 1.0)
        h_op_hdr = QHBoxLayout()
        h_op_hdr.addWidget(QLabel("EDL Opacity:"))
        self.lbl_edl_opacity = QLabel("1.00")
        self.lbl_edl_opacity.setStyleSheet("font-weight:bold; color:#38bdf8;")
        h_op_hdr.addWidget(self.lbl_edl_opacity)
        v_edl.addLayout(h_op_hdr)
        self.slider_edl_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_edl_opacity.setRange(10, 100)  # /100 => 0.1 to 1.0
        self.slider_edl_opacity.setValue(100)
        self.slider_edl_opacity.valueChanged.connect(self.on_edl_opacity_changed)
        v_edl.addWidget(self.slider_edl_opacity)

        v_settings.addWidget(self.edl_container)

        # Point Budget
        h_budget = QHBoxLayout()
        h_budget.addWidget(QLabel("Point Budget:"))
        self.combo_budget = QComboBox()
        self.combo_budget.addItem("1 Million", 1000000)
        self.combo_budget.addItem("2.5 Million", 2500000)
        self.combo_budget.addItem("5 Million", 5000000)
        self.combo_budget.addItem("10 Million", 10000000)
        self.combo_budget.addItem("15 Million", 15000000)
        self.combo_budget.setCurrentIndex(1)
        self.combo_budget.currentIndexChanged.connect(self.on_budget_changed)
        h_budget.addWidget(self.combo_budget)
        v_settings.addLayout(h_budget)

        # Color By
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
        v_settings.addLayout(h_color)

        # Button to toggle in-viewport floating menu
        btn_toggle_hud = QPushButton("🎛️ Toggle In-Viewport Settings Menu")
        btn_toggle_hud.setObjectName("btnSecondary")
        btn_toggle_hud.clicked.connect(self.toggle_in_viewport_settings)
        v_settings.addWidget(btn_toggle_hud)

        layout.addWidget(grp_settings)
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
    # Render & Visual Settings Controls (Point Type, Size, Splat, EDL)
    # -------------------------------------------------------------------------
    def on_point_type_changed(self):
        shape = self.combo_point_type.currentData()
        self.web_view.page().runJavaScript(f"window.setPointShape('{shape}');")
        # Sync menu action checkmark
        for act in self.grp_menu_point_type.actions():
            if act.data() == shape:
                act.setChecked(True)
                break

    def on_size_type_changed(self):
        stype = self.combo_size_type.currentData()
        self.web_view.page().runJavaScript(f"window.setPointSizeType('{stype}');")
        for act in self.grp_menu_point_sizing.actions():
            if act.data() == stype:
                act.setChecked(True)
                break

    def on_point_size_slider_changed(self, val: int):
        fval = val / 10.0
        self.spin_point_size.blockSignals(True)
        self.spin_point_size.setValue(fval)
        self.spin_point_size.blockSignals(False)
        self.web_view.page().runJavaScript(f"window.setPointSize({fval:.2f});")

    def on_point_size_spin_changed(self, fval: float):
        slider_val = int(round(fval * 10))
        self.slider_point_size.blockSignals(True)
        self.slider_point_size.setValue(slider_val)
        self.slider_point_size.blockSignals(False)
        self.web_view.page().runJavaScript(f"window.setPointSize({fval:.2f});")

    def on_splat_quality_changed(self):
        quality = self.combo_splat_quality.currentData()
        self.web_view.page().runJavaScript(f"window.setSplatQuality('{quality}');")
        for act in self.grp_menu_splat.actions():
            if act.data() == quality:
                act.setChecked(True)
                break

    def on_edl_toggled(self, checked: bool):
        self.edl_container.setEnabled(checked)
        self.act_menu_edl.blockSignals(True)
        self.act_menu_edl.setChecked(checked)
        self.act_menu_edl.blockSignals(False)
        val = "true" if checked else "false"
        self.web_view.page().runJavaScript(f"window.setEDLEnabled({val});")

    def on_edl_radius_changed(self, val: int):
        radius = val / 10.0
        self.lbl_edl_radius.setText(f"{radius:.1f}")
        self.web_view.page().runJavaScript(f"window.setEDLRadius({radius:.1f});")

    def on_edl_strength_changed(self, val: int):
        strength = val / 10.0
        self.lbl_edl_strength.setText(f"{strength:.1f}")
        self.web_view.page().runJavaScript(f"window.setEDLStrength({strength:.1f});")

    def on_edl_opacity_changed(self, val: int):
        opacity = val / 100.0
        self.lbl_edl_opacity.setText(f"{opacity:.2f}")
        self.web_view.page().runJavaScript(f"window.setEDLOpacity({opacity:.2f});")

    def on_budget_changed(self):
        budget = self.combo_budget.currentData()
        self.web_view.page().runJavaScript(f"window.setPointBudget({budget});")

    def on_color_mode_changed(self):
        attr = self.combo_color.currentData()
        self.web_view.page().runJavaScript(f"window.setColorAttribute('{attr}');")

    def toggle_in_viewport_settings(self):
        self.web_view.page().runJavaScript("window.toggleSettingsMenu();")
