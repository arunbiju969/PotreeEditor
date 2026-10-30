"""
Main Application Window for Potree Hybrid Viewer.
Combines:
- QWebEngineView hosting the Potree WebGL streaming renderer.
- Native PyQt6 sidebar with Optimistic Visual Staging, ITC assignment, noise tagging, and deferred LAS commit.
- Top Ribbon Category Bar (File, Point, Settings) with quick tools and HUD integration.
- High-performance streaming deferred commit for massive 200M+ point clouds.
"""

import os
import sys
import json
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox, QProgressBar, QSplitter,
    QFrame, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QSlider, QGroupBox, QStatusBar,
    QListWidget, QListWidgetItem,
    QScrollArea, QStackedWidget, QButtonGroup, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QProgressDialog, QSizePolicy, QTabWidget, QGridLayout
)
from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QObject, QThread, QTimer
from PyQt6.QtGui import QFont, QIcon, QColor, QAction, QActionGroup, QPixmap
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings

from pointcloud_manager import PointCloudManager
from server import HybridServer


class BridgeSignaler(QObject):
    selection_changed = pyqtSignal(dict)
    point_picked = pyqtSignal(dict)
    edit_applied = pyqtSignal(dict)
    edit_staged = pyqtSignal(dict)
    edit_removed = pyqtSignal(dict)
    edits_cleared = pyqtSignal(dict)


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


class MainWindow(QMainWindow):
    def __init__(self, server: HybridServer, manager: PointCloudManager, potree_dir: str):
        super().__init__()
        self.server = server
        self.manager = manager
        self.potree_dir = potree_dir

        self.current_las_path: Optional[str] = None
        self.current_potree_url: Optional[str] = None
        self.commit_dialog: Optional[QProgressDialog] = None
        self.commit_worker: Optional[CommitWorker] = None

        self.signaler = BridgeSignaler()
        self.signaler.selection_changed.connect(self.on_selection_changed)
        self.signaler.point_picked.connect(self.on_point_picked)
        self.signaler.edit_applied.connect(self.on_edit_applied)
        self.signaler.edit_staged.connect(self.on_edit_staged)
        self.signaler.edit_removed.connect(self.on_edit_removed)
        self.signaler.edits_cleared.connect(self.on_edits_cleared)

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
        elif event_name == "edit_staged":
            self.signaler.edit_staged.emit(data)
        elif event_name == "edit_removed":
            self.signaler.edit_removed.emit(data)
        elif event_name == "edits_cleared":
            self.signaler.edits_cleared.emit(data)

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
                border-radius: 8px;
                margin-top: 14px;
                padding-top: 12px;
                font-weight: bold;
                color: #93c5fd;
                background-color: #161b22;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
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
                min-height: 68px;
                max-height: 68px;
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
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 6px;
                gridline-color: #21262d;
                color: #c9d1d9;
                font-size: 12px;
            }
            QTableWidget::item:selected {
                background-color: #1f293d;
                color: #58a6ff;
            }
            QHeaderView::section {
                background-color: #161b22;
                color: #8b949e;
                font-size: 11px;
                font-weight: 600;
                border: none;
                border-bottom: 1px solid #30363d;
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

        sidebar = self.create_sidebar()
        sidebar.setMinimumWidth(340)
        sidebar.setMaximumWidth(450)

        # QWebEngineView setup
        self.web_view = QWebEngineView()
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)

        # Point WebEngine to the local hybrid server
        viewer_url = f"http://127.0.0.1:{self.server.port}/index.html"
        self.web_view.load(QUrl(viewer_url))

        splitter.addWidget(sidebar)
        splitter.addWidget(self.web_view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        central_layout.addWidget(splitter, 1)
        central_layout.setStretchFactor(ribbon_bar, 0)
        central_layout.setStretchFactor(splitter, 1)
        self.setCentralWidget(central_widget)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Open a point cloud or Potree dataset to begin.")

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

        # The staged-tree card is built with the layer overlays, because that is where
        # the overlay table lives, but the work belongs with forestry -- so re-parent it
        # into the Forestry tab once both tabs exist.  addWidget moves the widget rather
        # than copying it, so the Layers tab simply stops holding it.  The tab is a
        # QScrollArea, whose own layout() is None -- the cards live on its content
        # widget -- which is why the first attempt at this silently did nothing.
        forestry_tab = self.sidebar_tabs.widget(1)
        forestry_content = (forestry_tab.widget() if isinstance(forestry_tab, QScrollArea)
                            else forestry_tab)
        forestry_layout = forestry_content.layout() if forestry_content is not None else None
        if forestry_layout is not None and hasattr(self, "grp_staged_card"):
            forestry_layout.addWidget(self.grp_staged_card)

        layout.addWidget(self.sidebar_tabs)
        return container

    def create_layer_manager_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        v_main = QVBoxLayout(content)
        v_main.setSpacing(10)
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
        grp_pc = QGroupBox("Point Cloud Layer")
        v_pc = QVBoxLayout(grp_pc)
        v_pc.setSpacing(6)

        h_pc_title = QHBoxLayout()
        self.chk_layer_pc = QCheckBox("Active Point Cloud")
        self.chk_layer_pc.setChecked(True)
        self.chk_layer_pc.setStyleSheet("font-weight: 600; color: #e2e8f0;")
        self.chk_layer_pc.toggled.connect(self.toggle_pointcloud_layer)
        h_pc_title.addWidget(self.chk_layer_pc)
        h_pc_title.addStretch()
        v_pc.addLayout(h_pc_title)

        # Layer list.  The clouds themselves are the layers (potree's scene list is the
        # manager), so every loaded cloud gets a row: the checkbox is pc.visible, and the
        # selected row is the layer the colour, size and edit controls drive.
        self.layer_list = QListWidget()
        self.layer_list.setMaximumHeight(112)
        self.layer_list.itemSelectionChanged.connect(self._on_layer_row_selected)
        self.layer_list.itemChanged.connect(self._on_layer_row_checked)
        v_pc.addWidget(self.layer_list)

        h_layer_buttons = QHBoxLayout()
        btn_remove_layer = QPushButton("Remove Layer")
        btn_remove_layer.clicked.connect(self._on_layer_remove_clicked)
        btn_refresh_layers = QPushButton("Refresh Layers")
        btn_refresh_layers.clicked.connect(lambda: self.refresh_layers(0))
        h_layer_buttons.addWidget(btn_remove_layer)
        h_layer_buttons.addWidget(btn_refresh_layers)
        h_layer_buttons.addStretch()
        v_pc.addLayout(h_layer_buttons)

        # Guards the row rebuild: re-adding rows fires selection and check signals, and
        # those must not bounce back into the viewer as user edits.
        self._layer_sync_guard = False

        self.lbl_layer_cloud_name = QLabel("No point cloud loaded")
        self.lbl_layer_cloud_name.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.lbl_layer_cloud_name.setWordWrap(True)
        v_pc.addWidget(self.lbl_layer_cloud_name)

        self.lbl_layer_cloud_pts = QLabel("- points")
        self.lbl_layer_cloud_pts.setStyleSheet("color: #64748b; font-size: 11px;")
        v_pc.addWidget(self.lbl_layer_cloud_pts)

        # Quick Controls inside Point Cloud card
        h_color = QHBoxLayout()
        lbl_c = QLabel("Color Mode:")
        lbl_c.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.combo_layer_color = QComboBox()
        self.combo_layer_color.addItem("RGBA (True Color)", "rgba")
        self.combo_layer_color.addItem("ASPRS Classification", "classification")
        self.combo_layer_color.addItem("Intensity", "intensity")
        self.combo_layer_color.addItem("Elevation", "elevation")
        self.combo_layer_color.currentIndexChanged.connect(self._on_layer_color_changed)
        h_color.addWidget(lbl_c)
        h_color.addWidget(self.combo_layer_color)
        v_pc.addLayout(h_color)

        h_size = QHBoxLayout()
        lbl_s = QLabel("Point Size:")
        lbl_s.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.slider_layer_size = QSlider(Qt.Orientation.Horizontal)
        self.slider_layer_size.setRange(1, 40)
        self.slider_layer_size.setValue(10)
        self.lbl_layer_size_val = QLabel("1.00")
        self.lbl_layer_size_val.setStyleSheet("color: #38bdf8; font-size: 11px; min-width: 28px;")
        self.slider_layer_size.valueChanged.connect(self._on_layer_size_slider_changed)
        h_size.addWidget(lbl_s)
        h_size.addWidget(self.slider_layer_size)
        h_size.addWidget(self.lbl_layer_size_val)
        v_pc.addLayout(h_size)

        v_main.addWidget(grp_pc)

        # 3. Staged Delineations Layer Card
        grp_staged = QGroupBox("Staged Trees & Edits")
        self.grp_staged_card = grp_staged
        v_staged = QVBoxLayout(grp_staged)
        v_staged.setSpacing(6)

        h_staged_title = QHBoxLayout()
        self.chk_layer_staged = QCheckBox("Staged Overlays (3D)")
        self.chk_layer_staged.setChecked(True)
        self.chk_layer_staged.setStyleSheet("font-weight: 600; color: #e2e8f0;")
        self.chk_layer_staged.toggled.connect(self.toggle_staged_layer)
        h_staged_title.addWidget(self.chk_layer_staged)
        h_staged_title.addStretch()

        lbl_staged_badge = QLabel("STAGED")
        lbl_staged_badge.setStyleSheet("background-color: #064e3b; color: #10b981; font-size: 10px; font-weight: bold; padding: 2px 5px; border-radius: 3px;")
        h_staged_title.addWidget(lbl_staged_badge)
        v_staged.addLayout(h_staged_title)

        self.lbl_layer_staged_count = QLabel("0 staged edits (ready to commit)")
        self.lbl_layer_staged_count.setStyleSheet("color: #10b981; font-weight: 500; font-size: 11px;")
        v_staged.addWidget(self.lbl_layer_staged_count)

        # Staged items table with individual visibility, swatches, labels, delete, and double-click zoom
        self.table_layer_staged = QTableWidget()
        self.table_layer_staged.setColumnCount(4)
        self.table_layer_staged.setHorizontalHeaderLabels(["Vis", "Item", "Action", ""])
        self.table_layer_staged.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table_layer_staged.setColumnWidth(0, 32)
        self.table_layer_staged.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table_layer_staged.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_layer_staged.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self.table_layer_staged.setColumnWidth(3, 30)
        self.table_layer_staged.setMaximumHeight(180)
        self.table_layer_staged.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_layer_staged.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_layer_staged.cellDoubleClicked.connect(self._on_layer_staged_double_clicked)
        v_staged.addWidget(self.table_layer_staged)

        lbl_hint = QLabel("Double-click row to zoom camera to tree.")
        lbl_hint.setStyleSheet("color: #64748b; font-size: 10px; font-style: italic;")
        v_staged.addWidget(lbl_hint)

        h_staged_actions = QHBoxLayout()
        btn_commit_staged = QPushButton("Commit to LAS/LAZ...")
        btn_commit_staged.setObjectName("btnSuccess")
        btn_commit_staged.setStyleSheet("padding: 4px 8px; font-size: 11px;")
        btn_commit_staged.clicked.connect(self.commit_staged_edits_action)

        btn_clear_staged = QPushButton("Clear All")
        btn_clear_staged.setObjectName("btnDanger")
        btn_clear_staged.setStyleSheet("padding: 4px 8px; font-size: 11px;")
        btn_clear_staged.clicked.connect(self.clear_all_staged_edits)

        h_staged_actions.addWidget(btn_commit_staged)
        h_staged_actions.addWidget(btn_clear_staged)
        v_staged.addLayout(h_staged_actions)

        v_main.addWidget(grp_staged)

        # 4. Picked Point Highlight Layer Card
        grp_pick = QGroupBox("Picked Point Highlight")
        v_pick = QVBoxLayout(grp_pick)
        v_pick.setSpacing(6)

        h_pick_title = QHBoxLayout()
        self.chk_layer_pick = QCheckBox("Point Reticle & Badge")
        self.chk_layer_pick.setChecked(True)
        self.chk_layer_pick.setStyleSheet("font-weight: 600; color: #e2e8f0;")
        self.chk_layer_pick.toggled.connect(self.toggle_pick_layer)
        h_pick_title.addWidget(self.chk_layer_pick)
        h_pick_title.addStretch()
        v_pick.addLayout(h_pick_title)

        self.lbl_layer_pick_coords = QLabel("Position: None")
        self.lbl_layer_pick_coords.setStyleSheet("color: #94a3b8; font-size: 11px; font-family: monospace;")
        v_pick.addWidget(self.lbl_layer_pick_coords)

        self.lbl_layer_pick_attr = QLabel("Class: - | Tree ID: - | Intensity: -")
        self.lbl_layer_pick_attr.setStyleSheet("color: #94a3b8; font-size: 11px;")
        v_pick.addWidget(self.lbl_layer_pick_attr)

        btn_clear_pick = QPushButton("Clear Pick Highlight")
        btn_clear_pick.setObjectName("btnSecondary")
        btn_clear_pick.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        btn_clear_pick.clicked.connect(self.clear_picked_highlight)
        v_pick.addWidget(btn_clear_pick)

        v_main.addWidget(grp_pick)

        # 5. Active Delineation Boundary Layer Card
        grp_sel = QGroupBox("Active Delineation Boundary")
        v_sel = QVBoxLayout(grp_sel)
        v_sel.setSpacing(6)

        h_sel_title = QHBoxLayout()
        self.chk_layer_sel = QCheckBox("Active Selection Boundary")
        self.chk_layer_sel.setChecked(True)
        self.chk_layer_sel.setStyleSheet("font-weight: 600; color: #e2e8f0;")
        self.chk_layer_sel.toggled.connect(self.toggle_sel_layer)
        h_sel_title.addWidget(self.chk_layer_sel)
        h_sel_title.addStretch()

        lbl_sel_badge = QLabel("SELECTION")
        lbl_sel_badge.setStyleSheet("background-color: #172554; color: #60a5fa; font-size: 10px; font-weight: bold; padding: 2px 5px; border-radius: 3px;")
        h_sel_title.addWidget(lbl_sel_badge)
        v_sel.addLayout(h_sel_title)

        self.lbl_layer_sel_info = QLabel("Selection: None")
        self.lbl_layer_sel_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        v_sel.addWidget(self.lbl_layer_sel_info)

        h_sel_btns = QHBoxLayout()
        btn_clear_boundary = QPushButton("Clear Boundary")
        btn_clear_boundary.setObjectName("btnSecondary")
        btn_clear_boundary.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        btn_clear_boundary.clicked.connect(self.clear_selection)
        h_sel_btns.addWidget(btn_clear_boundary)

        btn_go_stage = QPushButton("Stage Tree ID...")
        btn_go_stage.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        btn_go_stage.clicked.connect(lambda: self.sidebar_tabs.setCurrentIndex(1))
        h_sel_btns.addWidget(btn_go_stage)

        v_sel.addLayout(h_sel_btns)
        v_main.addWidget(grp_sel)

        v_main.addStretch()
        scroll.setWidget(content)
        return scroll

    def create_delineation_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # 1. Active Selection Group
        grp_edit = QGroupBox("Active Delineation / Selection")
        v_edit = QVBoxLayout(grp_edit)

        self.lbl_sel_count = QLabel("Selection: None")
        self.lbl_sel_count.setStyleSheet("font-weight: bold; color: #60a5fa;")
        v_edit.addWidget(self.lbl_sel_count)

        self.lbl_sel_classes = QLabel("Classes: -")
        v_edit.addWidget(self.lbl_sel_classes)

        self.lbl_sel_itc = QLabel("ITC IDs: -")
        v_edit.addWidget(self.lbl_sel_itc)

        # Action: Assign Tree ID (ITC)
        h_itc = QHBoxLayout()
        h_itc.addWidget(QLabel("Tree ID:"))
        self.spin_itc = QSpinBox()
        self.spin_itc.setRange(1, 999999)
        self.spin_itc.setValue(1)
        h_itc.addWidget(self.spin_itc)
        v_edit.addLayout(h_itc)

        btn_assign_itc = QPushButton("Stage Tree ID to Polygon")
        btn_assign_itc.clicked.connect(self.apply_assign_itc)
        v_edit.addWidget(btn_assign_itc)

        # Action: Mark as Noise
        btn_noise = QPushButton("Stage Selection as Noise (Class 7)")
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

        btn_set_class = QPushButton("Stage Classification")
        btn_set_class.setObjectName("btnSecondary")
        btn_set_class.clicked.connect(self.apply_set_classification)
        v_edit.addWidget(btn_set_class)

        btn_clear = QPushButton("Clear Current Delineation")
        btn_clear.setObjectName("btnSecondary")
        btn_clear.clicked.connect(self.clear_selection)
        v_edit.addWidget(btn_clear)

        layout.addWidget(grp_edit)

        # 2. Optimistic Visual Staging & Deferred Commit Group
        grp_staged = QGroupBox("Staged Edits (Pending Commit)")
        v_staged = QVBoxLayout(grp_staged)

        self.lbl_staged_count = QLabel("Staged Edits: 0 (Ready to Commit)")
        self.lbl_staged_count.setStyleSheet("font-weight: bold; color: #10b981;")
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
        btn_commit.clicked.connect(self.commit_staged_edits_action)
        v_staged.addWidget(btn_commit)

        btn_clear_staged = QPushButton("Clear All Staged Edits")
        btn_clear_staged.setObjectName("btnDanger")
        btn_clear_staged.clicked.connect(self.clear_all_staged_edits)
        v_staged.addWidget(btn_clear_staged)

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
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # 1. Dataset Management Group
        grp_data = QGroupBox("Point Cloud Dataset")
        v_data = QVBoxLayout(grp_data)

        self.lbl_cloud_info = QLabel("No point cloud loaded.")
        self.lbl_cloud_info.setWordWrap(True)
        v_data.addWidget(self.lbl_cloud_info)

        btn_open_potree = QPushButton("Open Potree Converted...")
        btn_open_potree.clicked.connect(self.open_potree_file)
        v_data.addWidget(btn_open_potree)

        btn_open_las = QPushButton("Open LAS / LAZ (Header Only)...")
        btn_open_las.setObjectName("btnSecondary")
        btn_open_las.clicked.connect(self.open_las_file)
        v_data.addWidget(btn_open_las)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        v_data.addWidget(self.progress_bar)

        layout.addWidget(grp_data)

        # 2. Save Status & Export
        grp_save = QGroupBox("Export & Status")
        v_save = QVBoxLayout(grp_save)

        self.lbl_save_status = QLabel("Status: <span style='color:#10b981;'>Ready</span>")
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
        self.web_view.page().runJavaScript(f"window.setPointCloudVisibility({str(visible).lower()});")

    def toggle_staged_layer(self, visible: bool):
        if hasattr(self, "chk_layer_staged"):
            self.chk_layer_staged.blockSignals(True)
            self.chk_layer_staged.setChecked(visible)
            self.chk_layer_staged.blockSignals(False)
        if hasattr(self, "ribbon_chk_staged"):
            self.ribbon_chk_staged.blockSignals(True)
            self.ribbon_chk_staged.setChecked(visible)
            self.ribbon_chk_staged.blockSignals(False)
        self.web_view.page().runJavaScript(f"window.setStagedAllVisibility({str(visible).lower()});")

    def toggle_pick_layer(self, visible: bool):
        if hasattr(self, "chk_layer_pick"):
            self.chk_layer_pick.blockSignals(True)
            self.chk_layer_pick.setChecked(visible)
            self.chk_layer_pick.blockSignals(False)
        if hasattr(self, "ribbon_chk_pick"):
            self.ribbon_chk_pick.blockSignals(True)
            self.ribbon_chk_pick.setChecked(visible)
            self.ribbon_chk_pick.blockSignals(False)
        self.web_view.page().runJavaScript(f"window.setHighlightVisibility({str(visible).lower()});")

    def toggle_sel_layer(self, visible: bool):
        if hasattr(self, "chk_layer_sel"):
            self.chk_layer_sel.blockSignals(True)
            self.chk_layer_sel.setChecked(visible)
            self.chk_layer_sel.blockSignals(False)
        if hasattr(self, "ribbon_chk_sel"):
            self.ribbon_chk_sel.blockSignals(True)
            self.ribbon_chk_sel.setChecked(visible)
            self.ribbon_chk_sel.blockSignals(False)
        self.web_view.page().runJavaScript(f"window.setSelectionVisibility({str(visible).lower()});")

    def show_all_layers(self):
        self.toggle_pointcloud_layer(True)
        self.toggle_staged_layer(True)
        self.toggle_pick_layer(True)
        self.toggle_sel_layer(True)
        self.web_view.page().runJavaScript("window.setAllLayersVisibility(true);")

    def hide_all_layers(self):
        self.toggle_pointcloud_layer(False)
        self.toggle_staged_layer(False)
        self.toggle_pick_layer(False)
        self.toggle_sel_layer(False)
        self.web_view.page().runJavaScript("window.setAllLayersVisibility(false);")

    def clear_picked_highlight(self):
        self.web_view.page().runJavaScript("window.clearPickedPointHighlight();")
        if hasattr(self, "lbl_layer_pick_coords"):
            self.lbl_layer_pick_coords.setText("Position: None")
        if hasattr(self, "lbl_layer_pick_attr"):
            self.lbl_layer_pick_attr.setText("Class: - | Tree ID: - | Intensity: -")

    def _on_layer_staged_double_clicked(self, row: int, col: int):
        staged = self.manager.get_staged_edits()
        if 0 <= row < len(staged):
            edit_id = staged[row].get("id")
            if edit_id:
                self.web_view.page().runJavaScript(f"window.zoomToStagedEdit('{edit_id}');")

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
        self.web_view.page().runJavaScript(f"window.setStagedItemVisibility('{edit_id}', {str(visible).lower()});")

    # -------------------------------------------------------------------------
    # Dataset Loading & Fast Header-Only Inspection
    # -------------------------------------------------------------------------
    def open_potree_file(self):
        """Open an existing Potree metadata.json / cloud.js, or a COPC (.copc.laz).

        Potree dispatches on the entry file name: *.copc.laz goes to the COPC loader
        and streams by byte range, so it opens without conversion.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Potree Converted File or COPC",
            "",
            "Potree Files (metadata.json cloud.js *.copc.laz);;LAZ/LAS (*.laz *.las);;All Files (*)"
        )
        if not file_path:
            return

        folder = os.path.dirname(os.path.abspath(file_path))
        entry_file = os.path.basename(file_path)

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

        # Look for sibling or original LAS/LAZ file
        base_name = os.path.basename(folder).replace("_converted", "")
        parent_dir = os.path.dirname(folder)
        for ext in [".laz", ".las"]:
            candidate = os.path.join(parent_dir, base_name + ext)
            if os.path.exists(candidate):
                self.load_las_header_only(candidate)
                break

    def open_las_file(self):
        """Open a raw LAS/LAZ file, load header immediately, and offer streaming view."""
        las_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select LAS / LAZ Point Cloud",
            "",
            "LAS/LAZ Files (*.las *.laz);;All Files (*)"
        )
        if not las_path:
            return

        # 1. Read header metadata instantly without memory allocation
        self.load_las_header_only(las_path)

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
            self.load_url_in_potree(url, os.path.basename(converted_dir))
        else:
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
            self.status_bar.showMessage(f"Loaded '{info['filename']}' ({info['total_points']:,} points) in memory.")
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
                self.load_url_in_potree(url, os.path.basename(out_dir))
            else:
                print("Conversion error:", proc.stderr)

            self.progress_bar.setVisible(False)
            self.status_bar.showMessage("Conversion finished.")

        threading.Thread(target=run, daemon=True).start()

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

    def _sync_color_combos(self, attr):
        """Point both Color By dropdowns at one attribute.

        The ribbon combo and the sidebar's per-layer combo are two views of the same
        value -- the active layer's material.activeAttributeName -- so they are set
        together, with signals blocked so neither bounces a change back.  A cloud can
        carry attributes the fixed lists never offered (treeID, itc); those are added
        rather than leaving a dropdown showing a mode the cloud is not using.
        """
        if not attr:
            return
        for combo in (getattr(self, "combo_layer_color", None),
                      getattr(self, "ribbon_combo_color", None)):
            if combo is None:
                continue
            idx = combo.findData(str(attr))
            if idx < 0:
                combo.addItem(str(attr).replace("_", " ").title(), str(attr))
                idx = combo.count() - 1
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

        self.web_view.page().runJavaScript("window.listLayers ? window.listLayers() : [];", handle)

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
                self._sync_color_combos(active_info.get("colorAttribute"))
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
        self.web_view.page().runJavaScript(
            f"window.setActiveLayer({json.dumps(name)});",
            lambda res: self._render_layer_rows(res if isinstance(res, list) else []))

    def _on_layer_row_checked(self, item):
        """The row checkbox is pc.visible, potree's own per-layer visibility flag."""
        if self._layer_sync_guard:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        visible = item.checkState() == Qt.CheckState.Checked
        self.web_view.page().runJavaScript(
            f"window.setLayerVisible({json.dumps(name)}, {str(bool(visible)).lower()});")

    def _on_layer_remove_clicked(self):
        """Drop one cloud from the scene list and the graph, leaving the others alone."""
        item = self.layer_list.currentItem()
        if not item:
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        self.web_view.page().runJavaScript(
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
        self.web_view.page().runJavaScript(js)
        # The load resolves on its own schedule; refresh_layers retries until the new
        # cloud shows up, which is what makes it a layer instead of a replacement.
        QTimer.singleShot(600, lambda: self.refresh_layers(0))

    # -------------------------------------------------------------------------
    # Selection & Optimistic Visual Staging
    # -------------------------------------------------------------------------
    def on_selection_changed(self, data: dict):
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
            self.lbl_layer_pick_attr.setText(f"{cls_str} | {tree_str} | {int_str}")
        if hasattr(self, "chk_layer_pick"):
            self.chk_layer_pick.blockSignals(True)
            self.chk_layer_pick.setChecked(True)
            self.chk_layer_pick.blockSignals(False)
        if hasattr(self, "ribbon_chk_pick"):
            self.ribbon_chk_pick.blockSignals(True)
            self.ribbon_chk_pick.setChecked(True)
            self.ribbon_chk_pick.blockSignals(False)

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
            self.web_view.page().runJavaScript(f"window.addStagedVisualPolygon({json.dumps(staged)});")
            self.web_view.page().runJavaScript("window.clearSelection(false);")
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
            self.web_view.page().runJavaScript(f"window.addStagedVisualPolygon({json.dumps(staged)});")
            self.web_view.page().runJavaScript("window.clearSelection(false);")
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
            self.web_view.page().runJavaScript(f"window.addStagedVisualPolygon({json.dumps(staged)});")
            self.web_view.page().runJavaScript("window.clearSelection(false);")
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
            self.web_view.page().runJavaScript("window.clearSelection(false);")

    def clear_selection(self):
        self.manager.current_selection = None
        self.web_view.page().runJavaScript("window.clearSelection(true);")
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
            type_text = f"Poly ({coords_len} pts)" if edit.get("type") == "polygon" else edit.get("type")

            item_lbl = QTableWidgetItem(label_text)
            item_act = QTableWidgetItem(action_text)
            item_type = QTableWidgetItem(type_text)

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
        self.manager.remove_staged_edit(edit_id)
        self.web_view.page().runJavaScript(f"window.removeStagedVisualPolygon('{edit_id}');")
        self.refresh_staged_edits_table()
        self.status_bar.showMessage(f"Removed staged edit: {edit_id}")

    def clear_all_staged_edits(self):
        self.manager.clear_staged_edits()
        self.web_view.page().runJavaScript("window.clearAllStagedVisuals();")
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
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Source LAS/LAZ File to Commit Edits Into",
                "",
                "LAS/LAZ Files (*.las *.laz);;All Files (*)"
            )
            if not file_path:
                return
            self.current_las_path = file_path

        base, ext = os.path.splitext(self.current_las_path)
        default_out = f"{base}_edited{ext}"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
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
            self.lbl_save_status.setText("Status: <span style='color:#10b981;'>All changes saved</span>")
            QMessageBox.information(self, "Saved", f"Point cloud successfully saved to:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    # -------------------------------------------------------------------------
    # Render & Visual Settings Actions
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

    def set_color_mode(self, mode: str):
        self.web_view.page().runJavaScript(f"window.setColorAttribute('{mode}');")

    def toggle_in_viewport_settings(self):
        self.web_view.page().runJavaScript("window.toggleSettingsMenu();")

    def set_edl_opacity(self, opacity: float):
        self.web_view.page().runJavaScript(f"window.setEDLOpacity({opacity:.2f});")

    def set_min_node_size(self, size: int):
        self.web_view.page().runJavaScript(f"window.setMinNodeSize({size});")

    def set_background(self, bg: str):
        self.web_view.page().runJavaScript(f"window.setBackground('{bg}');")

    # -------------------------------------------------------------------------
    # Top Tabbed Ribbon Bar
    # -------------------------------------------------------------------------
    def create_ribbon_bar(self) -> QWidget:
        ribbon_container = QWidget()
        ribbon_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        ribbon_container.setFixedHeight(98)
        v_main = QVBoxLayout(ribbon_container)
        v_main.setContentsMargins(0, 0, 0, 0)
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

        self.ribbon_tab_group.addButton(self.btn_tab_file, 0)
        self.ribbon_tab_group.addButton(self.btn_tab_point, 1)
        self.ribbon_tab_group.addButton(self.btn_tab_layers, 2)
        self.ribbon_tab_group.addButton(self.btn_tab_settings, 3)

        h_tabs.addWidget(self.btn_tab_file)
        h_tabs.addWidget(self.btn_tab_point)
        h_tabs.addWidget(self.btn_tab_layers)
        h_tabs.addWidget(self.btn_tab_settings)
        h_tabs.addStretch()

        v_main.addWidget(category_bar)

        # Bottom Row: QStackedWidget switching pages based on active tab
        self.ribbon_stack = QStackedWidget()
        self.ribbon_stack.setObjectName("ribbonStack")
        self.ribbon_stack.setFixedHeight(68)

        page_file = self._create_ribbon_file_page()
        page_point = self._create_ribbon_point_page()
        page_layers = self._create_ribbon_layers_page()
        page_settings = self._create_ribbon_settings_page()

        self.ribbon_stack.addWidget(page_file)       # Index 0
        self.ribbon_stack.addWidget(page_point)      # Index 1
        self.ribbon_stack.addWidget(page_layers)     # Index 2
        self.ribbon_stack.addWidget(page_settings)   # Index 3

        self.ribbon_tab_group.idClicked.connect(self._on_ribbon_category_clicked)
        self.ribbon_stack.setCurrentIndex(1)

        v_main.addWidget(self.ribbon_stack)
        return ribbon_container

    # Category 1: File Page
    def _create_ribbon_file_page(self) -> QWidget:
        page = QFrame()
        page.setFixedHeight(68)
        h_layout = QHBoxLayout(page)
        h_layout.setContentsMargins(8, 3, 8, 3)
        h_layout.setSpacing(8)

        # Open Group
        grp_open = QFrame()
        grp_open.setProperty("class", "ribbonGroup")
        v_open = QVBoxLayout(grp_open)
        v_open.setContentsMargins(8, 4, 8, 4)
        v_open.setSpacing(3)

        h_open_btns = QHBoxLayout()
        h_open_btns.setSpacing(4)
        btn_open_potree = QPushButton("Open Potree...")
        btn_open_potree.setProperty("class", "ribbonToolBtn")
        btn_open_potree.clicked.connect(self.open_potree_file)

        btn_open_las = QPushButton("Open LAS/LAZ...")
        btn_open_las.setProperty("class", "ribbonToolBtn")
        btn_open_las.clicked.connect(self.open_las_file)

        h_open_btns.addWidget(btn_open_potree)
        h_open_btns.addWidget(btn_open_las)
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
        v_save.setContentsMargins(8, 4, 8, 4)
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

        # Application Group
        grp_app = QFrame()
        grp_app.setProperty("class", "ribbonGroup")
        v_app = QVBoxLayout(grp_app)
        v_app.setContentsMargins(8, 4, 8, 4)
        v_app.setSpacing(3)

        btn_exit = QPushButton("Exit Application")
        btn_exit.setProperty("class", "ribbonToolBtn")
        btn_exit.setObjectName("btnDanger")
        btn_exit.clicked.connect(self.close)
        v_app.addWidget(btn_exit)

        lbl_app = QLabel("APP")
        lbl_app.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_app.setProperty("class", "ribbonTitle")
        v_app.addWidget(lbl_app)
        h_layout.addWidget(grp_app)

        h_layout.addStretch()
        return page

    # Category 2: Point Page
    def _create_ribbon_point_page(self) -> QWidget:
        page = QFrame()
        page.setFixedHeight(68)
        h_layout = QHBoxLayout(page)
        h_layout.setContentsMargins(8, 3, 8, 3)
        h_layout.setSpacing(8)

        # 1. Point Settings Group
        grp_point = QFrame()
        grp_point.setProperty("class", "ribbonGroup")
        v_pt = QVBoxLayout(grp_point)
        v_pt.setContentsMargins(8, 4, 8, 4)
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
        v_color.setContentsMargins(8, 4, 8, 4)
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
        v_lod.setContentsMargins(8, 4, 8, 4)
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
        v_tools.setContentsMargins(8, 4, 8, 4)
        v_tools.setSpacing(3)

        h_tools = QHBoxLayout()
        h_tools.setSpacing(4)

        btn_orbit = QPushButton("Navigate")
        btn_orbit.setProperty("class", "ribbonToolBtn")
        btn_orbit.clicked.connect(lambda: self.activate_web_tool("nav"))

        btn_lasso = QPushButton("Polygon Select")
        btn_lasso.setProperty("class", "ribbonToolBtn")
        btn_lasso.clicked.connect(lambda: self.activate_web_tool("polygon"))

        btn_box = QPushButton("Box Select")
        btn_box.setProperty("class", "ribbonToolBtn")
        btn_box.clicked.connect(lambda: self.activate_web_tool("box"))

        btn_pick = QPushButton("Pick Point")
        btn_pick.setProperty("class", "ribbonToolBtn")
        btn_pick.clicked.connect(lambda: self.activate_web_tool("inspect"))

        btn_commit_ribbon = QPushButton("Commit Edits")
        btn_commit_ribbon.setProperty("class", "ribbonToolBtn")
        btn_commit_ribbon.setObjectName("btnSuccess")
        btn_commit_ribbon.clicked.connect(self.commit_staged_edits_action)

        btn_clear = QPushButton("Clear")
        btn_clear.setProperty("class", "ribbonToolBtn")
        btn_clear.setObjectName("btnDanger")
        btn_clear.clicked.connect(lambda: self.activate_web_tool("clear"))

        h_tools.addWidget(btn_orbit)
        h_tools.addWidget(btn_lasso)
        h_tools.addWidget(btn_box)
        h_tools.addWidget(btn_pick)
        h_tools.addWidget(btn_commit_ribbon)
        h_tools.addWidget(btn_clear)

        v_tools.addLayout(h_tools)
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
        page.setFixedHeight(68)
        h_layout = QHBoxLayout(page)
        h_layout.setContentsMargins(8, 3, 8, 3)
        h_layout.setSpacing(8)

        # 1. Quick Layer Visibility Toggles Group
        grp_vis = QFrame()
        grp_vis.setProperty("class", "ribbonGroup")
        v_vis = QVBoxLayout(grp_vis)
        v_vis.setContentsMargins(8, 4, 8, 4)
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
        v_master.setContentsMargins(8, 4, 8, 4)
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
        v_sidebar.setContentsMargins(8, 4, 8, 4)
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
        page.setFixedHeight(68)
        h_layout = QHBoxLayout(page)
        h_layout.setContentsMargins(8, 3, 8, 3)
        h_layout.setSpacing(8)

        # 1. Lighting (EDL) Group
        grp_edl = QFrame()
        grp_edl.setProperty("class", "ribbonGroup")
        v_edl = QVBoxLayout(grp_edl)
        v_edl.setContentsMargins(8, 4, 8, 4)
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
        v_display.setContentsMargins(8, 4, 8, 4)
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

    # Ribbon Event Handlers
    def _on_ribbon_category_clicked(self, idx: int):
        self.ribbon_stack.setCurrentIndex(idx)
        if idx == 2:  # Layers tab
            if hasattr(self, "sidebar_tabs"):
                self.sidebar_tabs.setCurrentIndex(0)

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
            self.web_view.page().runJavaScript(
                "window.getRenderSettings ? window.getRenderSettings() : null;",
                lambda settings: self._sync_viewer_render_widgets(
                    settings if isinstance(settings, dict) else {}))

        self.web_view.page().runJavaScript(
            f"window.setRenderMode({json.dumps(mode)});", apply_layers)

    def _on_ribbon_bg_changed(self):
        bg = self.ribbon_combo_bg.currentData()
        self.set_background(bg)

    def activate_web_tool(self, tool_name: str):
        if tool_name == "clear":
            self.clear_selection()
        else:
            self.web_view.page().runJavaScript(f"window.setActiveTool('{tool_name}');")
