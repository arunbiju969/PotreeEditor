"""
Potree Conversion Dialog & Background Worker.

Provides:
- Native PotreeConverter discovery across Linux, Windows, and system PATH.
- Real-time streaming stdout/stderr log console with auto-scroll and clipboard copy.
- Real-time stage tracking (Counting, Chunking, Indexing, Merging) and progress bar.
- Structured Conversion Statistics dashboard (points processed, duration, throughput, file size, sampling method).
- Direct thread-safe integration with Potree 2.0 streaming octree loader.
"""

import os
import sys
import re
import shutil
import subprocess
from typing import Optional, Dict, Any, List

from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QPlainTextEdit, QFrame,
    QGridLayout, QGroupBox, QCheckBox, QFileDialog,
    QApplication, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer, QSize
from PyQt6.QtGui import QFont, QColor, QClipboard, QCursor


def find_potree_converter(base_dir: Optional[str] = None) -> Optional[str]:
    """Locate the PotreeConverter executable on the system or within the repository.

    Checks in priority order:
    1. POTREE_CONVERTER environment variable
    2. System PATH (shutil.which)
    3. Bundled Linux release 2.1.2 (tools/PotreeConverter_2.1.2_x64_linux) — depth-10 octree
    4. Project built Linux binary (tools/PC-2.1.5/build/PotreeConverter)
    5. Windows binary (potree/libs/PotreeConverter2/PotreeConverter.exe)
    6. tools/ directory fallbacks
    """
    # 1. Environment variable
    env_bin = os.environ.get("POTREE_CONVERTER")
    if env_bin and os.path.isfile(env_bin):
        if sys.platform == "win32" or os.access(env_bin, os.X_OK):
            return os.path.abspath(env_bin)

    # 2. System PATH
    path_bin = shutil.which("PotreeConverter") or shutil.which("PotreeConverter.exe")
    if path_bin:
        return os.path.abspath(path_bin)

    # 3. Project directories
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    search_roots = [
        base_dir,
        os.path.dirname(base_dir),
        os.path.join(base_dir, "potree"),
    ]

    relative_candidates = [
        # Bundled Linux release 2.1.2 — octree depth 10 (8.5 mm leaves);
        # 2.1.5 auto-picks shallower depths (8 for a 213M-pt cloud)
        os.path.join("tools", "PotreeConverter_2.1.2_x64_linux", "PotreeConverter"),
        # Native Linux build (dynamic, 24-thread)
        os.path.join("tools", "PC-2.1.5", "build", "PotreeConverter"),
        # Other potential builds
        os.path.join("libs", "PotreeConverter2", "PotreeConverter.exe"),
        os.path.join("potree", "libs", "PotreeConverter2", "PotreeConverter.exe"),
        os.path.join("libs", "PotreeConverter2", "PotreeConverter"),
        os.path.join("tools", "PotreeConverter"),
        os.path.join("tools", "PotreeConverter.exe"),
    ]

    for root in search_roots:
        for rel in relative_candidates:
            cand = os.path.join(root, rel)
            if os.path.isfile(cand):
                if sys.platform != "win32" and not os.access(cand, os.X_OK):
                    continue
                return os.path.abspath(cand)

    return None


def format_file_size(num_bytes: float) -> str:
    """Format raw byte counts into human-readable strings."""
    if num_bytes <= 0:
        return "0 B"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0 or unit == "TB":
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{int(num_bytes)} B"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} GB"


def format_points_rate(pts: int, duration_sec: float) -> str:
    """Compute and format point processing rate."""
    if duration_sec <= 0 or pts <= 0:
        return "N/A"
    rate = pts / duration_sec
    if rate >= 1_000_000:
        return f"{rate / 1_000_000:.2f} Mpts/s"
    elif rate >= 1_000:
        return f"{rate / 1_000:.1f} Kpts/s"
    else:
        return f"{rate:.0f} pts/s"


class ConversionWorker(QThread):
    """Background worker that streams PotreeConverter subprocess output line-by-line

    and parses conversion stages, point progress, and final statistics block.
    """
    log_line = pyqtSignal(str)
    stage_changed = pyqtSignal(str)
    progress_updated = pyqtSignal(int, int, str)  # current, total, text
    stats_ready = pyqtSignal(dict)
    finished = pyqtSignal(bool, str, dict)  # success, message, stats

    def __init__(self, converter_bin: str, input_las: str, out_dir: str,
                 method: str = "poisson", extra_args: Optional[List[str]] = None):
        super().__init__()
        self.converter_bin = converter_bin
        self.input_las = input_las
        self.out_dir = out_dir
        self.method = method
        self.extra_args = extra_args or []

        self.proc: Optional[subprocess.Popen] = None
        self.is_cancelled = False
        self.total_points = 0
        self.stats: Dict[str, Any] = {}

    def run(self):
        cmd = [self.converter_bin, self.input_las, "-o", self.out_dir]
        if self.method:
            cmd.extend(["--method", self.method])
        if self.extra_args:
            cmd.extend(self.extra_args)

        self.stats = {
            "source_file": self.input_las,
            "output_directory": self.out_dir,
            "sampling_method": self.method,
            "status": "Running",
        }

        # Check input file size
        try:
            if os.path.exists(self.input_las):
                sz = os.path.getsize(self.input_las)
                self.stats["input_size_bytes"] = sz
                self.stats["input file size"] = format_file_size(sz)
        except Exception:
            pass

        self.log_line.emit(f"=== PotreeConverter Execution Started ===")
        self.log_line.emit(f"Command: {' '.join(cmd)}\n")
        self.stage_changed.emit("Starting")
        self.progress_updated.emit(0, 0, "Initializing PotreeConverter...")

        total_points = 0
        in_stats_block = False
        last_error_line = ""

        try:
            os.makedirs(self.out_dir, exist_ok=True)

            # Bundled 2.1.2 release ships liblaszip.so next to the binary
            env = os.environ.copy()
            conv_dir = os.path.dirname(os.path.abspath(self.converter_bin))
            ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{conv_dir}{os.pathsep}{ld}" if ld else conv_dir

            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )

            for line in self.proc.stdout:
                if self.is_cancelled:
                    break

                raw = line.rstrip("\r\n")
                self.log_line.emit(raw)
                s = raw.strip()
                if not s:
                    continue

                # Stage detection
                if "=== COUNTING" in s:
                    self.stage_changed.emit("Counting")
                    self.progress_updated.emit(0, 0, "Counting points and octree cells...")
                    continue
                elif "=== CREATING CHUNKS" in s:
                    self.stage_changed.emit("Creating Chunks")
                    self.progress_updated.emit(0, 0, "Partitioning points into chunks...")
                    continue
                elif "=== INDEXING" in s:
                    self.stage_changed.emit("Indexing")
                    self.progress_updated.emit(0, total_points, "Indexing octree nodes...")
                    continue
                elif "=== MERGING" in s:
                    self.stage_changed.emit("Merging")
                    self.progress_updated.emit(total_points, total_points, "Merging octree hierarchy & metadata...")
                    continue
                elif "=== STATS" in s:
                    in_stats_block = True
                    self.stage_changed.emit("Finalizing")
                    continue

                # Header and progress parsing
                if not in_stats_block:
                    if s.startswith("#points:"):
                        m = re.search(r"#points:\s*([\d',]+)", s)
                        if m:
                            total_points = int(m.group(1).replace("'", "").replace(",", ""))
                            self.total_points = total_points
                            self.stats["#points"] = f"{total_points:,}"
                            self.progress_updated.emit(0, total_points, f"Total points: {total_points:,}")
                    elif "total file size:" in s:
                        m = re.search(r"total file size:\s*([0-9.,]+)\s*bytes", s)
                        if m:
                            b_val = float(m.group(1).replace(",", ""))
                            self.stats["input_size_bytes"] = b_val
                            self.stats["input file size"] = format_file_size(b_val)
                    elif "processed points:" in s:
                        m = re.search(r"processed points:\s*([\d',]+)", s)
                        if m:
                            curr = int(m.group(1).replace("'", "").replace(",", ""))
                            pct = f" ({(curr / total_points * 100):.1f}%)" if total_points > 0 else ""
                            msg = f"Indexed {curr:,} / {total_points:,} points{pct}"
                            self.progress_updated.emit(curr, total_points, msg)
                else:
                    # In stats block
                    if ":" in s:
                        k, v = s.split(":", 1)
                        self.stats[k.strip()] = v.strip()
                    elif "throughput" in s.lower():
                        parts = re.split(r"\s{2,}", s, maxsplit=1)
                        if len(parts) == 2:
                            self.stats[parts[0].strip()] = parts[1].strip()

                # Track errors
                if any(err_word in s.lower() for err_word in ("error", "exception", "failed", "terminate called")):
                    last_error_line = s

            if self.is_cancelled:
                if self.proc and self.proc.poll() is None:
                    try:
                        self.proc.terminate()
                        self.proc.wait(timeout=1.0)
                    except Exception:
                        self.proc.kill()
                self.stats["status"] = "Cancelled"
                self.stage_changed.emit("Cancelled")
                self.finished.emit(False, "Conversion was cancelled by the user.", self.stats)
                return

            self.proc.wait()
            rc = self.proc.returncode

            meta_json = os.path.join(self.out_dir, "metadata.json")
            cloud_js = os.path.join(self.out_dir, "cloud.js")
            has_output = os.path.exists(meta_json) or os.path.exists(cloud_js)

            if rc == 0 and has_output:
                self.stats["status"] = "Success"
                if os.path.exists(meta_json):
                    try:
                        import json
                        with open(meta_json, "r") as mf:
                            mdata = json.load(mf)
                            if "points" in mdata and not self.stats.get("#points"):
                                self.stats["#points"] = f"{mdata['points']:,}"
                    except Exception:
                        pass

                # Calculate enhanced throughput if duration exists
                dur_str = self.stats.get("duration", "")
                if dur_str:
                    m_dur = re.search(r"([0-9.]+)", dur_str)
                    if m_dur:
                        try:
                            dur_sec = float(m_dur.group(1))
                            pts_count = total_points or int(self.stats.get("#points", "0").replace(",", "").replace("'", ""))
                            self.stats["calculated_throughput"] = format_points_rate(pts_count, dur_sec)
                        except Exception:
                            pass

                self.stage_changed.emit("Complete")
                self.stats_ready.emit(self.stats)
                self.progress_updated.emit(total_points or 100, total_points or 100, "Conversion completed successfully!")
                self.finished.emit(True, "Point cloud conversion completed successfully!", self.stats)
            else:
                err_text = last_error_line or f"PotreeConverter exited with error code {rc}"
                self.stats["status"] = "Failed"
                self.stage_changed.emit("Failed")
                self.finished.emit(False, f"Conversion failed: {err_text}", self.stats)

        except Exception as e:
            self.stats["status"] = "Error"
            self.stage_changed.emit("Error")
            self.finished.emit(False, f"Failed to execute PotreeConverter: {e}", self.stats)

    def cancel(self):
        """Request worker and child subprocess cancellation."""
        self.is_cancelled = True
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass


class ConversionDialog(QDialog):
    """Comprehensive Potree Point Cloud Conversion Dialog with live streaming logs,

    real-time octree stage tracking, and structured statistics dashboard.
    """
    conversion_completed = pyqtSignal(str, dict)  # out_dir, stats

    def __init__(self, converter_bin: str, input_las: str, out_dir: str,
                 parent: Optional[QWidget] = None, auto_start: bool = True):
        super().__init__(parent)
        self.converter_bin = converter_bin
        self.input_las = input_las
        self.out_dir = out_dir
        self.worker: Optional[ConversionWorker] = None
        self.final_stats: Dict[str, Any] = {}
        self.is_running = False

        self.setWindowTitle(f"PotreeConverter - {os.path.basename(input_las)}")
        self.setMinimumSize(780, 620)
        self.resize(820, 660)

        self._setup_style()
        self._build_ui()

        if auto_start:
            QTimer.singleShot(150, self.start_conversion)

    def _setup_style(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1e28;
                color: #e2e8f0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
            QGroupBox {
                border: 1px solid #2d3748;
                border-radius: 8px;
                margin-top: 14px;
                padding: 16px 12px 12px 12px;
                font-weight: 700;
                color: #60a5fa;
                font-size: 11px;
                letter-spacing: 0.5px;
                background-color: #141822;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                top: 0px;
                padding: 0 6px;
                background-color: #141822;
                border-radius: 4px;
            }
            QPushButton {
                background-color: #2563eb;
                color: #ffffff;
                border: 1px solid #1d4ed8;
                border-radius: 6px;
                padding: 6px 14px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3b82f6;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
            QPushButton:disabled {
                background-color: #1e293b;
                color: #64748b;
                border: 1px solid #334155;
            }
            QPushButton#btnCancel {
                background-color: #334155;
                border: 1px solid #475569;
                color: #f1f5f9;
            }
            QPushButton#btnCancel:hover {
                background-color: #dc2626;
                border-color: #ef4444;
            }
            QPushButton#btnOpen {
                background-color: #059669;
                border: 1px solid #10b981;
            }
            QPushButton#btnOpen:hover {
                background-color: #10b981;
            }
            QProgressBar {
                border: 1px solid #334155;
                border-radius: 6px;
                background-color: #0f172a;
                text-align: center;
                color: #f8fafc;
                font-size: 11px;
                font-weight: 600;
                min-height: 18px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2563eb, stop:0.5 #38bdf8, stop:1 #10b981);
                border-radius: 5px;
            }
            QPlainTextEdit {
                background-color: #0d1117;
                color: #cbd5e1;
                border: 1px solid #2d3748;
                border-radius: 6px;
                font-family: "DejaVu Sans Mono", "Consolas", "Courier New", monospace;
                font-size: 11px;
                padding: 6px;
            }
            QCheckBox {
                color: #94a3b8;
                font-size: 11px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border-radius: 3px;
                border: 1px solid #475569;
                background: #0f172a;
            }
            QCheckBox::indicator:checked {
                background: #2563eb;
                border-color: #3b82f6;
            }
        """)

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 1. Header Card with Status Badge
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #161b26;
                border: 1px solid #273142;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        h_header = QHBoxLayout(header_frame)
        h_header.setContentsMargins(12, 10, 12, 10)

        v_titles = QVBoxLayout()
        lbl_title = QLabel("Point Cloud LOD Octree Conversion")
        lbl_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #f8fafc;")

        # Input and Output paths display
        fname = os.path.basename(self.input_las)
        fsize = format_file_size(os.path.getsize(self.input_las)) if os.path.exists(self.input_las) else ""
        size_tag = f" ({fsize})" if fsize else ""

        lbl_paths = QLabel(
            f"<b>Source:</b> {fname}{size_tag} &nbsp;·&nbsp; "
            f"<b>Output:</b> {os.path.basename(self.out_dir)}"
        )
        lbl_paths.setStyleSheet("font-size: 11px; color: #94a3b8;")
        v_titles.addWidget(lbl_title)
        v_titles.addWidget(lbl_paths)
        h_header.addLayout(v_titles, stretch=1)

        # Status badge
        self.lbl_badge = QLabel("READY")
        self.lbl_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_badge.setStyleSheet("""
            background-color: #1e293b;
            color: #94a3b8;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 4px 14px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
        """)
        h_header.addWidget(self.lbl_badge)
        main_layout.addWidget(header_frame)

        # 2. Progress & Active Stage
        v_prog = QVBoxLayout()
        v_prog.setSpacing(5)

        self.lbl_stage_info = QLabel("Stage: Initializing...")
        self.lbl_stage_info.setStyleSheet("font-size: 12px; font-weight: 600; color: #e2e8f0;")
        v_prog.addWidget(self.lbl_stage_info)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # indeterminate until points known
        self.progress_bar.setValue(0)
        v_prog.addWidget(self.progress_bar)
        main_layout.addLayout(v_prog)

        # 3. Statistics Dashboard (Structured KPI Cards)
        grp_stats = QGroupBox("CONVERSION STATISTICS")
        grid_stats = QGridLayout(grp_stats)
        grid_stats.setContentsMargins(12, 16, 12, 10)
        grid_stats.setHorizontalSpacing(16)
        grid_stats.setVerticalSpacing(8)

        # Helper for KPI tiles
        def create_kpi_tile(label_text: str, default_val: str = "--"):
            w = QFrame()
            w.setStyleSheet("""
                QFrame {
                    background-color: #0f141e;
                    border: 1px solid #1f2937;
                    border-radius: 6px;
                    padding: 4px 8px;
                }
            """)
            vl = QVBoxLayout(w)
            vl.setContentsMargins(6, 4, 6, 4)
            vl.setSpacing(2)
            lbl_k = QLabel(label_text.upper())
            lbl_k.setStyleSheet("color: #64748b; font-size: 9px; font-weight: 700; letter-spacing: 0.5px;")
            lbl_v = QLabel(default_val)
            lbl_v.setStyleSheet("color: #f1f5f9; font-size: 13px; font-weight: 700;")
            vl.addWidget(lbl_k)
            vl.addWidget(lbl_v)
            return w, lbl_v

        tile_pts, self.kpi_points = create_kpi_tile("Total Points")
        tile_dur, self.kpi_duration = create_kpi_tile("Elapsed Duration")
        tile_rate, self.kpi_rate = create_kpi_tile("Throughput Rate")
        tile_sz, self.kpi_size = create_kpi_tile("Input File Size")
        tile_method, self.kpi_method = create_kpi_tile("Sampling Method")
        tile_phases, self.kpi_phases = create_kpi_tile("Phase Breakdown")

        if os.path.exists(self.input_las):
            self.kpi_size.setText(format_file_size(os.path.getsize(self.input_las)))
        self.kpi_method.setText("Poisson (LASZIP)")

        grid_stats.addWidget(tile_pts, 0, 0)
        grid_stats.addWidget(tile_dur, 0, 1)
        grid_stats.addWidget(tile_rate, 0, 2)
        grid_stats.addWidget(tile_sz, 1, 0)
        grid_stats.addWidget(tile_method, 1, 1)
        grid_stats.addWidget(tile_phases, 1, 2)

        main_layout.addWidget(grp_stats)

        # 4. Console Log Terminal
        grp_log = QGroupBox("REAL-TIME ENGINE LOG")
        v_log = QVBoxLayout(grp_log)
        v_log.setContentsMargins(10, 14, 10, 10)
        v_log.setSpacing(6)

        h_log_tools = QHBoxLayout()
        h_log_tools.setSpacing(8)

        lbl_log_sub = QLabel("Subprocess stdout / stderr stream:")
        lbl_log_sub.setStyleSheet("font-size: 11px; color: #64748b;")
        h_log_tools.addWidget(lbl_log_sub)
        h_log_tools.addStretch(1)

        self.chk_autoscroll = QCheckBox("Auto-scroll")
        self.chk_autoscroll.setChecked(True)
        h_log_tools.addWidget(self.chk_autoscroll)

        self.btn_copy_log = QPushButton("Copy Log")
        self.btn_copy_log.setStyleSheet("padding: 3px 10px; font-size: 11px;")
        self.btn_copy_log.clicked.connect(self._copy_log)
        h_log_tools.addWidget(self.btn_copy_log)

        self.btn_save_log = QPushButton("Save Log...")
        self.btn_save_log.setStyleSheet("padding: 3px 10px; font-size: 11px;")
        self.btn_save_log.clicked.connect(self._save_log)
        h_log_tools.addWidget(self.btn_save_log)

        v_log.addLayout(h_log_tools)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(10000)
        v_log.addWidget(self.log_text, stretch=1)

        main_layout.addWidget(grp_log, stretch=1)

        # 5. Bottom Action Controls
        h_bottom = QHBoxLayout()
        h_bottom.setSpacing(10)

        self.lbl_bottom_info = QLabel("Ready to convert.")
        self.lbl_bottom_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        h_bottom.addWidget(self.lbl_bottom_info)
        h_bottom.addStretch(1)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("btnCancel")
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        h_bottom.addWidget(self.btn_cancel)

        self.btn_open = QPushButton("Open Converted Cloud")
        self.btn_open.setObjectName("btnOpen")
        self.btn_open.setEnabled(False)
        self.btn_open.clicked.connect(self._on_open_clicked)
        h_bottom.addWidget(self.btn_open)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        h_bottom.addWidget(self.btn_close)

        main_layout.addLayout(h_bottom)

    def start_conversion(self):
        """Initialize worker thread and begin streaming conversion."""
        if self.is_running:
            return

        self.is_running = True
        self._update_badge("RUNNING", "#1e3a8a", "#60a5fa", "#3b82f6")
        self.lbl_bottom_info.setText("Conversion running...")
        self.btn_open.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        self.worker = ConversionWorker(
            converter_bin=self.converter_bin,
            input_las=self.input_las,
            out_dir=self.out_dir,
            method="poisson"
        )
        self.worker.log_line.connect(self._on_log_line)
        self.worker.stage_changed.connect(self._on_stage_changed)
        self.worker.progress_updated.connect(self._on_progress_updated)
        self.worker.stats_ready.connect(self._on_stats_ready)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _update_badge(self, text: str, bg: str, fg: str, border: str):
        self.lbl_badge.setText(text)
        self.lbl_badge.setStyleSheet(f"""
            background-color: {bg};
            color: {fg};
            border: 1px solid {border};
            border-radius: 12px;
            padding: 4px 14px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
        """)

    def _on_log_line(self, line: str):
        self.log_text.appendPlainText(line)
        if self.chk_autoscroll.isChecked():
            sb = self.log_text.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _on_stage_changed(self, stage: str):
        stage_clean = stage.upper()
        if "COUNTING" in stage_clean:
            self._update_badge("COUNTING", "#1e3a8a", "#93c5fd", "#3b82f6")
            self.lbl_stage_info.setText("Stage: Counting points and grid cells...")
        elif "CHUNK" in stage_clean:
            self._update_badge("CHUNKING", "#1e3a8a", "#93c5fd", "#3b82f6")
            self.lbl_stage_info.setText("Stage: Distributing points into temporary chunks...")
        elif "INDEX" in stage_clean:
            self._update_badge("INDEXING", "#065f46", "#6ee7b7", "#10b981")
            self.lbl_stage_info.setText("Stage: Indexing multi-resolution octree...")
        elif "MERG" in stage_clean:
            self._update_badge("MERGING", "#78350f", "#fcd34d", "#f59e0b")
            self.lbl_stage_info.setText("Stage: Merging octree hierarchy & generating metadata...")
        elif "COMPLETE" in stage_clean:
            self._update_badge("COMPLETE", "#064e3b", "#34d399", "#059669")
            self.lbl_stage_info.setText("Stage: Conversion Complete!")
        elif "FAIL" in stage_clean:
            self._update_badge("FAILED", "#7f1d1d", "#fca5a5", "#ef4444")
            self.lbl_stage_info.setText("Stage: Conversion Failed")
        elif "CANCEL" in stage_clean:
            self._update_badge("CANCELLED", "#78350f", "#fcd34d", "#f59e0b")
            self.lbl_stage_info.setText("Stage: Cancelled by user")

    def _on_progress_updated(self, current: int, total: int, text: str):
        self.lbl_stage_info.setText(f"Stage: {text}")
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
            pct = (current / total) * 100
            self.progress_bar.setFormat(f"{current:,} / {total:,} ({pct:.1f}%)")
        else:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat(text)

    def _on_stats_ready(self, stats: dict):
        self.final_stats.update(stats)

        # Update KPI tiles
        pts = stats.get("#points")
        if pts:
            self.kpi_points.setText(pts.replace("'", ","))

        dur = stats.get("duration")
        if dur:
            self.kpi_duration.setText(dur)

        rate = stats.get("calculated_throughput") or stats.get("throughput (points/s)")
        if rate:
            self.kpi_rate.setText(rate)

        sz = stats.get("input file size")
        if sz:
            self.kpi_size.setText(sz)

        c_time = stats.get("duration(chunking-total)")
        i_time = stats.get("duration(indexing)")
        if c_time or i_time:
            c_str = f"Chunk: {c_time}s" if c_time else ""
            i_str = f"Idx: {i_time}s" if i_time else ""
            self.kpi_phases.setText(f"{c_str} | {i_str}".strip(" |"))

    def _on_finished(self, success: bool, message: str, stats: dict):
        self.is_running = False
        self.final_stats.update(stats)
        self.btn_cancel.setEnabled(False)

        if success:
            self._update_badge("COMPLETE", "#064e3b", "#34d399", "#059669")
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("100% Complete")
            self.lbl_bottom_info.setText("Conversion finished successfully. Ready to view.")
            self.btn_open.setEnabled(True)
            self.btn_open.setText("Open in Potree")

            # Auto-emit completion so parent viewer can immediately load cloud
            self.conversion_completed.emit(self.out_dir, self.final_stats)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Failed")
            self.lbl_bottom_info.setText(message)

    def _on_cancel_clicked(self):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Cancel Conversion?",
                "Are you sure you want to abort the conversion process?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.cancel()
                self.btn_cancel.setEnabled(False)
                self.lbl_bottom_info.setText("Cancelling conversion...")

    def _on_open_clicked(self):
        self.conversion_completed.emit(self.out_dir, self.final_stats)
        self.accept()

    def _copy_log(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.log_text.toPlainText())
        self.btn_copy_log.setText("Copied!")
        QTimer.singleShot(1500, lambda: self.btn_copy_log.setText("Copy Log"))

    def _save_log(self):
        base = os.path.splitext(os.path.basename(self.input_las))[0]
        default_name = f"{base}_conversion_log.txt"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Conversion Log",
            default_name,
            "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(self.log_text.toPlainText())
                QMessageBox.information(self, "Log Saved", f"Conversion log saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error Saving Log", str(e))

    def closeEvent(self, event):
        if self.is_running and self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Conversion in Progress",
                "A conversion is currently active. Closing this window will cancel it.\n\n"
                "Are you sure you want to stop and close?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.cancel()
                self.worker.wait(1500)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
