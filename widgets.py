"""Small reusable widgets: toast notifications, shortcuts dialog, logger, log panel, FPS/RTF graph."""

import logging
from collections import deque
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel, QWidget,
    QComboBox, QCheckBox, QLineEdit,
)
from PySide6.QtCore import Qt, QTimer, QObject, Signal
from PySide6.QtGui import QFont, QPainter, QColor, QPen


# ── Centralized Logger Setup ──────────────────────────────────
class QtLogEmitter(QObject):
    log_signal = Signal(str)


class QtLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.emitter = QtLogEmitter()

    def emit(self, record):
        try:
            msg = self.format(record)
            self.emitter.log_signal.emit(msg)
        except Exception:
            pass


def setup_logger(name="mujoco_viewer"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(fmt)
    logger.addHandler(console)
    qt_handler = QtLogHandler()
    qt_handler.setLevel(logging.DEBUG)
    qt_handler.setFormatter(fmt)
    logger.addHandler(qt_handler)
    return logger, qt_handler.emitter


log, log_emitter = setup_logger()


# ── Toast with queuing support ────────────────────────────────
class ToastLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self._default_style = (
            "background-color: rgba(61,89,161,210); color:#fff; "
            "border-radius:8px; padding:8px 20px; font-weight:bold; font-size:13px;"
        )
        self._error_style = (
            "background-color: rgba(247,118,142,210); color:#fff; "
            "border-radius:8px; padding:8px 20px; font-weight:bold; font-size:13px;"
        )
        self._success_style = (
            "background-color: rgba(158,206,106,200); color:#1a1b26; "
            "border-radius:8px; padding:8px 20px; font-weight:bold; font-size:13px;"
        )
        self._warning_style = (                                    # NEW
            "background-color: rgba(224,175,104,210); color:#1a1b26; "
            "border-radius:8px; padding:8px 20px; font-weight:bold; font-size:13px;"
        )
        self.setStyleSheet(self._default_style)
        self.hide()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)
        self._queue = deque()                                       # NEW: toast queue

    def show_message(self, text, duration=2500):
        self._enqueue(text, self._default_style, duration)

    def show_error(self, text, duration=4000):
        self._enqueue(text, self._error_style, duration)

    def show_success(self, text, duration=2500):
        self._enqueue(text, self._success_style, duration)

    def show_warning(self, text, duration=3000):                    # NEW
        self._enqueue(text, self._warning_style, duration)

    def _enqueue(self, text, style, duration):                       # NEW
        self._queue.append((text, style, duration))
        if not self._timer.isActive():
            self._show_next()

    def _show_next(self):                                            # NEW
        if not self._queue:
            self.hide()
            return
        text, style, duration = self._queue.popleft()
        self.setStyleSheet(style)
        self._show(text, duration)

    def _on_timeout(self):                                           # NEW
        self._show_next()

    def _show(self, text, duration):
        self.setText(text)
        self.adjustSize()
        if self.parentWidget():
            pw = self.parentWidget().width()
            self.move((pw - self.width()) // 2, 16)
        self.show()
        self.raise_()
        self._timer.start(duration)


# ── Log Panel with filter and search ──────────────────────────
class LogPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Info", "Warning", "Error", "Debug"])
        self.filter_combo.setMaximumWidth(100)
        self.filter_combo.currentIndexChanged.connect(self._apply_filter)  # NEW: live filter
        filter_row.addWidget(self.filter_combo)

        # NEW: Search bar
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Search logs…")
        self.search_input.setProperty("class", "search")
        self.search_input.setMaximumWidth(180)
        self.search_input.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self.search_input)

        self.auto_scroll_cb = QCheckBox("Auto-scroll")
        self.auto_scroll_cb.setChecked(True)
        filter_row.addWidget(self.auto_scroll_cb)

        filter_row.addStretch()

        # NEW: Copy button
        copy_btn = QPushButton("📋 Copy")
        copy_btn.setFixedWidth(70)
        copy_btn.setToolTip("Copy all visible log text")
        copy_btn.clicked.connect(self._copy_visible)
        filter_row.addWidget(copy_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(self._clear)
        filter_row.addWidget(clear_btn)
        layout.addLayout(filter_row)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Consolas", 9))
        self.log_view.document().setMaximumBlockCount(5000)
        self.log_view.setStyleSheet(
            "QTextEdit { background-color: #16161e; color: #a9b1d6; "
            "border: 1px solid #24283b; border-radius: 4px; }"
        )
        layout.addWidget(self.log_view)
        self._all_messages = []

    def append_log(self, message: str):
        escaped = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if "[ERROR]" in message:
            color = "#f7768e"
        elif "[WARNING]" in message:
            color = "#e0af68"
        elif "[INFO]" in message:
            color = "#9ece6a"
        elif "[DEBUG]" in message:
            color = "#565f89"
        else:
            color = "#a9b1d6"

        self._all_messages.append((message, color))
        self._apply_filter()

        if self.auto_scroll_cb.isChecked():
            sb = self.log_view.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _apply_filter(self):
        level = self.filter_combo.currentText().lower()
        search = self.search_input.text().strip().lower()           # NEW
        self.log_view.clear()
        for msg, color in self._all_messages:
            if level != "all":
                if level == "info" and "[INFO]" not in msg:
                    continue
                elif level == "warning" and "[WARNING]" not in msg:
                    continue
                elif level == "error" and "[ERROR]" not in msg:
                    continue
                elif level == "debug" and "[DEBUG]" not in msg:
                    continue
            if search and search not in msg.lower():                 # NEW
                continue
            escaped = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            self.log_view.append(f'<span style="color:{color}">{escaped}</span>')

    def _copy_visible(self):                                         # NEW
        text = self.log_view.toPlainText()
        if text:
            from PySide6.QtWidgets import QApplication
            QApplication.clipboard().setText(text)

    def _clear(self):
        self.log_view.clear()
        self._all_messages.clear()


# ── FPS / RTF Graph Widget ────────────────────────────────────
class FPSGraph(QWidget):                                             # IMPROVED: now shows RTF
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(60)
        self._fps_history = deque(maxlen=120)
        self._rtf_history = deque(maxlen=120)                       # NEW
        self.setMinimumWidth(200)

    def add_fps(self, fps, rtf=1.0):                                # IMPROVED: accepts RTF
        self._fps_history.append(fps)
        self._rtf_history.append(rtf)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#16161e"))
        painter.setPen(QPen(QColor("#24283b"), 1))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

        if len(self._fps_history) < 2:
            painter.setPen(QColor("#565f89"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No FPS data")
            painter.end()
            return

        max_fps = max(max(self._fps_history), 60)
        w, h = self.width(), self.height()
        n = len(self._fps_history)
        dx = w / max(n - 1, 1)

        # 60 FPS reference line
        y60 = h - int((60.0 / max_fps) * (h - 4)) - 2
        painter.setPen(QPen(QColor(158, 206, 106, 60), 1, Qt.DashLine))
        painter.drawLine(0, y60, w, y60)

        # FPS line
        painter.setPen(QPen(QColor("#7aa2f7"), 1.5))
        points = []
        for i, fps in enumerate(self._fps_history):
            x = int(i * dx)
            y = h - int((fps / max_fps) * (h - 4)) - 2
            points.append((x, y))
        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])

        # NEW: RTF line (if we have RTF data)
        if len(self._rtf_history) >= 2:
            max_rtf = max(max(self._rtf_history), 1.5)
            # Scale RTF so 1.0 maps to a visual reference
            painter.setPen(QPen(QColor("#e0af68", 120), 1, Qt.DashLine))
            rtf_points = []
            for i, rtf in enumerate(self._rtf_history):
                x = int(i * dx)
                # Map RTF to upper portion of graph (0-50% height)
                y = h - int((rtf / max_rtf) * (h * 0.5)) - 2
                rtf_points.append((x, y))
            for i in range(len(rtf_points) - 1):
                painter.drawLine(rtf_points[i][0], rtf_points[i][1],
                                 rtf_points[i + 1][0], rtf_points[i + 1][1])

        last_fps = self._fps_history[-1]
        last_rtf = self._rtf_history[-1] if self._rtf_history else 1.0
        painter.setPen(QColor("#7aa2f7"))
        painter.setFont(QFont("Consolas", 8))
        painter.drawText(4, 12, f"{last_fps:.0f} FPS")
        painter.setPen(QColor("#e0af68"))
        painter.drawText(w - 70, 12, f"RTF {last_rtf:.2f}x")
        painter.end()


# ── Shortcuts Dialog ──────────────────────────────────────────
class ShortcutsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)
        info = QTextEdit()
        info.setReadOnly(True)
        info.setHtml("""
        <style>
            table { border-collapse: collapse; width: 100%; }
            td { padding: 5px 10px; border-bottom: 1px solid #24283b; }
            .key { color: #7aa2f7; font-family: Consolas, monospace; font-weight: bold; }
            .desc { color: #a9b1d6; }
            h3 { color: #7aa2f7; }
        </style>
        <h3>Simulation</h3>
        <table>
        <tr><td class="key">Space</td><td class="desc">Play / Pause</td></tr>
        <tr><td class="key">→</td><td class="desc">Step simulation forward</td></tr>
        <tr><td class="key">R</td><td class="desc">Reset simulation</td></tr>
        <tr><td class="key">1–8</td><td class="desc">Set speed (0.1x–10x)</td></tr>
        <tr><td class="key">Ctrl+Shift+R</td><td class="desc">Reload model from file</td></tr>
        </table>
        <h3>Camera</h3>
        <table>
        <tr><td class="key">F</td><td class="desc">Fit camera to scene</td></tr>
        <tr><td class="key">Ctrl+1–8</td><td class="desc">Camera presets</td></tr>
        <tr><td class="key">Left Drag</td><td class="desc">Rotate camera</td></tr>
        <tr><td class="key">Middle Drag / Shift+Left</td><td class="desc">Pan camera</td></tr>
        <tr><td class="key">Right Drag / Scroll</td><td class="desc">Zoom</td></tr>
        </table>
        <h3>Interaction</h3>
        <table>
        <tr><td class="key">Ctrl+Click</td><td class="desc">Select &amp; drag body</td></tr>
        <tr><td class="key">Double-Click</td><td class="desc">Focus on body</td></tr>
        <tr><td class="key">T</td><td class="desc">Toggle body traces</td></tr>
        <tr><td class="key">C</td><td class="desc">Clear traces</td></tr>
        <tr><td class="key">G</td><td class="desc">Toggle grid overlay</td></tr>
        <tr><td class="key">A</td><td class="desc">Toggle axis indicator</td></tr>
        <tr><td class="key">K</td><td class="desc">Save keyframe</td></tr>
        <tr><td class="key">Ctrl+K</td><td class="desc">Load last keyframe</td></tr>
        </table>
        <h3>File</h3>
        <table>
        <tr><td class="key">Ctrl+O</td><td class="desc">Open model file</td></tr>
        <tr><td class="key">Ctrl+S</td><td class="desc">Save screenshot</td></tr>
        </table>
        """)
        layout.addWidget(info)
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)