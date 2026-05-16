"""Small reusable widgets: toast notifications, shortcuts dialog, logger, and log panel."""

import logging
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel, QWidget,
)
from PySide6.QtCore import Qt, QTimer, QObject, Signal
from PySide6.QtGui import QFont


# ── Centralized Logger Setup ──────────────────────────────────
class QtLogEmitter(QObject):
    """QObject wrapper to safely emit log messages to the GUI via signals."""
    log_signal = Signal(str)


class QtLogHandler(logging.Handler):
    """Logging handler that forwards records to a Qt signal for GUI display."""
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

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )

    # Terminal handler
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Qt GUI handler
    qt_handler = QtLogHandler()
    qt_handler.setLevel(logging.DEBUG)
    qt_handler.setFormatter(fmt)
    logger.addHandler(qt_handler)

    return logger, qt_handler.emitter


log, log_emitter = setup_logger()


# ── UI Widgets ────────────────────────────────────────────────
class ToastLabel(QLabel):
    """Floating transient message that auto-hides."""

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
        self.setStyleSheet(self._default_style)
        self.hide()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_message(self, text, duration=2500):
        self.setStyleSheet(self._default_style)
        self._show(text, duration)

    def show_error(self, text, duration=4000):
        self.setStyleSheet(self._error_style)
        self._show(text, duration)

    def show_success(self, text, duration=2500):
        self.setStyleSheet(self._success_style)
        self._show(text, duration)

    def _show(self, text, duration):
        self.setText(text)
        self.adjustSize()
        if self.parentWidget():
            pw = self.parentWidget().width()
            self.move((pw - self.width()) // 2, 16)
        self.show()
        self.raise_()
        self._timer.start(duration)


class LogPanel(QWidget):
    """Scrollable log viewer with color-coded severity levels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Consolas", 9))
        self.log_view.document().setMaximumBlockCount(2000)
        self.log_view.setStyleSheet(
            "QTextEdit { background-color: #16161e; color: #a9b1d6; "
            "border: 1px solid #24283b; border-radius: 4px; }"
        )

        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.log_view.clear)
        btn_row.addStretch()
        btn_row.addWidget(clear_btn)

        layout.addWidget(self.log_view)
        layout.addLayout(btn_row)

    def append_log(self, message: str):
        """Append a log line with color based on severity."""
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

        self.log_view.append(f'<span style="color:{color}">{escaped}</span>')


class ShortcutsDialog(QDialog):
    """Modal dialog listing all keyboard shortcuts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        info = QTextEdit()
        info.setReadOnly(True)
        info.setHtml("""
        <style>
            table { border-collapse: collapse; width: 100%; }
            td { padding: 5px 10px; border-bottom: 1px solid #24283b; }
            .key { color: #7aa2f7; font-family: Consolas, monospace; font-weight: bold; }
            .desc { color: #a9b1d6; }
        </style>
        <table>
        <tr><td class="key">Space</td><td class="desc">Play / Pause</td></tr>
        <tr><td class="key">→</td><td class="desc">Step simulation forward</td></tr>
        <tr><td class="key">R</td><td class="desc">Reset simulation</td></tr>
        <tr><td class="key">F</td><td class="desc">Fit camera to scene</td></tr>
        <tr><td class="key">Ctrl+O</td><td class="desc">Open model file</td></tr>
        <tr><td class="key">Ctrl+S</td><td class="desc">Save screenshot</td></tr>
        <tr><td class="key">Ctrl+Click</td><td class="desc">Select & drag body</td></tr>
        <tr><td class="key">Left Drag</td><td class="desc">Rotate camera</td></tr>
        <tr><td class="key">Middle Drag</td><td class="desc">Pan camera</td></tr>
        <tr><td class="key">Right Drag / Scroll</td><td class="desc">Zoom</td></tr>
        <tr><td class="key">T</td><td class="desc">Toggle body traces</td></tr>
        <tr><td class="key">C</td><td class="desc">Clear traces</td></tr>
        <tr><td class="key">1-8</td><td class="desc">Set speed (0.1x–10x)</td></tr>
        </table>
        """)
        layout.addWidget(info)
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)