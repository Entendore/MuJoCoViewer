"""Small reusable widgets: toast notifications, shortcuts dialog, logger, log panel, FPS/RTF graph."""

import logging
from collections import deque
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel, QWidget,
    QComboBox, QCheckBox, QLineEdit,
)
from PySide6.QtCore import Qt, QTimer, QObject, Signal, QPoint
from PySide6.QtGui import QFont, QPainter, QColor, QPen, QBrush, QLinearGradient, QPolygon


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
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
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
    """Non-blocking toast notification with queue and style variants."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self._styles = {
            "default": (
                "background-color: rgba(41, 66, 141, 220); color: #ffffff; "
                "border: 1px solid rgba(122, 162, 247, 120); "
                "border-radius: 10px; padding: 10px 24px; font-weight: bold; font-size: 14px;"
            ),
            "error": (
                "background-color: rgba(160, 40, 60, 220); color: #ffffff; "
                "border: 1px solid rgba(247, 118, 142, 120); "
                "border-radius: 10px; padding: 10px 24px; font-weight: bold; font-size: 14px;"
            ),
            "success": (
                "background-color: rgba(40, 100, 50, 220); color: #ffffff; "
                "border: 1px solid rgba(158, 206, 106, 120); "
                "border-radius: 10px; padding: 10px 24px; font-weight: bold; font-size: 14px;"
            ),
            "warning": (
                "background-color: rgba(140, 100, 20, 220); color: #ffffff; "
                "border: 1px solid rgba(224, 175, 104, 120); "
                "border-radius: 10px; padding: 10px 24px; font-weight: bold; font-size: 14px;"
            ),
        }
        self.setStyleSheet(self._styles["default"])
        self.hide()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._show_next)
        self._queue: deque = deque()

    def show_message(self, text, duration=2500):
        self._enqueue(text, "default", duration)

    def show_error(self, text, duration=4000):
        self._enqueue(text, "error", duration)

    def show_success(self, text, duration=2500):
        self._enqueue(text, "success", duration)

    def show_warning(self, text, duration=3000):
        self._enqueue(text, "warning", duration)

    def _enqueue(self, text, style_key, duration):
        self._queue.append((text, self._styles[style_key], duration))
        if not self._timer.isActive():
            self._show_next()

    def _show_next(self):
        if not self._queue:
            self.hide()
            return
        text, style, duration = self._queue.popleft()
        self.setStyleSheet(style)
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
    """Scrolling log viewer with level filter, text search, and copy."""

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
        self.filter_combo.currentIndexChanged.connect(self._apply_filter)
        filter_row.addWidget(self.filter_combo)

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
            "QTextEdit { background-color: #0d0e16; color: #c0caf5; "
            "border: 1px solid #2a2f45; border-radius: 4px; padding: 4px; }"
        )
        layout.addWidget(self.log_view)
        self._all_messages: list[tuple[str, str]] = []

    def append_log(self, message: str):
        color = "#c0caf5"
        for tag, c in [("[ERROR]", "#f7768e"), ("[WARNING]", "#e0af68"),
                        ("[INFO]", "#9ece6a"), ("[DEBUG]", "#565f89")]:
            if tag in message:
                color = c
                break
        self._all_messages.append((message, color))
        self._apply_filter()
        if self.auto_scroll_cb.isChecked():
            sb = self.log_view.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _apply_filter(self):
        level = self.filter_combo.currentText().lower()
        search = self.search_input.text().strip().lower()
        self.log_view.clear()
        for msg, color in self._all_messages:
            if level != "all":
                level_tag = f"[{level.upper()}]"
                if level_tag not in msg:
                    continue
            if search and search not in msg.lower():
                continue
            escaped = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            self.log_view.append(f'<span style="color:{color}">{escaped}</span>')

    def _copy_visible(self):
        text = self.log_view.toPlainText()
        if text:
            from PySide6.QtWidgets import QApplication
            QApplication.clipboard().setText(text)

    def _clear(self):
        self.log_view.clear()
        self._all_messages.clear()


# ── FPS / RTF Graph Widget ────────────────────────────────────
class FPSGraph(QWidget):
    """Real-time FPS and RTF sparkline graph with gradient fills and grid."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(80)
        self._fps_history: deque = deque(maxlen=150)
        self._rtf_history: deque = deque(maxlen=150)
        self.setMinimumWidth(200)

    def add_fps(self, fps, rtf=1.0):
        self._fps_history.append(fps)
        self._rtf_history.append(rtf)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            w, h = self.width(), self.height()

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor("#0d0e16")))
            painter.drawRoundedRect(0, 0, w, h, 6, 6)

            painter.setPen(QPen(QColor("#2a2f45"), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(0, 0, w - 1, h - 1, 6, 6)

            margin_t, margin_b, margin_l, margin_r = 18, 6, 8, 8
            plot_w = w - margin_l - margin_r
            plot_h = h - margin_t - margin_b

            if len(self._fps_history) < 2:
                painter.setPen(QColor("#565f89"))
                painter.setFont(QFont("Consolas", 9))
                painter.drawText(self.rect(), Qt.AlignCenter, "No FPS data")
                return

            max_fps = max(max(self._fps_history), 60)
            n = len(self._fps_history)
            dx = plot_w / max(n - 1, 1)

            y60 = margin_t + int(((max_fps - 60) / max_fps) * plot_h)
            painter.setPen(QPen(QColor(158, 206, 106, 50), 1, Qt.DashLine))
            painter.drawLine(margin_l, y60, w - margin_r, y60)

            painter.setPen(QPen(QColor("#1e2235"), 1, Qt.DotLine))
            for frac in [0.25, 0.5, 0.75]:
                gy = margin_t + int(frac * plot_h)
                painter.drawLine(margin_l, gy, w - margin_r, gy)

            fps_points = []
            for i, fps in enumerate(self._fps_history):
                x = margin_l + int(i * dx)
                y = margin_t + int(((max_fps - fps) / max_fps) * plot_h)
                fps_points.append((x, y))

            fps_fill = QColor("#7aa2f7")
            fps_fill.setAlpha(25)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(fps_fill))
            poly = QPolygon()
            poly.append(QPoint(fps_points[0][0], fps_points[0][1]))
            for px, py in fps_points[1:]:
                poly.append(QPoint(px, py))
            poly.append(QPoint(fps_points[-1][0], margin_t + plot_h))
            poly.append(QPoint(fps_points[0][0], margin_t + plot_h))
            painter.drawPolygon(poly)

            painter.setPen(QPen(QColor("#7aa2f7"), 1.8))
            for i in range(len(fps_points) - 1):
                painter.drawLine(fps_points[i][0], fps_points[i][1],
                                 fps_points[i + 1][0], fps_points[i + 1][1])

            if len(self._rtf_history) >= 2:
                max_rtf = max(max(self._rtf_history), 1.5)
                rtf_points = []
                for i, rtf in enumerate(self._rtf_history):
                    x = margin_l + int(i * dx)
                    y = margin_t + int(((max_rtf - rtf) / max_rtf) * (plot_h * 0.5))
                    rtf_points.append((x, y))
                painter.setPen(QPen(QColor(224, 175, 104, 140), 1.2, Qt.DashLine))
                for i in range(len(rtf_points) - 1):
                    painter.drawLine(rtf_points[i][0], rtf_points[i][1],
                                     rtf_points[i + 1][0], rtf_points[i + 1][1])

            last_fps = self._fps_history[-1]
            last_rtf = self._rtf_history[-1] if self._rtf_history else 1.0

            painter.setFont(QFont("Consolas", 9, QFont.Bold))
            fps_color = QColor("#7aa2f7")
            if last_fps < 15:
                fps_color = QColor("#f7768e")
            elif last_fps < 30:
                fps_color = QColor("#e0af68")
            painter.setPen(fps_color)
            painter.drawText(margin_l + 4, 13, f"{last_fps:.0f} FPS")

            painter.setFont(QFont("Consolas", 8))
            painter.setPen(QColor("#e0af68"))
            painter.drawText(w - margin_r - 75, 13, f"RTF {last_rtf:.2f}x")
        finally:
            painter.end()


# ── Shortcuts Dialog ──────────────────────────────────────────
class ShortcutsDialog(QDialog):
    """Modal dialog showing all keyboard shortcuts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumWidth(480)
        self.setStyleSheet(
            "QDialog { background-color: #1a1b26; }"
            "QTextEdit { background-color: #0d0e16; color: #c0caf5; "
            "border: 1px solid #2a2f45; border-radius: 6px; padding: 12px; }"
            "QPushButton { background-color: #24283b; color: #c0caf5; "
            "border: 1px solid #3b4261; border-radius: 6px; padding: 8px 24px; }"
            "QPushButton:hover { background-color: #3d59a1; }"
        )
        layout = QVBoxLayout(self)
        info = QTextEdit()
        info.setReadOnly(True)
        info.setHtml("""
        <style>
            table { border-collapse: collapse; width: 100%; }
            td { padding: 6px 12px; border-bottom: 1px solid #2a2f45; font-size: 13px; }
            .key { color: #7aa2f7; font-family: Consolas, monospace; font-weight: bold; }
            .desc { color: #c0caf5; }
            h3 { color: #7aa2f7; font-size: 15px; margin-top: 16px; }
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
        <tr><td class="key">Ctrl+Shift+S</td><td class="desc">Save screenshot</td></tr>
        </table>
        """)
        layout.addWidget(info)
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)