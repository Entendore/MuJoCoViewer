"""Configuration tabs: XML editor with find/replace and render options panel."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QPlainTextEdit, QComboBox, QCheckBox,
    QLineEdit, QSlider, QSplitter,
)
from PySide6.QtCore import Qt, Signal
import mujoco
from constants import VIS_FLAGS, CAMERA_PRESETS
from viewport import MujocoViewport
from widgets import log


class XMLEditor(QWidget):
    apply_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Find/Replace bar ──
        self._find_bar = QWidget()
        self._find_bar.setVisible(False)
        fb_lay = QHBoxLayout(self._find_bar)
        fb_lay.setContentsMargins(0, 0, 0, 0)
        fb_lay.setSpacing(4)

        fb_lay.addWidget(QLabel("Find:"))
        self._find_input = QLineEdit()
        self._find_input.setPlaceholderText("Search text…")
        self._find_input.returnPressed.connect(self._find_next)
        fb_lay.addWidget(self._find_input)

        btn_find_next = QPushButton("▼")
        btn_find_next.setFixedWidth(28)
        btn_find_next.setToolTip("Find next")
        btn_find_next.clicked.connect(self._find_next)
        fb_lay.addWidget(btn_find_next)

        fb_lay.addWidget(QLabel("Replace:"))
        self._replace_input = QLineEdit()
        self._replace_input.setPlaceholderText("Replace with…")
        fb_lay.addWidget(self._replace_input)

        btn_replace = QPushButton("Replace")
        btn_replace.clicked.connect(self._replace_one)
        fb_lay.addWidget(btn_replace)

        btn_replace_all = QPushButton("All")
        btn_replace_all.clicked.connect(self._replace_all)
        fb_lay.addWidget(btn_replace_all)

        btn_close_find = QPushButton("✕")
        btn_close_find.setFixedWidth(24)
        btn_close_find.clicked.connect(lambda: self._find_bar.setVisible(False))
        fb_lay.addWidget(btn_close_find)

        layout.addWidget(self._find_bar)

        # ── Editor ──
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("Load a model to view XML…")
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.editor)

        # ── Status bar ──
        self.status_lbl = QLabel("")
        self.status_lbl.setProperty("class", "dim")

        btn_row = QHBoxLayout()

        btn_find = QPushButton("🔍 Find")
        btn_find.setFixedWidth(70)
        btn_find.clicked.connect(lambda: self._find_bar.setVisible(True))
        btn_row.addWidget(btn_find)

        self._wrap_cb = QCheckBox("Wrap")
        self._wrap_cb.setChecked(False)
        self._wrap_cb.toggled.connect(
            lambda v: self.editor.setLineWrapMode(
                QPlainTextEdit.WidgetWidth if v else QPlainTextEdit.NoWrap
            )
        )
        btn_row.addWidget(self._wrap_cb)

        btn_row.addWidget(self.status_lbl)
        btn_row.addStretch()

        self.revert_btn = QPushButton("⟲  Revert")
        self.revert_btn.clicked.connect(self._revert)
        btn_row.addWidget(self.revert_btn)

        self.apply_btn = QPushButton("✔  Apply Changes")
        self.apply_btn.clicked.connect(self.apply_requested.emit)
        self.apply_btn.setStyleSheet(
            "QPushButton { background-color: #2b3f7a; border-color: #3d59a1; }"
            "QPushButton:hover { background-color: #3d59a1; }"
        )
        btn_row.addWidget(self.apply_btn)
        layout.addLayout(btn_row)

        self._saved_xml = ""

    def set_xml(self, xml_string):
        self._saved_xml = xml_string
        self.editor.setPlainText(xml_string)
        self.status_lbl.setText("")
        self.status_lbl.setProperty("class", "dim")

    def _revert(self):
        self.editor.setPlainText(self._saved_xml)
        self.status_lbl.setText("Reverted")
        self.status_lbl.setProperty("class", "dim")

    def set_validation(self, valid, message=""):
        if valid:
            self.status_lbl.setText(f"✓ {message}" if message else "✓ Valid")
            self.status_lbl.setProperty("class", "success")
        else:
            self.status_lbl.setText(f"✗ {message}" if message else "✗ Error")
            self.status_lbl.setProperty("class", "error")
        self.status_lbl.style().unpolish(self.status_lbl)
        self.status_lbl.style().polish(self.status_lbl)

    def _find_next(self):
        text = self._find_input.text()
        if not text:
            return
        cursor = self.editor.document().find(text, self.editor.textCursor())
        if cursor.isNull():
            cursor = self.editor.document().find(text)
        if not cursor.isNull():
            self.editor.setTextCursor(cursor)

    def _replace_one(self):
        text = self._find_input.text()
        replacement = self._replace_input.text()
        cursor = self.editor.textCursor()
        if cursor.selectedText() == text:
            cursor.insertText(replacement)
        self._find_next()

    def _replace_all(self):
        text = self._find_input.text()
        replacement = self._replace_input.text()
        if not text:
            return
        content = self.editor.toPlainText()
        count = content.count(text)
        self.editor.setPlainText(content.replace(text, replacement))
        self.status_lbl.setText(f"Replaced {count} occurrences")


class RenderOptionsPanel(QWidget):
    def __init__(self, viewport: MujocoViewport, parent=None):
        super().__init__(parent)
        self.viewport = viewport
        self._checkboxes = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # ── Camera ──
        cam_group = QGroupBox("Camera")
        cam_lay = QVBoxLayout(cam_group)
        self.cam_combo = QComboBox()
        self.cam_combo.addItem("Free Camera")
        self.cam_combo.currentIndexChanged.connect(self._on_camera_changed)
        cam_lay.addWidget(self.cam_combo)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        for name, az, el in CAMERA_PRESETS:
            self.preset_combo.addItem(name)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self.preset_combo)
        cam_lay.addLayout(preset_row)

        follow_row = QHBoxLayout()
        follow_row.addWidget(QLabel("Follow:"))
        self.follow_combo = QComboBox()
        self.follow_combo.addItem("None")
        self.follow_combo.currentIndexChanged.connect(self._on_follow_changed)
        follow_row.addWidget(self.follow_combo)
        cam_lay.addLayout(follow_row)

        btn_row = QHBoxLayout()
        fit_btn = QPushButton("Fit to Scene")
        fit_btn.clicked.connect(self._fit_camera)
        btn_row.addWidget(fit_btn)
        center_btn = QPushButton("Center")
        center_btn.clicked.connect(self._center_camera)
        btn_row.addWidget(center_btn)
        cam_lay.addLayout(btn_row)
        layout.addWidget(cam_group)

        # ── Overlays ──
        overlay_group = QGroupBox("Overlays")
        overlay_lay = QVBoxLayout(overlay_group)
        self.fps_overlay_cb = QCheckBox("FPS Counter")
        self.fps_overlay_cb.setChecked(True)
        self.fps_overlay_cb.toggled.connect(lambda v: setattr(self.viewport, '_show_fps_overlay', v))
        overlay_lay.addWidget(self.fps_overlay_cb)
        self.axis_overlay_cb = QCheckBox("Axis Indicator")
        self.axis_overlay_cb.setChecked(True)
        self.axis_overlay_cb.toggled.connect(lambda v: setattr(self.viewport, '_show_axis_overlay', v))
        overlay_lay.addWidget(self.axis_overlay_cb)
        self.grid_overlay_cb = QCheckBox("Grid Overlay")
        self.grid_overlay_cb.setChecked(False)
        self.grid_overlay_cb.toggled.connect(lambda v: setattr(self.viewport, '_show_grid_overlay', v))
        overlay_lay.addWidget(self.grid_overlay_cb)
        self.info_overlay_cb = QCheckBox("Selection Info")
        self.info_overlay_cb.setChecked(True)
        self.info_overlay_cb.toggled.connect(lambda v: setattr(self.viewport, '_show_info_overlay', v))
        overlay_lay.addWidget(self.info_overlay_cb)
        layout.addWidget(overlay_group)

        # ── Traces ──
        trace_group = QGroupBox("Body Traces")
        trace_lay = QVBoxLayout(trace_group)
        self.trace_cb = QCheckBox("Show Traces")
        self.trace_cb.setChecked(False)
        self.trace_cb.toggled.connect(self._on_trace_toggled)
        trace_lay.addWidget(self.trace_cb)
        clear_traces_btn = QPushButton("Clear Traces")
        clear_traces_btn.clicked.connect(self.viewport.clear_traces)
        trace_lay.addWidget(clear_traces_btn)
        layout.addWidget(trace_group)

        # ── Visualization ──
        vis_group = QGroupBox("Visualization")
        vis_lay = QVBoxLayout(vis_group)
        for name, flag in VIS_FLAGS:
            cb = QCheckBox(name)
            cb.setChecked(self.viewport.vopt.flags[flag])
            cb.toggled.connect(self._make_flag_cb(flag))
            vis_lay.addWidget(cb)
            self._checkboxes.append(cb)
        layout.addWidget(vis_group)

        # ── Geometry Groups ──
        geom_group = QGroupBox("Geometry Groups")
        geom_lay = QGridLayout(geom_group)
        for g in range(6):
            cb = QCheckBox(f"Group {g}")
            cb.setChecked(True)
            cb.toggled.connect(self._make_geom_cb(g))
            geom_lay.addWidget(cb, g // 3, g % 3)
        layout.addWidget(geom_group)

        # ── Transparency ──
        trans_group = QGroupBox("Rendering")
        trans_lay = QVBoxLayout(trans_group)
        trans_lay.addWidget(QLabel("Stereo:"))
        self.scheme_combo = QComboBox()
        self.scheme_combo.addItems(["None", "Side-by-Side", "Quad Buffered"])
        self.scheme_combo.currentIndexChanged.connect(self._on_scheme_changed)
        trans_lay.addWidget(self.scheme_combo)
        layout.addWidget(trans_group)

        layout.addStretch()

    def populate_cameras(self, model):
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        self.cam_combo.addItem("Free Camera")
        if model is not None:
            for i in range(model.ncam):
                self.cam_combo.addItem(
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                    or f"camera_{i}"
                )
        self.cam_combo.blockSignals(False)
        self.cam_combo.setCurrentIndex(0)

        self.follow_combo.blockSignals(True)
        self.follow_combo.clear()
        self.follow_combo.addItem("None")
        if model is not None:
            for i in range(1, model.nbody):
                name = (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                    or f"body_{i}"
                )
                self.follow_combo.addItem(name)
        self.follow_combo.blockSignals(False)
        self.follow_combo.setCurrentIndex(0)

    def _on_camera_changed(self, idx):
        if idx <= 0:
            self.viewport.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.viewport.cam.fixedcamid = -1
        else:
            self.viewport.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewport.cam.fixedcamid = idx - 1
        self.viewport.render()

    def _on_preset_changed(self, idx):
        if 0 <= idx < len(CAMERA_PRESETS):
            _, az, el = CAMERA_PRESETS[idx]
            self.viewport.set_camera_preset(az, el)

    def _on_follow_changed(self, idx):
        self.viewport.set_follow_body(idx)

    def _fit_camera(self):
        self.viewport.reset_camera()
        self.cam_combo.setCurrentIndex(0)
        self.viewport.render()

    def _center_camera(self):
        if self.viewport.model is not None:
            self.viewport.cam.lookat = self.viewport.model.stat.center.copy()
            self.viewport.render()

    def _on_trace_toggled(self, checked):
        self.viewport._show_traces = checked
        if not checked:
            self.viewport._traces = {}
            self.viewport._trace_counter = 0
        self.viewport.render()

    def _on_scheme_changed(self, idx):
        if self.viewport.renderer is None:
            return
        try:
            if idx == 0:
                self.viewport.renderer.scene.stereo = 0
            elif idx == 1:
                self.viewport.renderer.scene.stereo = 1
        except Exception:
            pass
        self.viewport.render()

    def _make_flag_cb(self, flag):
        def cb(checked):
            self.viewport.vopt.flags[flag] = checked
            self.viewport.render()
        return cb

    def _make_geom_cb(self, group):
        def cb(checked):
            self.viewport.vopt.geomgroup[group] = checked
            self.viewport.render()
        return cb