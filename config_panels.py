"""Configuration tabs: XML editor and render options panel."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QPlainTextEdit, QComboBox, QCheckBox,
)
from PySide6.QtCore import Signal
import mujoco
from constants import VIS_FLAGS
from viewport import MujocoViewport


class XMLEditor(QWidget):
    apply_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("Load a model to view XML…")
        layout.addWidget(self.editor)

        self.status_lbl = QLabel("")
        self.status_lbl.setProperty("class", "dim")

        btn_row = QHBoxLayout()
        self.apply_btn = QPushButton("✔  Apply Changes")
        self.apply_btn.clicked.connect(self.apply_requested.emit)
        self.revert_btn = QPushButton("⟲  Revert")
        self.revert_btn.clicked.connect(self._revert)
        btn_row.addWidget(self.status_lbl)
        btn_row.addStretch()
        btn_row.addWidget(self.revert_btn)
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


class RenderOptionsPanel(QWidget):
    def __init__(self, viewport: MujocoViewport, parent=None):
        super().__init__(parent)
        self.viewport = viewport
        self._checkboxes = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        cam_group = QGroupBox("Camera")
        cam_lay = QVBoxLayout(cam_group)
        self.cam_combo = QComboBox()
        self.cam_combo.addItem("Free Camera")
        self.cam_combo.currentIndexChanged.connect(self._on_camera_changed)
        cam_lay.addWidget(self.cam_combo)
        fit_btn = QPushButton("Fit to Scene")
        fit_btn.clicked.connect(self._fit_camera)
        cam_lay.addWidget(fit_btn)
        layout.addWidget(cam_group)

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

        vis_group = QGroupBox("Visualization")
        vis_lay = QVBoxLayout(vis_group)
        for name, flag in VIS_FLAGS:
            cb = QCheckBox(name)
            cb.setChecked(self.viewport.vopt.flags[flag])
            cb.toggled.connect(self._make_flag_cb(flag))
            vis_lay.addWidget(cb)
            self._checkboxes.append(cb)
        layout.addWidget(vis_group)

        geom_group = QGroupBox("Geometry Groups")
        geom_lay = QGridLayout(geom_group)
        for g in range(6):
            cb = QCheckBox(f"Group {g}")
            cb.setChecked(True)
            cb.toggled.connect(self._make_geom_cb(g))
            geom_lay.addWidget(cb, g // 3, g % 3)
        layout.addWidget(geom_group)

        scheme_group = QGroupBox("Render Scheme")
        scheme_lay = QVBoxLayout(scheme_group)
        self.scheme_combo = QComboBox()
        self.scheme_combo.addItems(["Default", "White", "Gray"])
        self.scheme_combo.currentIndexChanged.connect(self._on_scheme_changed)
        scheme_lay.addWidget(self.scheme_combo)
        layout.addWidget(scheme_group)

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

    def _on_camera_changed(self, idx):
        if idx <= 0:
            self.viewport.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.viewport.cam.fixedcamid = -1
        else:
            self.viewport.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.viewport.cam.fixedcamid = idx - 1
        self.viewport.render()

    def _fit_camera(self):
        self.viewport.reset_camera()
        self.cam_combo.setCurrentIndex(0)
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