"""Joint inspector panel with sliders for hinge/slide joints."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSlider, QGroupBox, QPushButton, QHBoxLayout,
)
import mujoco
from constants import JOINT_TYPE_NAMES, QPOS_DIMS
from widgets import log


class JointPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries = []
        self._model = None
        self._data = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)

    def build(self, model, data):
        self._model, self._data = model, data
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._entries = []

        if model is None or model.njnt == 0:
            lbl = QLabel("No joints")
            lbl.setProperty("class", "dim")
            self._layout.addWidget(lbl)
            self._layout.addStretch()
            return

        for i in range(model.njnt):
            try:
                name = (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    or f"joint_{i}"
                )
                jtype = model.jnt_type[i]
                jtype_name = JOINT_TYPE_NAMES.get(jtype, "?")
                qpos_adr = model.jnt_qposadr[i]

                group = QGroupBox(f"{name}  ({jtype_name})")
                gl = QVBoxLayout(group)
                gl.setContentsMargins(6, 6, 6, 6)
                gl.setSpacing(2)

                if jtype == mujoco.mjtJoint.mjJNT_FREE:
                    val_lbl = QLabel("pos: —  quat: —")
                    val_lbl.setProperty("class", "value")
                    val_lbl.setWordWrap(True)
                    gl.addWidget(val_lbl)
                    self._entries.append(("free", i, val_lbl))
                elif jtype == mujoco.mjtJoint.mjJNT_BALL:
                    val_lbl = QLabel("quat: —")
                    val_lbl.setProperty("class", "value")
                    val_lbl.setWordWrap(True)
                    gl.addWidget(val_lbl)
                    self._entries.append(("ball", i, val_lbl))
                else:
                    val_lbl = QLabel("—")
                    val_lbl.setProperty("class", "value")
                    gl.addWidget(val_lbl)
                    slider = QSlider(Qt.Horizontal)
                    slider.setMinimum(0)
                    slider.setMaximum(1000)
                    slider.setValue(500)
                    rng = model.jnt_range[i]
                    has_range = rng[1] > rng[0]
                    if has_range:
                        v = data.qpos[qpos_adr]
                        slider.setValue(
                            int(np.clip((v - rng[0]) / (rng[1] - rng[0]), 0, 1) * 1000)
                        )
                        slider.valueChanged.connect(
                            self._make_slider_cb(i, qpos_adr, rng[0], rng[1])
                        )
                    else:
                        slider.setEnabled(False)
                    gl.addWidget(slider)

                    reset_row = QHBoxLayout()
                    reset_btn = QPushButton("⟲ Reset")
                    reset_btn.setFixedHeight(20)
                    if has_range:
                        reset_btn.clicked.connect(
                            self._make_reset_cb(i, qpos_adr, rng[0], rng[1], slider)
                        )
                    gl.addWidget(reset_btn)

                    self._entries.append(("scalar", i, val_lbl, slider, qpos_adr, has_range))

                group.setMaximumHeight(130)
                self._layout.addWidget(group)
            except Exception as e:
                log.error(f"Error building joint panel for index {i}: {e}")
        self._layout.addStretch()

    def _make_slider_cb(self, jnt_id, qpos_adr, lo, hi):
        def cb(val):
            self._data.qpos[qpos_adr] = lo + (val / 1000.0) * (hi - lo)
        return cb

    def _make_reset_cb(self, jnt_id, qpos_adr, lo, hi, slider):
        def cb():
            mid = (lo + hi) / 2.0
            self._data.qpos[qpos_adr] = mid
            slider.blockSignals(True)
            slider.setValue(int(np.clip((mid - lo) / (hi - lo), 0, 1) * 1000))
            slider.blockSignals(False)
        return cb

    def refresh(self):
        if self._data is None or self._model is None:
            return
        for entry in self._entries:
            try:
                kind, jnt_id, lbl = entry[0], entry[1], entry[2]
                qpos_adr = self._model.jnt_qposadr[jnt_id]

                if kind == "free":
                    p = self._data.qpos[qpos_adr: qpos_adr + 3]
                    q = self._data.qpos[qpos_adr + 3: qpos_adr + 7]
                    lbl.setText(
                        f"pos: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})\n"
                        f"quat: ({q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f})"
                    )
                elif kind == "ball":
                    q = self._data.qpos[qpos_adr: qpos_adr + 4]
                    lbl.setText(f"quat: ({q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f})")
                elif kind == "scalar":
                    v = self._data.qpos[qpos_adr]
                    lbl.setText(f"{v:.4f}")
                    slider, has_range = entry[3], entry[5]
                    if has_range and not slider.isSliderDown():
                        rng = self._model.jnt_range[jnt_id]
                        lo, hi = rng[0], rng[1]
                        slider.blockSignals(True)
                        slider.setValue(int(np.clip((v - lo) / (hi - lo), 0, 1) * 1000))
                        slider.blockSignals(False)
            except Exception as e:
                log.debug(f"Error refreshing joint entry: {e}")