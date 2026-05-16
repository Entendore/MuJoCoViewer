"""Actuator control panel with sliders and reset buttons."""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QGroupBox, QPushButton
from PySide6.QtCore import Qt
import mujoco


class ActuatorPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._model, self._data = None, None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)

    def build(self, model, data):
        self._model, self._data = model, data
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if model is None or model.nu == 0:
            lbl = QLabel("No actuators")
            lbl.setProperty("class", "dim")
            self._layout.addWidget(lbl)
            self._layout.addStretch()
            return
        for i in range(model.nu):
            name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                or f"act_{i}"
            )
            cr = model.actuator_ctrlrange[i]
            lo, hi = float(cr[0]), float(cr[1])
            if hi <= lo:
                hi = lo + 1.0
            group = QGroupBox(name)
            gl = QVBoxLayout(group)
            gl.setContentsMargins(6, 6, 6, 6)
            gl.setSpacing(2)

            val_lbl = QLabel(f"{data.ctrl[i]:.2f}")
            val_lbl.setProperty("class", "value")
            gl.addWidget(val_lbl)

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(1000)
            slider.setValue(
                int(np.clip((data.ctrl[i] - lo) / (hi - lo), 0, 1) * 1000)
            )
            slider.valueChanged.connect(self._make_cb(i, lo, hi, val_lbl))
            gl.addWidget(slider)

            reset_btn = QPushButton("⟲ Reset")
            reset_btn.setFixedHeight(22)
            reset_btn.clicked.connect(self._make_reset_cb(i, lo, hi, slider, val_lbl))
            gl.addWidget(reset_btn)

            group.setMaximumHeight(130)
            self._layout.addWidget(group)
        self._layout.addStretch()

    def _make_cb(self, act_id, lo, hi, lbl):
        def cb(val):
            v = lo + (val / 1000.0) * (hi - lo)
            self._data.ctrl[act_id] = v
            lbl.setText(f"{v:.2f}")
        return cb

    def _make_reset_cb(self, act_id, lo, hi, slider, lbl):
        def cb():
            mid = (lo + hi) / 2.0
            self._data.ctrl[act_id] = mid
            lbl.setText(f"{mid:.2f}")
            slider.blockSignals(True)
            slider.setValue(int(np.clip((mid - lo) / (hi - lo), 0, 1) * 1000))
            slider.blockSignals(False)
        return cb