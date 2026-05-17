"""Actuator control panel with sliders, spin boxes, type labels, and reset buttons."""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QGroupBox, QPushButton, QDoubleSpinBox, QSizePolicy,
)
from PySide6.QtCore import Qt
import mujoco
from constants import ACTUATOR_TYPE_NAMES


class ActuatorPanel(QWidget):
    """Scrollable panel of per-actuator slider + spin-box controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model, self._data = None, None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._actuator_count = 0
        self._reset_cbs: list = []

    def build(self, model, data):
        self._model, self._data = model, data
        # Clear old widgets
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if model is None or model.nu == 0:
            lbl = QLabel("No actuators")
            lbl.setProperty("class", "dim")
            self._layout.addWidget(lbl)
            self._layout.addStretch()
            self._actuator_count = 0
            return

        self._actuator_count = model.nu
        self._reset_cbs = []

        # Header row
        top_row = QHBoxLayout()
        info_lbl = QLabel(f"{model.nu} actuator{'s' if model.nu != 1 else ''}")
        info_lbl.setProperty("class", "dim")
        top_row.addWidget(info_lbl)
        top_row.addStretch()
        reset_all_btn = QPushButton("⟲ Reset All")
        reset_all_btn.setFixedHeight(24)
        reset_all_btn.setProperty("class", "danger")
        reset_all_btn.clicked.connect(self._reset_all)
        top_row.addWidget(reset_all_btn)
        self._layout.addLayout(top_row)

        for i in range(model.nu):
            name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                or f"act_{i}"
            )
            cr = model.actuator_ctrlrange[i]
            lo, hi = float(cr[0]), float(cr[1])
            if hi <= lo:
                hi = lo + 1.0

            atype = int(model.actuator_dyntype[i])
            atype_name = ACTUATOR_TYPE_NAMES.get(atype, "Motor")

            group = QGroupBox(f"{name}")
            gl = QVBoxLayout(group)
            gl.setContentsMargins(6, 6, 6, 6)
            gl.setSpacing(2)

            # Type + range info row
            info_row = QHBoxLayout()
            type_lbl = QLabel(f"[{atype_name}]")
            type_lbl.setProperty("class", "dim")
            type_lbl.setFixedHeight(16)
            info_row.addWidget(type_lbl)
            info_row.addStretch()
            range_lbl = QLabel(f"[{lo:.2f}, {hi:.2f}]")
            range_lbl.setProperty("class", "dim")
            range_lbl.setFixedHeight(16)
            info_row.addWidget(range_lbl)
            gl.addLayout(info_row)

            # Value display
            val_lbl = QLabel(f"{data.ctrl[i]:.3f}")
            val_lbl.setProperty("class", "value")
            gl.addWidget(val_lbl)

            # Slider + SpinBox row
            ctrl_row = QHBoxLayout()
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(1000)
            slider.setValue(
                int(np.clip((data.ctrl[i] - lo) / (hi - lo), 0, 1) * 1000)
            )
            slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            ctrl_row.addWidget(slider, stretch=3)

            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setDecimals(3)
            spin.setSingleStep((hi - lo) / 100.0)
            spin.setValue(float(data.ctrl[i]))
            spin.setFixedWidth(80)
            spin.setAlignment(Qt.AlignRight)
            ctrl_row.addWidget(spin, stretch=1)
            gl.addLayout(ctrl_row)

            # Reset button
            reset_btn = QPushButton("⟲ Reset")
            reset_btn.setFixedHeight(22)
            gl.addWidget(reset_btn)

            # Connect signals with closures
            slider_cb = self._make_slider_cb(i, lo, hi, val_lbl, spin)
            spin_cb = self._make_spin_cb(i, lo, hi, val_lbl, slider)
            reset_cb = self._make_reset_cb(i, lo, hi, slider, val_lbl, spin)

            slider.valueChanged.connect(slider_cb)
            spin.valueChanged.connect(spin_cb)
            reset_btn.clicked.connect(reset_cb)
            self._reset_cbs.append(reset_cb)

            # Color-code initial value
            self._update_value_style(val_lbl, float(data.ctrl[i]), lo, hi)

            group.setMaximumHeight(140)
            self._layout.addWidget(group)

        self._layout.addStretch()

    # ── Callbacks ──────────────────────────────────────────────

    def _reset_all(self):
        for cb in self._reset_cbs:
            cb()

    def _make_slider_cb(self, act_id, lo, hi, lbl, spin):
        """Slider → update data, label, and spin box."""
        def cb(val):
            v = lo + (val / 1000.0) * (hi - lo)
            self._data.ctrl[act_id] = v
            lbl.setText(f"{v:.3f}")
            spin.blockSignals(True)
            spin.setValue(v)
            spin.blockSignals(False)
            self._update_value_style(lbl, v, lo, hi)
        return cb

    def _make_spin_cb(self, act_id, lo, hi, lbl, slider):
        def cb(val):
            self._data.ctrl[act_id] = val
            lbl.setText(f"{val:.3f}")
            slider.blockSignals(True)
            slider.setValue(int(np.clip((val - lo) / (hi - lo), 0, 1) * 1000))
            slider.blockSignals(False)
            self._update_value_style(lbl, val, lo, hi)
        return cb

    def _make_reset_cb(self, act_id, lo, hi, slider, lbl, spin):
        def cb():
            mid = (lo + hi) / 2.0
            self._data.ctrl[act_id] = mid
            lbl.setText(f"{mid:.3f}")
            slider.blockSignals(True)
            slider.setValue(int(np.clip((mid - lo) / (hi - lo), 0, 1) * 1000))
            slider.blockSignals(False)
            spin.blockSignals(True)
            spin.setValue(mid)
            spin.blockSignals(False)
            self._update_value_style(lbl, mid, lo, hi)
        return cb

    @staticmethod
    def _update_value_style(lbl, val, lo, hi):
        mid = (lo + hi) / 2.0
        extent = (hi - lo) / 2.0
        if extent == 0:
            return
        ratio = abs(val - mid) / extent  # 0=center, 1=edge
        if ratio > 0.8:
            lbl.setStyleSheet("color: #f7768e;")
        elif ratio > 0.5:
            lbl.setStyleSheet("color: #e0af68;")
        else:
            lbl.setStyleSheet("color: #7aa2f7;")