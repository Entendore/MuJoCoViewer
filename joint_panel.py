"""Joint inspector panel with sliders, spin boxes, range labels, and reset buttons."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QGroupBox,
    QPushButton, QDoubleSpinBox, QSizePolicy,
)
import mujoco
from constants import JOINT_TYPE_NAMES, QPOS_DIMS
from widgets import log


class JointPanel(QWidget):
    """Scrollable panel showing per-joint state and controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entries: list = []
        self._model = None
        self._data = None
        self._reset_cbs: list = []
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)

    def build(self, model, data):
        self._model, self._data = model, data
        # Clear
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._entries = []
        self._reset_cbs = []

        if model is None or model.njnt == 0:
            lbl = QLabel("No joints")
            lbl.setProperty("class", "dim")
            self._layout.addWidget(lbl)
            self._layout.addStretch()
            return

        # Header + Reset All
        top_row = QHBoxLayout()
        info_lbl = QLabel(f"{model.njnt} joint{'s' if model.njnt != 1 else ''}")
        info_lbl.setProperty("class", "dim")
        top_row.addWidget(info_lbl)
        top_row.addStretch()
        reset_all_btn = QPushButton("⟲ Reset All")
        reset_all_btn.setFixedHeight(24)
        reset_all_btn.setProperty("class", "danger")
        reset_all_btn.clicked.connect(self._reset_all)
        top_row.addWidget(reset_all_btn)
        self._layout.addLayout(top_row)

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
                    reset_row = QHBoxLayout()
                    reset_btn = QPushButton("⟲ Reset")
                    reset_btn.setFixedHeight(20)
                    free_cb = self._make_free_reset_cb(i, qpos_adr)
                    reset_btn.clicked.connect(free_cb)
                    self._reset_cbs.append(free_cb)
                    reset_row.addWidget(reset_btn)
                    reset_row.addStretch()
                    gl.addLayout(reset_row)
                    self._entries.append(("free", i, val_lbl))

                elif jtype == mujoco.mjtJoint.mjJNT_BALL:
                    val_lbl = QLabel("quat: —")
                    val_lbl.setProperty("class", "value")
                    val_lbl.setWordWrap(True)
                    gl.addWidget(val_lbl)
                    reset_row = QHBoxLayout()
                    reset_btn = QPushButton("⟲ Reset")
                    reset_btn.setFixedHeight(20)
                    ball_cb = self._make_ball_reset_cb(i, qpos_adr)
                    reset_btn.clicked.connect(ball_cb)
                    self._reset_cbs.append(ball_cb)
                    reset_row.addWidget(reset_btn)
                    reset_row.addStretch()
                    gl.addLayout(reset_row)
                    self._entries.append(("ball", i, val_lbl))

                else:
                    # Hinge or Slide — scalar DOF with range
                    rng = model.jnt_range[i]
                    has_range = rng[1] > rng[0]
                    lo, hi = float(rng[0]), float(rng[1])

                    if has_range:
                        range_lbl = QLabel(f"Range: [{lo:.2f}, {hi:.2f}]")
                        range_lbl.setProperty("class", "dim")
                        range_lbl.setFixedHeight(16)
                        gl.addWidget(range_lbl)

                    val_lbl = QLabel("—")
                    val_lbl.setProperty("class", "value")
                    gl.addWidget(val_lbl)

                    # Slider + SpinBox row
                    ctrl_row = QHBoxLayout()
                    slider = QSlider(Qt.Horizontal)
                    slider.setMinimum(0)
                    slider.setMaximum(1000)
                    slider.setValue(500)
                    if has_range:
                        v = data.qpos[qpos_adr]
                        slider.setValue(
                            int(np.clip((v - lo) / (hi - lo), 0, 1) * 1000)
                        )
                    else:
                        slider.setEnabled(False)
                    ctrl_row.addWidget(slider, stretch=3)

                    spin = None
                    if has_range:
                        spin = QDoubleSpinBox()
                        spin.setRange(lo, hi)
                        spin.setDecimals(4)
                        spin.setSingleStep((hi - lo) / 100.0)
                        spin.setValue(float(data.qpos[qpos_adr]))
                        spin.setFixedWidth(85)
                        spin.setAlignment(Qt.AlignRight)
                        slider.valueChanged.connect(
                            self._make_slider_to_spin_cb(spin, lo, hi)
                        )
                        spin.valueChanged.connect(
                            self._make_spin_cb(i, qpos_adr, lo, hi, val_lbl, slider)
                        )
                        ctrl_row.addWidget(spin, stretch=1)

                    if has_range:
                        slider.valueChanged.connect(
                            self._make_slider_cb(i, qpos_adr, lo, hi, val_lbl)
                        )

                    gl.addLayout(ctrl_row)

                    reset_row = QHBoxLayout()
                    reset_btn = QPushButton("⟲ Reset")
                    reset_btn.setFixedHeight(20)
                    if has_range:
                        cb = self._make_reset_cb(i, qpos_adr, lo, hi, slider, val_lbl, spin)
                        reset_btn.clicked.connect(cb)
                        self._reset_cbs.append(cb)
                    reset_row.addWidget(reset_btn)
                    reset_row.addStretch()
                    gl.addLayout(reset_row)

                    self._entries.append(
                        ("scalar", i, val_lbl, slider, qpos_adr, has_range, spin)
                    )

                group.setMaximumHeight(150)
                self._layout.addWidget(group)
            except Exception as e:
                log.error(f"Error building joint panel for index {i}: {e}")

        self._layout.addStretch()

    # ── Callbacks ──────────────────────────────────────────────

    def _reset_all(self):
        for cb in self._reset_cbs:
            try:
                cb()
            except Exception:
                pass

    def _make_slider_cb(self, jnt_id, qpos_adr, lo, hi, lbl):
        def cb(val):
            v = lo + (val / 1000.0) * (hi - lo)
            self._data.qpos[qpos_adr] = v
            lbl.setText(f"{v:.4f}")
        return cb

    def _make_slider_to_spin_cb(self, spin, lo, hi):
        def cb(val):
            v = lo + (val / 1000.0) * (hi - lo)
            if spin is not None:
                spin.blockSignals(True)
                spin.setValue(v)
                spin.blockSignals(False)
        return cb

    def _make_spin_cb(self, jnt_id, qpos_adr, lo, hi, lbl, slider):
        def cb(val):
            self._data.qpos[qpos_adr] = val
            lbl.setText(f"{val:.4f}")
            slider.blockSignals(True)
            slider.setValue(int(np.clip((val - lo) / (hi - lo), 0, 1) * 1000))
            slider.blockSignals(False)
        return cb

    def _make_reset_cb(self, jnt_id, qpos_adr, lo, hi, slider, lbl, spin):
        def cb():
            mid = (lo + hi) / 2.0
            self._data.qpos[qpos_adr] = mid
            lbl.setText(f"{mid:.4f}")
            slider.blockSignals(True)
            slider.setValue(int(np.clip((mid - lo) / (hi - lo), 0, 1) * 1000))
            slider.blockSignals(False)
            if spin is not None:
                spin.blockSignals(True)
                spin.setValue(mid)
                spin.blockSignals(False)
        return cb

    def _make_free_reset_cb(self, jnt_id, qpos_adr):
        def cb():
            self._data.qpos[qpos_adr:qpos_adr + 7] = [0, 0, 0, 1, 0, 0, 0]
        return cb

    def _make_ball_reset_cb(self, jnt_id, qpos_adr):
        def cb():
            self._data.qpos[qpos_adr:qpos_adr + 4] = [1, 0, 0, 0]
        return cb

    # ── Refresh (called each tick) ────────────────────────────

    def refresh(self):
        if self._data is None or self._model is None:
            return
        for entry in self._entries:
            try:
                kind = entry[0]
                jnt_id = entry[1]
                lbl = entry[2]
                qpos_adr = self._model.jnt_qposadr[jnt_id]

                if kind == "free":
                    p = self._data.qpos[qpos_adr:qpos_adr + 3]
                    q = self._data.qpos[qpos_adr + 3:qpos_adr + 7]
                    lbl.setText(
                        f"pos: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})\n"
                        f"quat: ({q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f})"
                    )
                elif kind == "ball":
                    q = self._data.qpos[qpos_adr:qpos_adr + 4]
                    lbl.setText(
                        f"quat: ({q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f})"
                    )
                elif kind == "scalar":
                    v = self._data.qpos[qpos_adr]
                    lbl.setText(f"{v:.4f}")
                    slider = entry[3]
                    has_range = entry[5]
                    spin = entry[6] if len(entry) > 6 else None
                    if has_range and not slider.isSliderDown():
                        rng = self._model.jnt_range[jnt_id]
                        lo, hi = rng[0], rng[1]
                        slider.blockSignals(True)
                        slider.setValue(
                            int(np.clip((v - lo) / (hi - lo), 0, 1) * 1000)
                        )
                        slider.blockSignals(False)
                        if spin is not None and not spin.hasFocus():
                            spin.blockSignals(True)
                            spin.setValue(v)
                            spin.blockSignals(False)
            except Exception as e:
                log.debug(f"Error refreshing joint entry: {e}")