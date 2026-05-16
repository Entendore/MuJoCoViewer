"""Body tree, watch panel, sensor panel, energy, and contacts inspector panels."""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QGroupBox,
    QTreeWidget, QTreeWidgetItem, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QComboBox, QSpinBox,
)
from PySide6.QtCore import Qt, Signal
import mujoco
from widgets import log


class BodyTreePanel(QWidget):
    focus_body = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Body", "Joints", "Geoms", "Mass"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setColumnWidth(0, 160)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Joints", "No Children"])
        self.filter_combo.setMaximumWidth(120)
        filter_row.addWidget(self.filter_combo)
        filter_row.addStretch()
        expand_btn = QPushButton("Expand All")
        expand_btn.clicked.connect(self.tree.expandAll)
        filter_row.addWidget(expand_btn)
        collapse_btn = QPushButton("Collapse")
        collapse_btn.clicked.connect(self.tree.collapseAll)
        filter_row.addWidget(collapse_btn)
        layout.addLayout(filter_row)

        layout.addWidget(self.tree)

        self._model = None

    def build(self, model):
        self._model = model
        self.tree.clear()
        if model is None:
            return
        items = {}
        for i in range(model.nbody):
            name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                or f"body_{i}"
            )
            parent = model.body_parentid[i]
            njnt = sum(1 for j in range(model.njnt) if model.jnt_bodyid[j] == i)
            ngeom = sum(1 for g in range(model.ngeom) if model.geom_bodyid[g] == i)
            mass = float(model.body_mass[i])
            item = QTreeWidgetItem([
                name,
                str(njnt) if njnt else "—",
                str(ngeom) if ngeom else "—",
                f"{mass:.2f}" if mass > 0.001 else "—",
            ])
            item.setData(0, Qt.UserRole, i)
            if i == 0:
                font = item.font(0)
                font.setBold(True)
                item.setFont(0, font)
            if parent == -1 or parent not in items:
                self.tree.addTopLevelItem(item)
            else:
                items[parent].addChild(item)
            items[i] = item
        self.tree.expandAll()

    def _on_double_click(self, item, column):
        body_id = item.data(0, Qt.UserRole)
        if body_id is not None and body_id > 0:
            self.focus_body.emit(body_id)

    def _on_context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None:
            return
        body_id = item.data(0, Qt.UserRole)
        if body_id is not None and body_id > 0:
            self.focus_body.emit(body_id)


class WatchPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._model, self._data = None, None
        self._watches = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        add_row = QHBoxLayout()
        add_row.addWidget(QLabel("Add:"))
        self.cat_combo = QComboBox()
        self.cat_combo.addItems(["qpos", "qvel", "ctrl", "act", "sensordata"])
        add_row.addWidget(self.cat_combo)

        add_row.addWidget(QLabel("Idx:"))
        self.idx_spin = QSpinBox()
        self.idx_spin.setRange(0, 9999)
        self.idx_spin.setValue(0)
        self.idx_spin.setFixedWidth(60)
        add_row.addWidget(self.idx_spin)

        add_btn = QPushButton("+ Add")
        add_btn.clicked.connect(self._add_watch)
        add_row.addWidget(add_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_watches)
        add_row.addWidget(clear_btn)

        layout.addLayout(add_row)

        self.watch_table = QTableWidget()
        self.watch_table.setColumnCount(3)
        self.watch_table.setHorizontalHeaderLabels(["Variable", "Value", ""])
        self.watch_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.watch_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.watch_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.watch_table.setColumnWidth(2, 30)
        self.watch_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.watch_table.setAlternatingRowColors(True)
        self.watch_table.verticalHeader().setVisible(False)
        layout.addWidget(self.watch_table)

        layout.addStretch()

    def build(self, model, data):
        self._model, self._data = model, data

    def _add_watch(self):
        cat = self.cat_combo.currentText()
        idx = self.idx_spin.value()
        for w_cat, w_idx, _ in self._watches:
            if w_cat == cat and w_idx == idx:
                return
        label = f"{cat}[{idx}]"
        self._watches.append((cat, idx, label))
        self._rebuild_table()

    def _clear_watches(self):
        self._watches.clear()
        self.watch_table.setRowCount(0)

    def _rebuild_table(self):
        self.watch_table.setRowCount(len(self._watches))
        for i, (cat, idx, label) in enumerate(self._watches):
            self.watch_table.setItem(i, 0, QTableWidgetItem(label))
            self.watch_table.setItem(i, 1, QTableWidgetItem("—"))
            remove_btn = QPushButton("✕")
            remove_btn.setFixedWidth(26)
            remove_btn.clicked.connect(lambda checked, row=i: self._remove_watch(row))
            self.watch_table.setCellWidget(i, 2, remove_btn)

    def _remove_watch(self, row):
        if 0 <= row < len(self._watches):
            self._watches.pop(row)
            self._rebuild_table()

    def refresh(self):
        if self._data is None:
            return
        for i, (cat, idx, label) in enumerate(self._watches):
            try:
                arr = getattr(self._data, cat, None)
                if arr is not None and idx < len(arr):
                    val = float(arr[idx])
                    item = self.watch_table.item(i, 1)
                    if item:
                        item.setText(f"{val:.6f}")
            except Exception:
                pass


class SensorPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Sensor", "Type", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setColumnWidth(1, 80)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        self._nsensor = 0
        self._sensor_adr = []
        self._sensor_dim = []

    def build(self, model, data):
        if model is None:
            self.table.setRowCount(0)
            return
        nsensor = model.nsensor
        self._nsensor = nsensor
        self._sensor_adr = []
        self._sensor_dim = []
        self.table.setRowCount(nsensor)
        for i in range(nsensor):
            name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
                or f"sensor_{i}"
            )
            stype = int(model.sensor_type[i])
            type_names = {
                0: "None", 1: "Magnetometer", 2: "Gyro", 3: "Accelerometer",
                4: "Velocimeter", 5: "GyroF", 6: "AccelerometerF",
                7: "VelocimeterF", 8: "Force", 9: "Torque", 10: "ForceF",
                11: "TorqueF", 12: "JointPos", 13: "JointVel",
                14: "TendonPos", 15: "TendonVel", 16: "ActuatorPos",
                17: "ActuatorVel", 18: "ActuatorFrc",
                19: "BallJointAng", 20: "BallJointVel",
                21: "JointLimitPos", 22: "JointLimitVel",
                23: "TendonLimitPos", 24: "TendonLimitVel",
                25: "FramePos", 26: "FrameQuat", 27: "FrameXaxis",
                28: "FrameYaxis", 29: "FrameZaxis",
                30: "FrameLinVel", 31: "FrameAngVel",
                32: "FrameLinAcc", 33: "FrameAngAcc",
                34: "SubtreeCom", 35: "SubtreeLinVel", 36: "SubtreeAngMom",
                100: "User",
            }
            type_name = type_names.get(stype, f"type_{stype}")
            dim = int(model.sensor_dim[i])
            self._sensor_adr.append(int(model.sensor_adr[i]))
            self._sensor_dim.append(dim)
            self.table.setItem(i, 0, QTableWidgetItem(name))
            self.table.setItem(i, 1, QTableWidgetItem(type_name))
            self.table.setItem(i, 2, QTableWidgetItem("—"))

    def refresh(self, data):
        if data is None or not self._sensor_adr:
            return
        for i in range(self._nsensor):
            try:
                adr = self._sensor_adr[i]
                dim = self._sensor_dim[i]
                if dim == 1:
                    val = f"{float(data.sensordata[adr]):.4f}"
                elif dim <= 3:
                    vals = [f"{float(data.sensordata[adr + j]):.4f}" for j in range(dim)]
                    val = f"({', '.join(vals)})"
                else:
                    val = f"[{dim} values]"
                item = self.table.item(i, 2)
                if item:
                    item.setText(val)
            except Exception:
                pass


class EnergyPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.kinetic_lbl = QLabel("0.0 J")
        self.kinetic_lbl.setProperty("class", "value")
        self.kinetic_bar = QProgressBar()

        self.potential_lbl = QLabel("0.0 J")
        self.potential_lbl.setProperty("class", "value")
        self.potential_bar = QProgressBar()

        self.total_lbl = QLabel("0.0 J")
        self.total_lbl.setProperty("class", "value")
        self.total_bar = QProgressBar()

        for bar in [self.kinetic_bar, self.potential_bar, self.total_bar]:
            bar.setMinimum(-100)
            bar.setMaximum(100)
            bar.setValue(0)
            bar.setFormat("%v J")

        gk = QGroupBox("Kinetic")
        gkl = QVBoxLayout(gk)
        gkl.addWidget(self.kinetic_lbl)
        gkl.addWidget(self.kinetic_bar)

        gp = QGroupBox("Potential")
        gpl = QVBoxLayout(gp)
        gpl.addWidget(self.potential_lbl)
        gpl.addWidget(self.potential_bar)

        gt = QGroupBox("Total")
        gtl = QVBoxLayout(gt)
        gtl.addWidget(self.total_lbl)
        gtl.addWidget(self.total_bar)

        layout.addWidget(gk)
        layout.addWidget(gp)
        layout.addWidget(gt)
        layout.addStretch()

    def refresh(self, model, data):
        if model is None or data is None:
            return
        try:
            if not (model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_ENERGY):
                model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
                mujoco.mj_forward(model, data)
            ke, pe = float(data.energy[0]), float(data.energy[1])
            total = ke + pe
            self.kinetic_lbl.setText(f"{ke:.3f} J")
            self.potential_lbl.setText(f"{pe:.3f} J")
            self.total_lbl.setText(f"{total:.3f} J")
            max_e = max(abs(ke), abs(pe), abs(total), 1.0) * 1.2
            for bar, val in [
                (self.kinetic_bar, ke),
                (self.potential_bar, pe),
                (self.total_bar, total),
            ]:
                bar.setRange(int(-max_e), int(max_e))
                bar.setValue(int(val))
                bar.setFormat(f"{val:.2f} J")
        except Exception:
            pass


class ContactsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Geom 1", "Geom 2", "Distance", "Force"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

    def refresh(self, model, data):
        if model is None or data is None:
            return
        ncon = data.ncon
        self.table.setRowCount(ncon)
        for i in range(ncon):
            c = data.contact[i]
            g1_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
                or f"geom_{c.geom1}"
            )
            g2_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
                or f"geom_{c.geom2}"
            )
            self.table.setItem(i, 0, QTableWidgetItem(g1_name))
            self.table.setItem(i, 1, QTableWidgetItem(g2_name))
            self.table.setItem(i, 2, QTableWidgetItem(f"{c.dist:.5f}"))
            try:
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                f_norm = float(np.linalg.norm(force[:3]))
                self.table.setItem(i, 3, QTableWidgetItem(f"{f_norm:.2f}"))
            except Exception:
                self.table.setItem(i, 3, QTableWidgetItem("—"))