"""Body tree, watch panel, sensor panel, energy, keyframe, and contacts inspector panels."""

import numpy as np
from collections import deque
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QGroupBox,
    QTreeWidget, QTreeWidgetItem, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QComboBox, QSpinBox, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPainter, QColor, QPen
import mujoco
from widgets import log


# ═══════════════════════════════════════════════════════════════
#  Body Tree Panel
# ═══════════════════════════════════════════════════════════════

class BodyTreePanel(QWidget):
    focus_body = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Search filter
        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Filter bodies…")
        self.search_input.setProperty("class", "search")
        self.search_input.textChanged.connect(self._apply_filter)
        search_row.addWidget(self.search_input)
        layout.addLayout(search_row)

        # Filter combo + expand/collapse
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Joints", "No Children"])
        self.filter_combo.setMaximumWidth(120)
        self.filter_combo.currentIndexChanged.connect(self._apply_filter)
        filter_row.addWidget(self.filter_combo)
        filter_row.addStretch()
        expand_btn = QPushButton("Expand All")
        expand_btn.clicked.connect(lambda: self.tree.expandAll())
        filter_row.addWidget(expand_btn)
        collapse_btn = QPushButton("Collapse")
        collapse_btn.clicked.connect(lambda: self.tree.collapseAll())
        filter_row.addWidget(collapse_btn)
        layout.addLayout(filter_row)

        # Tree
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Body", "Joints", "Geoms", "Mass"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setColumnWidth(0, 160)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self.tree)

        self._model = None
        self._all_items: dict = {}

    def build(self, model):
        self._model = model
        self.tree.clear()
        self._all_items = {}
        if model is None:
            return
        items: dict[int, QTreeWidgetItem] = {}
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
            self._all_items[i] = (name, njnt, ngeom, item)
        self.tree.expandAll()

    def _apply_filter(self):
        search = self.search_input.text().strip().lower()
        filter_mode = self.filter_combo.currentText()
        for body_id, (name, njnt, ngeom, item) in self._all_items.items():
            hidden = False
            if search and search not in name.lower():
                hidden = True
            if filter_mode == "Joints" and njnt == 0:
                hidden = True
            elif filter_mode == "No Children":
                has_child = any(
                    self._model.body_parentid[bid] == body_id
                    for bid in range(self._model.nbody)
                ) if self._model else False
                if has_child:
                    hidden = True
            item.setHidden(hidden)

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


# ═══════════════════════════════════════════════════════════════
#  Watch Panel
# ═══════════════════════════════════════════════════════════════

class WatchPanel(QWidget):
    """User-configurable variable watcher (qpos, qvel, ctrl, act, sensordata)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model, self._data = None, None
        self._watches: list[tuple[str, int, str]] = []

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

        # Quick-add presets
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Quick:"))
        btn_qpos = QPushButton("All qpos")
        btn_qpos.clicked.connect(self._add_all_qpos)
        preset_row.addWidget(btn_qpos)
        btn_ctrl = QPushButton("All ctrl")
        btn_ctrl.clicked.connect(self._add_all_ctrl)
        preset_row.addWidget(btn_ctrl)
        preset_row.addStretch()
        layout.addLayout(preset_row)

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
        if any(w[0] == cat and w[1] == idx for w in self._watches):
            return
        label = f"{cat}[{idx}]"
        self._watches.append((cat, idx, label))
        self._rebuild_table()

    def _add_all_qpos(self):
        if self._data is None:
            return
        for i in range(len(self._data.qpos)):
            if not any(w[0] == "qpos" and w[1] == i for w in self._watches):
                self._watches.append(("qpos", i, f"qpos[{i}]"))
        self._rebuild_table()

    def _add_all_ctrl(self):
        if self._data is None:
            return
        for i in range(len(self._data.ctrl)):
            if not any(w[0] == "ctrl" and w[1] == i for w in self._watches):
                self._watches.append(("ctrl", i, f"ctrl[{i}]"))
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


# ═══════════════════════════════════════════════════════════════
#  Sensor Panel
# ═══════════════════════════════════════════════════════════════

class SensorPanel(QWidget):
    _SENSOR_TYPE_NAMES = {
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
        self._sensor_adr: list[int] = []
        self._sensor_dim: list[int] = []

    def build(self, model, data):
        if model is None:
            self.table.setRowCount(0)
            return
        self._nsensor = model.nsensor
        self._sensor_adr = []
        self._sensor_dim = []
        self.table.setRowCount(self._nsensor)
        for i in range(self._nsensor):
            name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
                or f"sensor_{i}"
            )
            stype = int(model.sensor_type[i])
            type_name = self._SENSOR_TYPE_NAMES.get(stype, f"type_{stype}")
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


# ═══════════════════════════════════════════════════════════════
#  Energy Panel
# ═══════════════════════════════════════════════════════════════

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

        self._history_graph = EnergyHistoryGraph()
        layout.addWidget(self._history_graph)
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
            self._history_graph.add_data(ke, pe, total)
        except Exception:
            pass


class EnergyHistoryGraph(QWidget):
    """Sparkline graph of kinetic, potential, and total energy over time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(80)
        self._ke: deque = deque(maxlen=200)
        self._pe: deque = deque(maxlen=200)
        self._total: deque = deque(maxlen=200)

    def add_data(self, ke, pe, total):
        self._ke.append(ke)
        self._pe.append(pe)
        self._total.append(total)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), QColor("#16161e"))
            painter.setPen(QPen(QColor("#24283b"), 1))
            painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

            if len(self._total) < 2:
                painter.setPen(QColor("#565f89"))
                painter.setFont(QFont("Consolas", 8))
                painter.drawText(self.rect(), Qt.AlignCenter, "Energy history")
                return

            w, h = self.width(), self.height()
            all_vals = list(self._ke) + list(self._pe) + list(self._total)
            min_v, max_v = min(all_vals), max(all_vals)
            span = max_v - min_v if max_v != min_v else 1.0

            for data, color in [
                (self._ke, "#f7768e"),
                (self._pe, "#7aa2f7"),
                (self._total, "#9ece6a"),
            ]:
                painter.setPen(QPen(QColor(color), 1.2))
                n = len(data)
                dx = w / max(n - 1, 1)
                points = [
                    (int(i * dx), h - int(((v - min_v) / span) * (h - 8)) - 4)
                    for i, v in enumerate(data)
                ]
                for i in range(len(points) - 1):
                    painter.drawLine(points[i][0], points[i][1],
                                     points[i + 1][0], points[i + 1][1])

            painter.setFont(QFont("Consolas", 7))
            for x, label, color in [(4, "KE", "#f7768e"),
                                     (24, "PE", "#7aa2f7"),
                                     (44, "Tot", "#9ece6a")]:
                painter.setPen(QColor(color))
                painter.drawText(x, 10, label)
        finally:
            painter.end()


# ═══════════════════════════════════════════════════════════════
#  Contacts Panel
# ═══════════════════════════════════════════════════════════════

class ContactsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header_row = QHBoxLayout()
        self.count_lbl = QLabel("0 contacts")
        self.count_lbl.setProperty("class", "dim")
        header_row.addWidget(self.count_lbl)
        header_row.addStretch()
        self.auto_refresh_cb = QComboBox()
        self.auto_refresh_cb.addItems(["Auto", "Every 5s", "Manual"])
        self.auto_refresh_cb.setMaximumWidth(100)
        header_row.addWidget(self.auto_refresh_cb)
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setFixedWidth(80)
        refresh_btn.clicked.connect(lambda: setattr(self, '_force_refresh_flag', True))
        header_row.addWidget(refresh_btn)
        layout.addLayout(header_row)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Geom 1", "Geom 2", "Distance", "Force", "Normal"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        self._refresh_counter = 0
        self._force_refresh_flag = False

    def refresh(self, model, data):
        if model is None or data is None:
            return

        mode = self.auto_refresh_cb.currentText()
        if mode == "Manual" and not self._force_refresh_flag:
            return
        if mode == "Every 5s":
            self._refresh_counter += 1
            if self._refresh_counter % 300 != 0 and not self._force_refresh_flag:
                return

        self._force_refresh_flag = False
        ncon = data.ncon
        self.count_lbl.setText(f"{ncon} contact{'s' if ncon != 1 else ''}")
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

            dist_item = QTableWidgetItem(f"{c.dist:.5f}")
            if c.dist < 0:
                dist_item.setForeground(QColor("#f7768e"))
            elif c.dist < 0.001:
                dist_item.setForeground(QColor("#e0af68"))
            else:
                dist_item.setForeground(QColor("#9ece6a"))
            self.table.setItem(i, 2, dist_item)

            try:
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                f_norm = float(np.linalg.norm(force[:3]))
                force_item = QTableWidgetItem(f"{f_norm:.2f}")
                if f_norm > 100:
                    force_item.setForeground(QColor("#f7768e"))
                elif f_norm > 10:
                    force_item.setForeground(QColor("#e0af68"))
                self.table.setItem(i, 3, force_item)
                normal = c.frame[:3]
                self.table.setItem(i, 4, QTableWidgetItem(
                    f"({normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f})"
                ))
            except Exception:
                self.table.setItem(i, 3, QTableWidgetItem("—"))
                self.table.setItem(i, 4, QTableWidgetItem("—"))


# ═══════════════════════════════════════════════════════════════
#  Keyframe Panel
# ═══════════════════════════════════════════════════════════════

class KeyframePanel(QWidget):
    """Save and restore simulation keyframes (qpos, qvel, ctrl)."""

    load_keyframe = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model, self._data = None, None
        self._keyframes: dict[str, dict] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        save_row = QHBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Keyframe name…")
        save_row.addWidget(self.name_input)
        save_btn = QPushButton("💾 Save")
        save_btn.setProperty("class", "success")
        save_btn.clicked.connect(self._save_keyframe)
        save_row.addWidget(save_btn)
        layout.addLayout(save_row)

        self.kf_list = QListWidget()
        self.kf_list.setAlternatingRowColors(True)
        self.kf_list.itemDoubleClicked.connect(self._on_load)
        layout.addWidget(self.kf_list)

        action_row = QHBoxLayout()
        load_btn = QPushButton("📂 Load Selected")
        load_btn.clicked.connect(lambda: self._on_load(self.kf_list.currentItem()))
        action_row.addWidget(load_btn)
        del_btn = QPushButton("🗑 Delete")
        del_btn.setProperty("class", "danger")
        del_btn.clicked.connect(self._delete_selected)
        action_row.addWidget(del_btn)
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_all)
        action_row.addWidget(clear_btn)
        layout.addLayout(action_row)
        layout.addStretch()

    def build(self, model, data):
        self._model, self._data = model, data
        if self._keyframes and model is not None:
            nq = model.nq
            for name, kf in list(self._keyframes.items()):
                if len(kf.get("qpos", [])) != nq:
                    log.warning(f"Keyframe '{name}' may be incompatible with new model")

    def _save_keyframe(self):
        if self._data is None:
            return
        name = self.name_input.text().strip()
        if not name:
            name = f"KF_{len(self._keyframes) + 1}"
        if name in self._keyframes:
            reply = QMessageBox.question(
                self, "Overwrite?", f"Keyframe '{name}' exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

        self._keyframes[name] = {
            "qpos": self._data.qpos.copy(),
            "qvel": self._data.qvel.copy(),
            "ctrl": self._data.ctrl.copy(),
            "time": float(self._data.time),
        }
        self._rebuild_list()
        self.name_input.clear()
        log.info(f"Keyframe saved: {name}")

    def _on_load(self, item):
        if item is None or self._data is None:
            return
        name = item.text().split("  ")[0]
        if name not in self._keyframes:
            return
        kf = self._keyframes[name]
        try:
            if len(kf["qpos"]) == len(self._data.qpos):
                self._data.qpos[:] = kf["qpos"]
            if len(kf["qvel"]) == len(self._data.qvel):
                self._data.qvel[:] = kf["qvel"]
            if len(kf["ctrl"]) == len(self._data.ctrl):
                self._data.ctrl[:] = kf["ctrl"]
            self.load_keyframe.emit(kf)
            log.info(f"Keyframe loaded: {name}")
        except Exception as e:
            log.error(f"Error loading keyframe '{name}': {e}")

    def _delete_selected(self):
        item = self.kf_list.currentItem()
        if item is None:
            return
        name = item.text().split("  ")[0]
        if name in self._keyframes:
            del self._keyframes[name]
            self._rebuild_list()

    def _clear_all(self):
        self._keyframes.clear()
        self._rebuild_list()

    def _rebuild_list(self):
        self.kf_list.clear()
        for name, kf in self._keyframes.items():
            self.kf_list.addItem(f"{name}  (t={kf['time']:.2f}s)")