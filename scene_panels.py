"""Body tree, energy, and contacts inspector panels."""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QProgressBar, QGroupBox,
    QTreeWidget, QTreeWidgetItem, QTableWidget, QTableWidgetItem,
    QHeaderView,
)
from PySide6.QtCore import Qt, Signal
import mujoco


class BodyTreePanel(QWidget):
    focus_body = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Body", "Joints", "Geoms"])
        self.tree.setAlternatingRowColors(True)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.tree)

    def build(self, model):
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
            item = QTreeWidgetItem(
                [name, str(njnt) if njnt else "—", str(ngeom) if ngeom else "—"]
            )
            item.setData(0, Qt.UserRole, i)
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
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Geom 1", "Geom 2", "Distance"])
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