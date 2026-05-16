#!/usr/bin/env python3
"""
MuJoCo PySide6 Viewer — A professional physics simulation viewer.
Requires: pip install mujoco PySide6 numpy
"""

import sys
import time
import os
import numpy as np
from collections import deque

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QTreeWidget, QTreeWidgetItem,
    QFileDialog, QToolBar, QComboBox, QGroupBox, QCheckBox,
    QSplitter, QScrollArea, QMessageBox, QTabWidget, QSizePolicy,
    QFrame, QSpacerItem, QGridLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QPlainTextEdit, QProgressBar, QMenu, QSpinBox,
    QDialog, QDialogButtonBox, QTextEdit, QToolButton
)
from PySide6.QtCore import Qt, QTimer, QSize, Signal
from PySide6.QtGui import (
    QImage, QAction, QPainter, QFont, QKeySequence, QShortcut, QColor
)

import mujoco

# ═══════════════════════════════════════════════════════════════
# Built-in Examples
# ═══════════════════════════════════════════════════════════════
EXAMPLES = {}

EXAMPLES["Demo Scene"] = """
<mujoco model="demo_scene">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <default><joint damping="0.5"/><geom friction="0.8 0.02 0.02"/></default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.35 0.35 0.35" rgb2="0.2 0.2 0.2"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="5 5 0.1" material="grid"/>
    <body name="pendulum_base" pos="0 0 2.5">
      <joint name="pend1" type="hinge" axis="0 1 0" damping="0.2"/>
      <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.05" rgba="0.85 0.35 0.15 1" mass="0.5"/>
      <body name="pend_mid" pos="0 0 -1">
        <joint name="pend2" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.8" size="0.04" rgba="0.2 0.4 0.85 1" mass="0.3"/>
        <geom name="pend_bob" type="sphere" pos="0 0 -0.8" size="0.1" rgba="1 0.75 0.1 1" mass="1"/>
      </body>
    </body>
    <body name="arm_base" pos="2.5 0 0.3">
      <joint name="arm_yaw" type="hinge" axis="0 0 1" damping="2"/>
      <geom type="cylinder" size="0.12 0.15" rgba="0.45 0.45 0.5 1"/>
      <body name="arm_shoulder" pos="0 0 0.3">
        <joint name="arm_pitch" type="hinge" axis="0 1 0" range="-90 45" damping="1"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.7" size="0.06" rgba="0.85 0.6 0.2 1"/>
        <body name="arm_elbow" pos="0 0 0.7">
          <joint name="arm_elbow" type="hinge" axis="0 1 0" range="-135 0" damping="0.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.55" size="0.05" rgba="0.85 0.6 0.2 1"/>
          <body name="arm_wrist" pos="0 0 0.55">
            <joint name="arm_wrist" type="hinge" axis="0 0 1" damping="0.3"/>
            <geom type="sphere" size="0.06" rgba="0.3 0.3 0.35 1"/>
          </body>
        </body>
      </body>
    </body>
    <body name="cart" pos="-2.5 0 0.25">
      <joint name="cart_x" type="slide" axis="1 0 0" range="-2 2" damping="0.5"/>
      <geom type="box" size="0.25 0.15 0.1" rgba="0.7 0.7 0.25 1" mass="1"/>
      <body name="pole" pos="0 0 0.1">
        <joint name="pole_hinge" type="hinge" axis="0 1 0" damping="0.05"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.8" size="0.03" rgba="0.9 0.5 0.2 1" mass="0.3"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="cart_force" joint="cart_x" ctrlrange="-50 50" ctrllimited="true"/></actuator>
</mujoco>
"""

EXAMPLES["Cartpole"] = """
<mujoco model="cartpole">
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <worldbody>
    <light pos="0 0 5"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.3 0.3 0.3 1"/>
    <body name="cart" pos="0 0 0.2">
      <joint name="slider" type="slide" axis="1 0 0" range="-3 3"/>
      <geom type="box" size="0.2 0.15 0.1" rgba="0.8 0.8 0.2 1" mass="1"/>
      <body name="pole" pos="0 0 0.15">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01"/>
        <geom type="capsule" fromto="0 0 0 0 0 1" size="0.04" rgba="0.2 0.8 0.2 1" mass="0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="force" joint="slider" ctrlrange="-100 100" ctrllimited="true"/>
  </actuator>
</mujoco>
"""

EXAMPLES["Ant (Quadruped)"] = """
<mujoco model="ant">
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <default><geom friction="0.8 0.005 0.0001"/></default>
  <worldbody>
    <light pos="0 0 5"/><geom name="floor" type="plane" size="5 5 0.1" rgba="0.3 0.3 0.3 1"/>
    <body name="torso" pos="0 0 0.75">
      <freejoint/>
      <geom type="sphere" size="0.25" rgba="0.9 0.2 0.2 1"/>
      <body name="leg1_1" pos="0.2 0.2 0">
        <joint name="hip_1" type="hinge" axis="0 0 1" range="-40 40"/>
        <geom type="capsule" size="0.05" fromto="0 0 0 0.2 0.2 -0.2" rgba="0.2 0.2 0.9 1"/>
        <body name="leg1_2" pos="0.2 0.2 -0.2">
          <joint name="ankle_1" type="hinge" axis="0 1 0" range="-30 70"/>
          <geom type="capsule" size="0.04" fromto="0 0 0 0 0 -0.3" rgba="0.2 0.2 0.9 1"/>
        </body>
      </body>
      <body name="leg2_1" pos="-0.2 0.2 0">
        <joint name="hip_2" type="hinge" axis="0 0 1" range="-40 40"/>
        <geom type="capsule" size="0.05" fromto="0 0 0 -0.2 0.2 -0.2" rgba="0.2 0.9 0.2 1"/>
        <body name="leg2_2" pos="-0.2 0.2 -0.2">
          <joint name="ankle_2" type="hinge" axis="0 1 0" range="-30 70"/>
          <geom type="capsule" size="0.04" fromto="0 0 0 0 0 -0.3" rgba="0.2 0.9 0.2 1"/>
        </body>
      </body>
      <body name="leg3_1" pos="0.2 -0.2 0">
        <joint name="hip_3" type="hinge" axis="0 0 1" range="-40 40"/>
        <geom type="capsule" size="0.05" fromto="0 0 0 0.2 -0.2 -0.2" rgba="0.9 0.9 0.2 1"/>
        <body name="leg3_2" pos="0.2 -0.2 -0.2">
          <joint name="ankle_3" type="hinge" axis="0 1 0" range="-30 70"/>
          <geom type="capsule" size="0.04" fromto="0 0 0 0 0 -0.3" rgba="0.9 0.9 0.2 1"/>
        </body>
      </body>
      <body name="leg4_1" pos="-0.2 -0.2 0">
        <joint name="hip_4" type="hinge" axis="0 0 1" range="-40 40"/>
        <geom type="capsule" size="0.05" fromto="0 0 0 -0.2 -0.2 -0.2" rgba="0.9 0.2 0.9 1"/>
        <body name="leg4_2" pos="-0.2 -0.2 -0.2">
          <joint name="ankle_4" type="hinge" axis="0 1 0" range="-30 70"/>
          <geom type="capsule" size="0.04" fromto="0 0 0 0 0 -0.3" rgba="0.9 0.2 0.9 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_1" joint="hip_1" ctrlrange="-1 1"/>
    <motor name="ankle_1" joint="ankle_1" ctrlrange="-1 1"/>
    <motor name="hip_2" joint="hip_2" ctrlrange="-1 1"/>
    <motor name="ankle_2" joint="ankle_2" ctrlrange="-1 1"/>
    <motor name="hip_3" joint="hip_3" ctrlrange="-1 1"/>
    <motor name="ankle_3" joint="ankle_3" ctrlrange="-1 1"/>
    <motor name="hip_4" joint="hip_4" ctrlrange="-1 1"/>
    <motor name="ankle_4" joint="ankle_4" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

EXAMPLES["Robotic Gripper"] = """
<mujoco model="gripper">
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <worldbody>
    <light pos="0 0 3"/><geom name="floor" type="plane" size="2 2 0.1" rgba="0.3 0.3 0.3 1"/>
    <body name="box" pos="0 0 0.05">
      <freejoint/>
      <geom type="box" size="0.04 0.04 0.04" rgba="0.9 0.2 0.1 1" mass="0.2"/>
    </body>
    <body name="base" pos="0 0 0.4">
      <joint name="lift" type="slide" axis="0 0 1" range="0 0.35"/>
      <geom type="box" size="0.1 0.1 0.02" rgba="0.4 0.4 0.5 1" mass="2"/>
      <body name="finger_L" pos="-0.06 0 -0.02">
        <joint name="finger_L_slide" type="slide" axis="1 0 0" range="-0.04 0.04"/>
        <geom type="box" size="0.02 0.02 0.08" rgba="0.2 0.8 0.2 1"/>
      </body>
      <body name="finger_R" pos="0.06 0 -0.02">
        <joint name="finger_R_slide" type="slide" axis="1 0 0" range="-0.04 0.04"/>
        <geom type="box" size="0.02 0.02 0.08" rgba="0.2 0.2 0.8 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="lift_pos" joint="lift" kp="50" ctrlrange="0 0.35"/>
    <position name="finger_L_pos" joint="finger_L_slide" kp="20" ctrlrange="-0.04 0.04"/>
    <position name="finger_R_pos" joint="finger_R_slide" kp="20" ctrlrange="-0.04 0.04"/>
  </actuator>
</mujoco>
"""

EXAMPLES["Bouncing Balls"] = """
<mujoco model="balls">
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <default><geom friction="0.8 0.02 0.02"/></default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.2 0.2 0.2" rgb2="0.15 0.15 0.15"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 5"/>
    <geom name="floor" type="plane" size="5 5 0.1" material="grid"/>
    <body name="ball1" pos="0 0 1.5">
      <freejoint/>
      <geom type="sphere" size="0.2" mass="1" rgba="0.9 0.2 0.2 1"/>
    </body>
    <body name="ball2" pos="0.5 0 2.5">
      <freejoint/>
      <geom type="sphere" size="0.3" mass="2" rgba="0.2 0.2 0.9 1"/>
    </body>
    <body name="ball3" pos="-0.3 0 3.5">
      <freejoint/>
      <geom type="sphere" size="0.15" mass="0.5" rgba="0.2 0.9 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""

EXAMPLES["Humanoid Stick"] = """
<mujoco model="humanoid">
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <default><joint damping="2" armature="0.1"/><geom friction="0.8 0.02 0.02" type="capsule"/></default>
  <worldbody>
    <light pos="0 0 5"/><geom name="floor" type="plane" size="5 5 0.1" rgba="0.3 0.3 0.3 1"/>
    <body name="torso" pos="0 0 1.4">
      <freejoint/>
      <geom name="torso_geom" size="0.07 0.2" rgba="0.8 0.3 0.3 1"/>
      <body name="head" pos="0 0 0.28">
        <joint name="neck" type="ball" damping="3"/>
        <geom name="head_geom" type="sphere" size="0.1" rgba="0.9 0.7 0.5 1"/>
      </body>
      <body name="upper_arm_R" pos="0 -0.1 0.15">
        <joint name="shoulder_R" type="ball" damping="2"/>
        <geom size="0.035 0.16" rgba="0.3 0.3 0.8 1"/>
        <body name="lower_arm_R" pos="0 -0.08 -0.32">
          <joint name="elbow_R" type="hinge" axis="0 1 0" range="-150 0"/>
          <geom size="0.03 0.14" rgba="0.3 0.3 0.8 1"/>
        </body>
      </body>
      <body name="upper_arm_L" pos="0 0.1 0.15">
        <joint name="shoulder_L" type="ball" damping="2"/>
        <geom size="0.035 0.16" rgba="0.3 0.8 0.3 1"/>
        <body name="lower_arm_L" pos="0 0.08 -0.32">
          <joint name="elbow_L" type="hinge" axis="0 1 0" range="-150 0"/>
          <geom size="0.03 0.14" rgba="0.3 0.8 0.3 1"/>
        </body>
      </body>
      <body name="upper_leg_R" pos="0 -0.08 -0.2">
        <joint name="hip_R" type="ball" damping="3"/>
        <geom size="0.05 0.2" rgba="0.8 0.8 0.2 1"/>
        <body name="lower_leg_R" pos="0 -0.02 -0.4">
          <joint name="knee_R" type="hinge" axis="0 1 0" range="-150 0"/>
          <geom size="0.04 0.18" rgba="0.8 0.8 0.2 1"/>
        </body>
      </body>
      <body name="upper_leg_L" pos="0 0.08 -0.2">
        <joint name="hip_L" type="ball" damping="3"/>
        <geom size="0.05 0.2" rgba="0.8 0.2 0.8 1"/>
        <body name="lower_leg_L" pos="0 0.02 -0.4">
          <joint name="knee_L" type="hinge" axis="0 1 0" range="-150 0"/>
          <geom size="0.04 0.18" rgba="0.8 0.2 0.8 1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# ═══════════════════════════════════════════════════════════════
# Dark Theme
# ═══════════════════════════════════════════════════════════════
DARK_STYLE = """
QMainWindow, QWidget { background-color: #1a1b26; color: #a9b1d6; font-family: 'Segoe UI', 'Arial', sans-serif; font-size: 12px; }
QMenuBar { background-color: #1a1b26; border-bottom: 1px solid #24283b; padding: 2px; }
QMenuBar::item { padding: 4px 10px; } QMenuBar::item:selected { background-color: #24283b; border-radius: 4px; }
QMenu { background-color: #1a1b26; border: 1px solid #3d59a1; padding: 4px; } QMenu::item { padding: 5px 24px; } QMenu::item:selected { background-color: #3d59a1; border-radius: 3px; } QMenu::separator { height: 1px; background: #24283b; margin: 4px 8px; }
QToolBar { background-color: #16161e; border-bottom: 1px solid #24283b; spacing: 6px; padding: 3px 6px; }
QToolButton { background-color: #24283b; border: 1px solid #3b4261; border-radius: 4px; padding: 5px 10px; color: #a9b1d6; font-weight: 500; }
QToolButton:hover { background-color: #3d59a1; color: #c0caf5; } QToolButton:pressed { background-color: #2b3f7a; } QToolButton:checked { background-color: #3d59a1; color: #ffffff; }
QPushButton { background-color: #24283b; border: 1px solid #3b4261; border-radius: 4px; padding: 5px 14px; color: #a9b1d6; font-weight: 500; }
QPushButton:hover { background-color: #3d59a1; color: #c0caf5; border-color: #3d59a1; } QPushButton:pressed { background-color: #2b3f7a; } QPushButton:checked { background-color: #3d59a1; color: #ffffff; }
QPushButton:disabled { background-color: #1a1b26; color: #3b4261; border-color: #24283b; }
QSlider::groove:horizontal { height: 6px; background: #24283b; border-radius: 3px; } QSlider::handle:horizontal { width: 16px; height: 16px; background: #7aa2f7; border-radius: 8px; margin: -5px 0; } QSlider::sub-page:horizontal { background: #3d59a1; border-radius: 3px; }
QGroupBox { border: 1px solid #24283b; border-radius: 6px; margin-top: 12px; padding: 12px 8px 8px 8px; font-weight: bold; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #7aa2f7; }
QTreeWidget { background-color: #1a1b26; border: 1px solid #24283b; alternate-background-color: #1f2035; border-radius: 4px; } QTreeWidget::item { padding: 3px 0; } QTreeWidget::item:selected { background-color: #3d59a1; }
QHeaderView::section { background-color: #16161e; border: none; border-bottom: 1px solid #24283b; padding: 4px; font-weight: bold; color: #565f89; }
QScrollArea { border: none; background: transparent; } QTabWidget::pane { border: 1px solid #24283b; border-radius: 4px; top: -1px; }
QTabBar::tab { background-color: #16161e; border: 1px solid #24283b; padding: 7px 16px; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; }
QTabBar::tab:selected { background-color: #24283b; border-bottom-color: #24283b; color: #7aa2f7; }
QLabel { color: #a9b1d6; background: transparent; } QLabel[class="value"] { color: #7aa2f7; font-family: 'Consolas', 'Courier New', monospace; } QLabel[class="dim"] { color: #565f89; font-size: 11px; } QLabel[class="error"] { color: #f7768e; } QLabel[class="success"] { color: #9ece6a; }
QStatusBar { background-color: #16161e; border-top: 1px solid #24283b; color: #565f89; font-size: 11px; } QStatusBar QLabel { color: #565f89; }
QComboBox { background-color: #24283b; border: 1px solid #3b4261; border-radius: 4px; padding: 4px 8px; color: #a9b1d6; min-width: 80px; } QComboBox QAbstractItemView { background-color: #1a1b26; border: 1px solid #3d59a1; selection-background-color: #3d59a1; }
QCheckBox { spacing: 8px; background: transparent; } QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #3b4261; border-radius: 4px; background-color: #24283b; } QCheckBox::indicator:checked { background-color: #7aa2f7; border-color: #7aa2f7; }
QSplitter::handle { background-color: #24283b; width: 2px; }
QTableWidget { background-color: #1a1b26; border: 1px solid #24283b; alternate-background-color: #1f2035; border-radius: 4px; gridline-color: #24283b; }
QPlainTextEdit { background-color: #16161e; border: 1px solid #24283b; color: #a9b1d6; font-family: 'Consolas', 'Courier New', monospace; font-size: 12px; border-radius: 4px; padding: 6px; selection-background-color: #3d59a1; }
QProgressBar { border: 1px solid #24283b; border-radius: 4px; background-color: #16161e; text-align: center; color: #a9b1d6; } QProgressBar::chunk { background-color: #3d59a1; border-radius: 3px; }
QSpinBox { background-color: #24283b; border: 1px solid #3b4261; border-radius: 4px; padding: 3px 6px; color: #a9b1d6; }
QDialog { background-color: #1a1b26; }
"""

# ═══════════════════════════════════════════════════════════════
# Enums & Helpers
# ═══════════════════════════════════════════════════════════════
VIS_FLAGS = []
_vis_flag_names = [
    "mjVIS_JOINT", "mjVIS_ACTUATOR", "mjVIS_CONTACTPOINT",
    "mjVIS_CONTACTFORCE", "mjVIS_COM", "mjVIS_BODYFRAME",
    "mjVIS_BODYPOS", "mjVIS_WIREFRAME", "mjVIS_TRANSPARENT",
    "mjVIS_SHADOW", "mjVIS_LIGHT",
]
for name in _vis_flag_names:
    if hasattr(mujoco.mjtVisFlag, name):
        flag = getattr(mujoco.mjtVisFlag, name)
        VIS_FLAGS.append((name[6:].replace('_', ' ').title(), flag))

JOINT_TYPE_NAMES = {}
for name in ["mjJNT_FREE", "mjJNT_BALL", "mjJNT_SLIDE", "mjJNT_HINGE"]:
    if hasattr(mujoco.mjtJoint, name):
        JOINT_TYPE_NAMES[getattr(mujoco.mjtJoint, name)] = name[6:].title()

QPOS_DIMS = {}
for jtype, jname in [
    (mujoco.mjtJoint.mjJNT_FREE, 7),
    (mujoco.mjtJoint.mjJNT_BALL, 4),
    (mujoco.mjtJoint.mjJNT_SLIDE, 1),
    (mujoco.mjtJoint.mjJNT_HINGE, 1),
]:
    QPOS_DIMS[jtype] = jname


# ═══════════════════════════════════════════════════════════════
# Toast Notification
# ═══════════════════════════════════════════════════════════════
class ToastLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self._default_style = (
            "background-color: rgba(61,89,161,210); color:#fff; "
            "border-radius:8px; padding:8px 20px; font-weight:bold; font-size:13px;"
        )
        self._error_style = (
            "background-color: rgba(247,118,142,210); color:#fff; "
            "border-radius:8px; padding:8px 20px; font-weight:bold; font-size:13px;"
        )
        self._success_style = (
            "background-color: rgba(158,206,106,200); color:#1a1b26; "
            "border-radius:8px; padding:8px 20px; font-weight:bold; font-size:13px;"
        )
        self.setStyleSheet(self._default_style)
        self.hide()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_message(self, text, duration=2500):
        self.setStyleSheet(self._default_style)
        self._show(text, duration)

    def show_error(self, text, duration=4000):
        self.setStyleSheet(self._error_style)
        self._show(text, duration)

    def show_success(self, text, duration=2500):
        self.setStyleSheet(self._success_style)
        self._show(text, duration)

    def _show(self, text, duration):
        self.setText(text)
        self.adjustSize()
        if self.parentWidget():
            pw = self.parentWidget().width()
            self.move((pw - self.width()) // 2, 16)
        self.show()
        self.raise_()
        self._timer.start(duration)


# ═══════════════════════════════════════════════════════════════
# Shortcuts Help Dialog
# ═══════════════════════════════════════════════════════════════
class ShortcutsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        info = QTextEdit()
        info.setReadOnly(True)
        info.setHtml("""
        <style>
            table { border-collapse: collapse; width: 100%; }
            td { padding: 5px 10px; border-bottom: 1px solid #24283b; }
            .key { color: #7aa2f7; font-family: Consolas, monospace; font-weight: bold; }
            .desc { color: #a9b1d6; }
        </style>
        <table>
        <tr><td class="key">Space</td><td class="desc">Play / Pause</td></tr>
        <tr><td class="key">→</td><td class="desc">Step simulation forward</td></tr>
        <tr><td class="key">R</td><td class="desc">Reset simulation</td></tr>
        <tr><td class="key">F</td><td class="desc">Fit camera to scene</td></tr>
        <tr><td class="key">Ctrl+O</td><td class="desc">Open model file</td></tr>
        <tr><td class="key">Ctrl+S</td><td class="desc">Save screenshot</td></tr>
        <tr><td class="key">Ctrl+Click</td><td class="desc">Select & drag body</td></tr>
        <tr><td class="key">Left Drag</td><td class="desc">Rotate camera</td></tr>
        <tr><td class="key">Middle Drag</td><td class="desc">Pan camera</td></tr>
        <tr><td class="key">Right Drag / Scroll</td><td class="desc">Zoom</td></tr>
        <tr><td class="key">T</td><td class="desc">Toggle body traces</td></tr>
        <tr><td class="key">C</td><td class="desc">Clear traces</td></tr>
        <tr><td class="key">1-8</td><td class="desc">Set speed (0.1x–10x)</td></tr>
        </table>
        """)
        layout.addWidget(info)
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)


# ═══════════════════════════════════════════════════════════════
# Viewport — with Perturbation, Traces, Overlay, Drag-Drop
# ═══════════════════════════════════════════════════════════════
class MujocoViewport(QWidget):
    body_focused = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAcceptDrops(True)

        self.model, self.data = None, None
        self.renderer = None
        self.cam = mujoco.MjvCamera()
        self.vopt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()

        self._image = None
        self._render_w, self._render_h = 0, 0
        self._last_pos = None
        self._active_btn = None
        self._ctrl_held = False
        self._perturbing = False
        self._selected_body = -1

        self._paused = False

        # Traces
        self._show_traces = False
        self._traces = {}
        self._trace_interval = 4
        self._trace_counter = 0
        self._max_trace_len = 400

        # Toast
        self._toast = ToastLabel(self)

    # ── Model & Rendering ──────────────────────────────────────
    def set_model(self, model, data):
        self.model, self.data = model, data
        self.pert = mujoco.MjvPerturb()
        self._traces = {}
        self._trace_counter = 0
        self._selected_body = -1
        mujoco.mj_forward(model, data)
        self._init_renderer()
        self.reset_camera()
        self.render()

    def _init_renderer(self):
        w, h = max(self.width(), 64), max(self.height(), 64)
        self._render_w, self._render_h = w, h
        try:
            self.renderer = mujoco.Renderer(self.model, height=h, width=w)
        except Exception as e:
            QMessageBox.critical(self, "Renderer Error", f"Failed:\n{e}")
            self.renderer = None

    def reset_camera(self):
        if self.model is None:
            return
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.cam)

    def focus_on_body(self, body_id):
        if self.model is None or body_id < 0 or body_id >= self.model.nbody:
            return
        pos = self.data.xipos[body_id].copy()
        self.cam.lookat = pos
        self.cam.distance = max(2.0, self.model.stat.extent * 0.8)
        self.cam.azimuth = 135.0
        self.cam.elevation = -20.0
        self.render()

    def render(self):
        if self.renderer is None or self.model is None:
            return
        try:
            mujoco.mjv_updateScene(
                self.model, self.data, self.vopt,
                self.pert, self.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.renderer.scene,
            )
            if self._show_traces:
                self._add_trace_geoms()
            pixels = np.ascontiguousarray(self.renderer.render())
            h, w, ch = pixels.shape
            self._image = QImage(pixels.data, w, h, ch * w, QImage.Format_RGB888).copy()
        except Exception as e:
            print(f"Render error: {e}")
            return
        self.update()

    # ── Traces ─────────────────────────────────────────────────
    def _add_trace_geoms(self):
        scene = self.renderer.scene
        for body_id, positions in self._traces.items():
            if body_id <= 0 or len(positions) < 2:
                continue
            n = len(positions)
            for idx, pos in enumerate(positions):
                if scene.ngeom >= scene.maxgeom:
                    return
                alpha = 0.15 + 0.5 * (idx / max(n - 1, 1))
                rgba = np.array([0.9, 0.6, 0.1, alpha], dtype=np.float32)
                size_arr = np.array([0.008, 0, 0], dtype=np.float64)
                pos_arr = np.array(pos, dtype=np.float64)
                mat_arr = np.eye(3, dtype=np.float64).flatten()
                try:
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        size_arr, pos_arr, mat_arr, rgba,
                    )
                    scene.ngeom += 1
                except Exception:
                    return

    def record_trace(self):
        if not self._show_traces or self.model is None:
            return
        self._trace_counter += 1
        if self._trace_counter % self._trace_interval != 0:
            return
        for i in range(1, self.model.nbody):
            pos = self.data.xipos[i].copy()
            if i not in self._traces:
                self._traces[i] = deque(maxlen=self._max_trace_len)
            self._traces[i].append(pos)

    def clear_traces(self):
        self._traces = {}
        self._trace_counter = 0
        self._toast.show_message("Traces cleared")

    def toggle_traces(self):
        self._show_traces = not self._show_traces
        if not self._show_traces:
            self._traces = {}
            self._trace_counter = 0
        self._toast.show_message(
            f"Traces {'ON' if self._show_traces else 'OFF'}"
        )

    # ── Body Selection ─────────────────────────────────────────
    def _try_select(self, pos):
        if self.renderer is None or self.model is None:
            return
        rect = self._get_render_rect()
        if rect is None:
            return
        dx, dy, dw, dh = rect
        relx = (pos.x() - dx) / max(dw, 1)
        rely = (pos.y() - dy) / max(dh, 1)
        aspectratio = dw / max(dh, 1)
        if relx < 0 or relx > 1 or rely < 0 or rely > 1:
            return
        try:
            result = mujoco.mjv_select(
                self.model, self.data, self.vopt,
                aspectratio, relx, rely,
                self.renderer.scene,
            )
            body_id = -1
            selpnt = np.zeros(3)
            if isinstance(result, (tuple, list)):
                if len(result) >= 5:
                    body_id = int(result[4])
                    selpnt = np.asarray(result[0])
                elif len(result) >= 2:
                    body_id = int(result[-1])
                    selpnt = np.asarray(result[0])
            elif isinstance(result, (int, np.integer)):
                body_id = int(result)
            if body_id > 0:
                self.pert.select = body_id
                self._selected_body = body_id
                try:
                    self.pert.selectpos[:] = selpnt[:3]
                except Exception:
                    pass
                self.body_focused.emit(body_id)
                name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                ) or f"body_{body_id}"
                self._toast.show_message(f"Selected: {name}")
            else:
                self.pert.select = 0
                self._selected_body = -1
                self._toast.show_message("Selection cleared")
        except Exception as e:
            print(f"Selection error: {e}")

    # ── Painting ───────────────────────────────────────────────
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0f0f14"))
        if self._image is not None:
            scaled = self._image.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawImage(x, y, scaled)

            if self._paused:
                painter.fillRect(self.rect(), QColor(0, 0, 0, 60))
                painter.setPen(QColor(255, 255, 255, 180))
                painter.setFont(QFont("Segoe UI", 28, QFont.Bold))
                painter.drawText(self.rect(), Qt.AlignCenter, "⏸  PAUSED")
                painter.setPen(QColor(255, 255, 255, 100))
                painter.setFont(QFont("Segoe UI", 12))
                r = self.rect()
                r.setTop(r.top() + 30)
                painter.drawText(r, Qt.AlignCenter, "Press Space to resume")

            if self._selected_body > 0 and self.model is not None:
                name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, self._selected_body
                ) or f"body_{self._selected_body}"
                painter.setPen(QColor(122, 162, 247, 220))
                painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
                painter.drawText(12, 24, f"🔵 {name}")
                if self._perturbing:
                    painter.setPen(QColor(158, 206, 106, 200))
                    painter.setFont(QFont("Segoe UI", 10))
                    painter.drawText(12, 42, "Dragging…")
        else:
            painter.setPen(QColor("#565f89"))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(
                self.rect(), Qt.AlignCenter,
                "Load a model to begin\nDrag & drop an XML file here"
            )
        painter.end()

    def _get_render_rect(self):
        if self._image is None:
            return None
        iw, ih = self._render_w, self._render_h
        ww, wh = self.width(), self.height()
        scale = min(ww / max(iw, 1), wh / max(ih, 1))
        dw, dh = int(iw * scale), int(ih * scale)
        dx, dy = (ww - dw) // 2, (wh - dh) // 2
        return dx, dy, dw, dh

    # ── Resize ─────────────────────────────────────────────────
    def resizeEvent(self, event):
        w, h = self.width(), self.height()
        if (
            self.model
            and self.renderer
            and (abs(w - self._render_w) > 8 or abs(h - self._render_h) > 8)
        ):
            self._render_w, self._render_h = max(w, 64), max(h, 64)
            try:
                self.renderer = mujoco.Renderer(
                    self.model, height=self._render_h, width=self._render_w
                )
            except Exception:
                pass
        super().resizeEvent(event)

    # ── Keyboard ───────────────────────────────────────────────
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            self._ctrl_held = True
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self._ctrl_held = False
        super().keyReleaseEvent(event)

    # ── Mouse / Perturbation ───────────────────────────────────
    def mousePressEvent(self, event):
        self._last_pos = event.position()
        self._active_btn = event.button()

        if event.button() == Qt.LeftButton and (
            self._ctrl_held or event.modifiers() & Qt.ControlModifier
        ):
            self._try_select(event.position())
            self._perturbing = True
            return

    def mouseReleaseEvent(self, event):
        self._last_pos = None
        self._active_btn = None
        if self._perturbing:
            self._perturbing = False
            if self.model is not None:
                self.data.xfrc_applied[:] = 0
            self.render()

    def mouseMoveEvent(self, event):
        if self._last_pos is None or self.model is None:
            return
        pos = event.position()
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()

        # Perturbation: Ctrl + drag
        if self._perturbing and self._selected_body > 0 and self.renderer is not None:
            try:
                mujoco.mjv_movePerturb(
                    self.model, self.data, self.vopt,
                    mujoco.mjtMouse.mjMOUSE_MOVE_V,
                    dx, dy,
                    self.renderer.scene, self.pert,
                )
                if self.pert.select > 0:
                    body_id = self.pert.select
                    force = self.pert.force.copy()
                    torque = self.pert.torque.copy()
                    self.data.xfrc_applied[body_id, :3] = force
                    self.data.xfrc_applied[body_id, 3:] = torque
            except Exception:
                # Fallback: simple screen-space force
                if self._selected_body > 0 and self._selected_body < self.model.nbody:
                    scale = 50.0
                    self.data.xfrc_applied[self._selected_body, 0] = dx * scale
                    self.data.xfrc_applied[self._selected_body, 1] = -dy * scale
                    self.data.xfrc_applied[self._selected_body, 2] = 0
            self._last_pos = pos
            self.render()
            return

        # Camera control
        if self._active_btn == Qt.LeftButton:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif self._active_btn == Qt.MiddleButton:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._active_btn == Qt.RightButton:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            return
        if self.renderer is not None:
            try:
                self.renderer.update_scene(
                    self.data, camera=self.cam, scene_option=self.vopt
                )
                mujoco.mjv_moveCamera(
                    self.model, action, dx, dy, self.renderer.scene, self.cam
                )
            except Exception:
                pass
        self._last_pos = pos
        self.render()

    def wheelEvent(self, event):
        if self.model is None or self.renderer is None:
            return
        delta = event.angleDelta().y() / 120.0
        try:
            self.renderer.update_scene(
                self.data, camera=self.cam, scene_option=self.vopt
            )
            mujoco.mjv_moveCamera(
                self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                0, -delta * 30, self.renderer.scene, self.cam,
            )
        except Exception:
            pass
        self.render()

    # ── Drag & Drop ────────────────────────────────────────────
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith('.xml'):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith('.xml'):
                mw = self.window()
                if hasattr(mw, '_load_model_path'):
                    mw._load_model_path(path)
                break


# ═══════════════════════════════════════════════════════════════
# Joint Panel — fixed unpacking, ball/free joint support
# ═══════════════════════════════════════════════════════════════
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
            name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                or f"joint_{i}"
            )
            jtype = model.jnt_type[i]
            jtype_name = JOINT_TYPE_NAMES.get(jtype, "?")
            qpos_adr = model.jnt_qposadr[i]
            ndim = QPOS_DIMS.get(jtype, 1)

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
                # Hinge or Slide — scalar, 1 DoF
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
                    # No range defined — use a wide default and display only
                    slider.setEnabled(False)
                gl.addWidget(slider)
                self._entries.append(("scalar", i, val_lbl, slider, qpos_adr, has_range))

            group.setMaximumHeight(120)
            self._layout.addWidget(group)
        self._layout.addStretch()

    def _make_slider_cb(self, jnt_id, qpos_adr, lo, hi):
        def cb(val):
            self._data.qpos[qpos_adr] = lo + (val / 1000.0) * (hi - lo)

        return cb

    def refresh(self):
        if self._data is None or self._model is None:
            return
        for entry in self._entries:
            kind = entry[0]
            jnt_id = entry[1]
            lbl = entry[2]
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
                lbl.setText(
                    f"quat: ({q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f})"
                )
            elif kind == "scalar":
                v = self._data.qpos[qpos_adr]
                lbl.setText(f"{v:.4f}")
                slider = entry[3]
                has_range = entry[5]
                if has_range and not slider.isSliderDown():
                    rng = self._model.jnt_range[jnt_id]
                    lo, hi = rng[0], rng[1]
                    slider.blockSignals(True)
                    slider.setValue(
                        int(np.clip((v - lo) / (hi - lo), 0, 1) * 1000)
                    )
                    slider.blockSignals(False)


# ═══════════════════════════════════════════════════════════════
# Actuator Panel
# ═══════════════════════════════════════════════════════════════
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

            # Reset button
            reset_btn = QPushButton("⟲ Reset")
            reset_btn.setFixedHeight(22)
            reset_btn.clicked.connect(
                self._make_reset_cb(i, lo, hi, slider, val_lbl)
            )
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


# ═══════════════════════════════════════════════════════════════
# Body Tree Panel — with double-click focus
# ═══════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════
# Energy Panel
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


# ═══════════════════════════════════════════════════════════════
# Contacts Panel — fixed c.elem bug
# ═══════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════
# XML Editor — with validation indicator
# ═══════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════
# Render Options Panel — with trace toggle
# ═══════════════════════════════════════════════════════════════
class RenderOptionsPanel(QWidget):
    def __init__(self, viewport: MujocoViewport, parent=None):
        super().__init__(parent)
        self.viewport = viewport
        self._checkboxes = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Camera
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

        # Traces
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

        # Visualization
        vis_group = QGroupBox("Visualization")
        vis_lay = QVBoxLayout(vis_group)
        for name, flag in VIS_FLAGS:
            cb = QCheckBox(name)
            cb.setChecked(self.viewport.vopt.flags[flag])
            cb.toggled.connect(self._make_flag_cb(flag))
            vis_lay.addWidget(cb)
            self._checkboxes.append(cb)
        layout.addWidget(vis_group)

        # Geometry Groups
        geom_group = QGroupBox("Geometry Groups")
        geom_lay = QGridLayout(geom_group)
        for g in range(6):
            cb = QCheckBox(f"Group {g}")
            cb.setChecked(True)
            cb.toggled.connect(self._make_geom_cb(g))
            geom_lay.addWidget(cb, g // 3, g % 3)
        layout.addWidget(geom_group)

        # Render color scheme
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
            elif idx == 1:
                # Try setting white background via lighthead
                pass
            elif idx == 2:
                pass
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


# ═══════════════════════════════════════════════════════════════
# Main Window
# ═══════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MuJoCo Viewer")
        self.resize(1280, 800)
        self.model, self.data = None, None
        self.playing = False
        self.speed_factor = 1.0
        self._sim_time_accumulator = 0.0
        self._current_xml = EXAMPLES["Demo Scene"]
        self._current_file_path = ""

        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        self.last_real_time = time.time()
        self.last_sim_time = 0.0
        self.rtf = 0.0

        self._build_ui()
        self._build_menu()
        self._build_toolbar()
        self._build_shortcuts()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)

        self._load_xml_string(self._current_xml)

    # ── UI Construction ────────────────────────────────────────
    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        self.viewport = MujocoViewport()
        splitter.addWidget(self.viewport)

        right = QWidget()
        right.setMinimumWidth(260)
        right.setMaximumWidth(440)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.joint_panel = JointPanel()
        jscroll = QScrollArea()
        jscroll.setWidgetResizable(True)
        jscroll.setWidget(self.joint_panel)
        self.tabs.addTab(jscroll, "Joints")

        self.actuator_panel = ActuatorPanel()
        ascroll = QScrollArea()
        ascroll.setWidgetResizable(True)
        ascroll.setWidget(self.actuator_panel)
        self.tabs.addTab(ascroll, "Actuators")

        self.body_panel = BodyTreePanel()
        self.body_panel.focus_body.connect(self._focus_body)
        self.tabs.addTab(self.body_panel, "Bodies")

        self.energy_panel = EnergyPanel()
        escroll = QScrollArea()
        escroll.setWidgetResizable(True)
        escroll.setWidget(self.energy_panel)
        self.tabs.addTab(escroll, "Energy")

        self.contacts_panel = ContactsPanel()
        self.tabs.addTab(self.contacts_panel, "Contacts")

        self.xml_editor = XMLEditor()
        self.xml_editor.apply_requested.connect(self._apply_xml)
        self.tabs.addTab(self.xml_editor, "XML Editor")

        self.options_panel = RenderOptionsPanel(self.viewport)
        oscroll = QScrollArea()
        oscroll.setWidgetResizable(True)
        oscroll.setWidget(self.options_panel)
        self.tabs.addTab(oscroll, "Options")

        right_lay.addWidget(self.tabs)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([900, 320])
        self.setCentralWidget(splitter)

        # Status bar
        self.status_time = QLabel("Time: 0.000s")
        self.status_step = QLabel("Step: 0")
        self.status_rtf = QLabel("RTF: —")
        self.status_fps = QLabel("FPS: —")
        self.status_info = QLabel("")
        self.statusBar().addPermanentWidget(self.status_time)
        self.statusBar().addPermanentWidget(self.status_step)
        self.statusBar().addPermanentWidget(self.status_rtf)
        self.statusBar().addPermanentWidget(self.status_fps)
        self.statusBar().addPermanentWidget(self.status_info)

        # Connect viewport body_focused
        self.viewport.body_focused.connect(self._focus_body)

    def _build_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        load_act = QAction("&Load Model…", self)
        load_act.setShortcut(QKeySequence.Open)
        load_act.triggered.connect(self._load_model_file)
        file_menu.addAction(load_act)
        file_menu.addSeparator()
        save_state_act = QAction("Save &State…", self)
        save_state_act.triggered.connect(self._save_state)
        file_menu.addAction(save_state_act)
        load_state_act = QAction("Load S&tate…", self)
        load_state_act.triggered.connect(self._load_state)
        file_menu.addAction(load_state_act)
        file_menu.addSeparator()
        screenshot_act = QAction("Save &Screenshot…", self)
        screenshot_act.setShortcut(QKeySequence("Ctrl+S"))
        screenshot_act.triggered.connect(self._screenshot)
        file_menu.addAction(screenshot_act)
        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        sim_menu = menubar.addMenu("&Simulation")
        self.play_act = QAction("▶  &Play", self)
        self.play_act.triggered.connect(self._toggle_play)
        sim_menu.addAction(self.play_act)
        step_act = QAction("⏭  S&tep", self)
        step_act.triggered.connect(self._step_once)
        sim_menu.addAction(step_act)
        reset_act = QAction("⏮  &Reset", self)
        reset_act.triggered.connect(self._reset_sim)
        sim_menu.addAction(reset_act)

        help_menu = menubar.addMenu("&Help")
        shortcuts_act = QAction("&Keyboard Shortcuts", self)
        shortcuts_act.triggered.connect(self._show_shortcuts)
        help_menu.addAction(shortcuts_act)
        about_act = QAction("&About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _build_toolbar(self):
        tb = QToolBar("Controls")
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        self.addToolBar(tb)

        self.btn_play = QPushButton("▶  Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setMinimumWidth(90)
        self.btn_play.clicked.connect(self._toggle_play)
        tb.addWidget(self.btn_play)

        self.btn_step = QPushButton("⏭  Step")
        self.btn_step.clicked.connect(self._step_once)
        tb.addWidget(self.btn_step)

        self.btn_reset = QPushButton("⏮  Reset")
        self.btn_reset.clicked.connect(self._reset_sim)
        tb.addWidget(self.btn_reset)

        tb.addSeparator()

        tb.addWidget(QLabel("  Speed: "))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(
            ["0.1x", "0.25x", "0.5x", "1x", "2x", "4x", "8x", "10x"]
        )
        self.speed_combo.setCurrentIndex(3)
        self.speed_combo.currentIndexChanged.connect(self._speed_changed)
        tb.addWidget(self.speed_combo)

        tb.addSeparator()

        tb.addWidget(QLabel("  Step×: "))
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 500)
        self.step_spin.setValue(1)
        self.step_spin.setToolTip("Number of physics steps per Step click")
        tb.addWidget(self.step_spin)

        tb.addSeparator()

        tb.addWidget(QLabel("  Example: "))
        self.example_combo = QComboBox()
        self.example_combo.addItems(EXAMPLES.keys())
        self.example_combo.currentTextChanged.connect(self._load_example)
        tb.addWidget(self.example_combo)

        tb.addSeparator()

        self.btn_load = QPushButton("📂  Load File")
        self.btn_load.clicked.connect(self._load_model_file)
        tb.addWidget(self.btn_load)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)

        self.btn_trace = QPushButton("◉  Traces")
        self.btn_trace.setCheckable(True)
        self.btn_trace.setToolTip("Toggle body traces (T)")
        self.btn_trace.clicked.connect(self._toggle_traces)
        tb.addWidget(self.btn_trace)

        self.btn_screenshot = QPushButton("📸  Screenshot")
        self.btn_screenshot.clicked.connect(self._screenshot)
        tb.addWidget(self.btn_screenshot)

        self.btn_fit = QPushButton("🎯  Fit Camera")
        self.btn_fit.clicked.connect(self._fit_camera)
        tb.addWidget(self.btn_fit)

    def _build_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self, self._toggle_play)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._step_once)
        QShortcut(QKeySequence(Qt.Key_R), self, self._reset_sim)
        QShortcut(QKeySequence(Qt.Key_F), self, self._fit_camera)
        QShortcut(QKeySequence(Qt.Key_T), self, self._toggle_traces_key)
        QShortcut(QKeySequence(Qt.Key_C), self, self._clear_traces)
        QShortcut(QKeySequence("1"), self, lambda: self._set_speed_index(0))
        QShortcut(QKeySequence("2"), self, lambda: self._set_speed_index(1))
        QShortcut(QKeySequence("3"), self, lambda: self._set_speed_index(2))
        QShortcut(QKeySequence("4"), self, lambda: self._set_speed_index(3))
        QShortcut(QKeySequence("5"), self, lambda: self._set_speed_index(4))
        QShortcut(QKeySequence("6"), self, lambda: self._set_speed_index(5))
        QShortcut(QKeySequence("7"), self, lambda: self._set_speed_index(6))
        QShortcut(QKeySequence("8"), self, lambda: self._set_speed_index(7))

    # ── Model Loading ──────────────────────────────────────────
    def _load_example(self, name):
        if name in EXAMPLES:
            self._load_xml_string(EXAMPLES[name])

    def _load_xml_string(self, xml_string: str):
        try:
            model = mujoco.MjModel.from_xml_string(xml_string)
            data = mujoco.MjData(model)
        except Exception as e:
            QMessageBox.critical(self, "Model Error", f"Failed to parse XML:\n{e}")
            self.xml_editor.set_validation(False, str(e)[:80])
            return
        self._current_xml = xml_string
        self._current_file_path = ""
        self.xml_editor.set_xml(xml_string)
        self.xml_editor.set_validation(True, "Loaded")
        self._set_model(model, data)

    def _load_model_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load MuJoCo Model", "", "MuJoCo XML (*.xml);;All Files (*)"
        )
        if not path:
            return
        self._load_model_path(path)

    def _load_model_path(self, path: str):
        if not os.path.isfile(path):
            QMessageBox.critical(self, "Load Error", f"File not found: {path}")
            return
        try:
            # Use from_xml_path for proper asset resolution
            model = mujoco.MjModel.from_xml_path(path)
            data = mujoco.MjData(model)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed:\n{e}")
            return
        # Read XML for editor
        try:
            with open(path, "r", encoding="utf-8") as f:
                xml_string = f.read()
        except Exception:
            xml_string = f"<!-- Loaded from {path} -->"
        self._current_xml = xml_string
        self._current_file_path = path
        self.xml_editor.set_xml(xml_string)
        self.xml_editor.set_validation(True, f"Loaded from {os.path.basename(path)}")
        self._set_model(model, data)
        self.viewport._toast.show_success(f"Loaded: {os.path.basename(path)}")

    def _apply_xml(self):
        xml_string = self.xml_editor.editor.toPlainText()
        try:
            model = mujoco.MjModel.from_xml_string(xml_string)
            data = mujoco.MjData(model)
        except Exception as e:
            self.xml_editor.set_validation(False, str(e)[:100])
            self.viewport._toast.show_error("XML parse error")
            return
        self._current_xml = xml_string
        self.xml_editor.set_xml(xml_string)
        self.xml_editor.set_validation(True, "Applied")
        self._set_model(model, data)
        self.viewport._toast.show_success("XML applied successfully")

    def _set_model(self, model, data):
        self.model, self.data = model, data
        mujoco.mj_forward(model, data)
        self.playing = False
        self.btn_play.setChecked(False)
        self.btn_play.setText("▶  Play")
        self._sim_time_accumulator = 0.0

        self.viewport.set_model(model, data)
        self.joint_panel.build(model, data)
        self.actuator_panel.build(model, data)
        self.body_panel.build(model)
        self.options_panel.populate_cameras(model)

        self.status_info.setText(
            f"Bodies: {model.nbody} | Joints: {model.njnt} | "
            f"Actuators: {model.nu} | DoFs: {model.nv}"
        )
        self.last_sim_time = data.time
        self.last_real_time = time.time()

    # ── Simulation Controls ────────────────────────────────────
    def _toggle_play(self):
        if not self.model:
            return
        self.playing = not self.playing
        self.btn_play.setChecked(self.playing)
        self.btn_play.setText("⏸  Pause" if self.playing else "▶  Play")
        self.viewport._paused = not self.playing
        if self.playing:
            self.last_real_time = time.time()
            self.last_sim_time = self.data.time
            self._sim_time_accumulator = 0.0

    def _step_once(self):
        if self.model and self.data:
            steps = self.step_spin.value()
            for _ in range(steps):
                mujoco.mj_step(self.model, self.data)
            self._update_ui()

    def _reset_sim(self):
        if self.model and self.data:
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.playing = False
            self.btn_play.setChecked(False)
            self.btn_play.setText("▶  Play")
            self.viewport._paused = False
            self._sim_time_accumulator = 0.0
            self.last_sim_time = self.data.time
            self.last_real_time = time.time()
            self.viewport.clear_traces()
            self._update_ui()
            self.viewport._toast.show_message("Simulation reset")

    def _speed_changed(self, index):
        speeds = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0]
        if 0 <= index < len(speeds):
            self.speed_factor = speeds[index]

    def _set_speed_index(self, idx):
        self.speed_combo.setCurrentIndex(idx)

    def _fit_camera(self):
        self.viewport.reset_camera()
        self.options_panel.cam_combo.setCurrentIndex(0)
        self.viewport.render()

    def _focus_body(self, body_id):
        self.viewport.focus_on_body(body_id)

    # ── Traces ─────────────────────────────────────────────────
    def _toggle_traces(self):
        self.viewport.toggle_traces()
        self.btn_trace.setChecked(self.viewport._show_traces)
        self.options_panel.trace_cb.setChecked(self.viewport._show_traces)

    def _toggle_traces_key(self):
        self._toggle_traces()

    def _clear_traces(self):
        self.viewport.clear_traces()

    # ── State Save/Load ────────────────────────────────────────
    def _save_state(self):
        if self.model is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save State", "state.npz", "NumPy Archive (*.npz)"
        )
        if not path:
            return
        try:
            np.savez(
                path,
                qpos=self.data.qpos.copy(),
                qvel=self.data.qvel.copy(),
                ctrl=self.data.ctrl.copy(),
                act=self.data.act.copy(),
                time=np.array([self.data.time]),
            )
            self.viewport._toast.show_success(f"State saved: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _load_state(self):
        if self.model is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load State", "", "NumPy Archive (*.npz);;All Files (*)"
        )
        if not path:
            return
        try:
            state = np.load(path)
            if len(state["qpos"]) == len(self.data.qpos):
                self.data.qpos[:] = state["qpos"]
            if len(state["qvel"]) == len(self.data.qvel):
                self.data.qvel[:] = state["qvel"]
            if "ctrl" in state and len(state["ctrl"]) == len(self.data.ctrl):
                self.data.ctrl[:] = state["ctrl"]
            if "act" in state and len(state["act"]) == len(self.data.act):
                self.data.act[:] = state["act"]
            mujoco.mj_forward(self.model, self.data)
            self.actuator_panel.build(self.model, self.data)
            self._update_ui()
            self.viewport._toast.show_success(
                f"State loaded: {os.path.basename(path)}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Load State Error", str(e))

    # ── Screenshot ─────────────────────────────────────────────
    def _screenshot(self):
        if self.viewport._image is None:
            return
        default_name = f"mujoco_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", default_name, "PNG Image (*.png)"
        )
        if path:
            if self.viewport._image.save(path):
                self.viewport._toast.show_success(
                    f"Screenshot saved: {os.path.basename(path)}"
                )
            else:
                self.viewport._toast.show_error("Failed to save screenshot")

    # ── Help ───────────────────────────────────────────────────
    def _show_shortcuts(self):
        dlg = ShortcutsDialog(self)
        dlg.exec()

    def _show_about(self):
        QMessageBox.about(
            self,
            "About MuJoCo Viewer",
            f"<h3>MuJoCo Viewer</h3>"
            f"<p>A professional physics simulation viewer built with "
            f"PySide6 and MuJoCo.</p>"
            f"<p>MuJoCo version: {mujoco.__version__}</p>"
            f"<p><b>Controls:</b> Left-drag = rotate, "
            f"Middle-drag = pan, Right-drag/scroll = zoom, "
            f"Ctrl+Click = select & drag body</p>",
        )

    # ── Main Loop ──────────────────────────────────────────────
    def _tick(self):
        if self.model and self.data:
            if self.playing:
                now = time.time()
                real_dt = min(now - self.last_real_time, 0.1)
                self.last_real_time = now
                self._sim_time_accumulator += real_dt * self.speed_factor

                max_steps = 50
                step_count = 0
                timestep = self.model.opt.timestep
                while (
                    self._sim_time_accumulator >= timestep
                    and step_count < max_steps
                ):
                    mujoco.mj_step(self.model, self.data)
                    self._sim_time_accumulator -= timestep
                    step_count += 1

                # Prevent time accumulator spiral
                if step_count >= max_steps:
                    self._sim_time_accumulator = 0.0

                self.viewport.record_trace()

            self._update_ui()
            self._calc_fps()

    def _update_ui(self):
        self.viewport.render()
        self.joint_panel.refresh()
        self.energy_panel.refresh(self.model, self.data)
        self.contacts_panel.refresh(self.model, self.data)
        self.status_time.setText(f"Time: {self.data.time:.3f}s")

    def _calc_fps(self):
        self.frame_count += 1
        now = time.time()

        # RTF
        sim_dt = self.data.time - self.last_sim_time
        real_dt = now - self.last_real_time
        if real_dt > 0.2:
            self.rtf = sim_dt / real_dt if real_dt > 0 else 0
            self.last_sim_time = self.data.time
            self.last_real_time = now
            self.status_rtf.setText(f"RTF: {self.rtf:.2f}x")

        # FPS
        elapsed = now - self.last_fps_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_time = now
            self.status_fps.setText(f"FPS: {self.fps:.0f}")


# ═══════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())