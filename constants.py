"""Shared constants, enums, the dark theme stylesheet, and built-in example scenes."""

import mujoco

# ── Visualization flags ────────────────────────────────────────
VIS_FLAGS = []
_vis_flag_names = [
    "mjVIS_JOINT", "mjVIS_ACTUATOR", "mjVIS_CONTACTPOINT",
    "mjVIS_CONTACTFORCE", "mjVIS_COM", "mjVIS_BODYFRAME",
    "mjVIS_BODYPOS", "mjVIS_WIREFRAME", "mjVIS_TRANSPARENT",
    "mjVIS_SHADOW", "mjVIS_LIGHT", "mjVIS_TEXTURE",
    "mjVIS_SPLINE", "mjVIS_INERTIA",
]
for name in _vis_flag_names:
    if hasattr(mujoco.mjtVisFlag, name):
        flag = getattr(mujoco.mjtVisFlag, name)
        VIS_FLAGS.append((name[6:].replace('_', ' ').title(), flag))

# ── Joint type display names ───────────────────────────────────
JOINT_TYPE_NAMES = {}
for name in ["mjJNT_FREE", "mjJNT_BALL", "mjJNT_SLIDE", "mjJNT_HINGE"]:
    if hasattr(mujoco.mjtJoint, name):
        JOINT_TYPE_NAMES[getattr(mujoco.mjtJoint, name)] = name[6:].title()

# ── qpos dimension per joint type ─────────────────────────────
QPOS_DIMS = {
    mujoco.mjtJoint.mjJNT_FREE: 7,
    mujoco.mjtJoint.mjJNT_BALL: 4,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
}

# ── Camera presets ─────────────────────────────────────────────
CAMERA_PRESETS = [
    ("Front",    0.0,   0.0),
    ("Back",   180.0,   0.0),
    ("Left",    90.0,   0.0),
    ("Right", -90.0,   0.0),
    ("Top",      0.0, -89.0),
    ("Bottom",   0.0,  89.0),
    ("Iso",    135.0, -20.0),
    ("Iso Back",-45.0, -20.0),
]

# ── Built-in example MuJoCo XML scenes ────────────────────────
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

EXAMPLES["Walker"] = """
<mujoco model="walker">
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <default><joint damping="1" armature="0.05"/><geom friction="0.8 0.02 0.02" type="capsule"/></default>
  <worldbody>
    <light pos="0 0 5"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.3 0.3 0.3 1"/>
    <body name="torso" pos="0 0 1.1">
      <joint name="torso_z" type="slide" axis="0 0 1" damping="5"/>
      <joint name="torso_rx" type="hinge" axis="1 0 0" damping="2"/>
      <joint name="torso_ry" type="hinge" axis="0 1 0" damping="2"/>
      <geom size="0.06 0.2" rgba="0.2 0.6 0.9 1" mass="3"/>
      <body name="thigh_R" pos="0 -0.06 -0.2">
        <joint name="hip_R" type="hinge" axis="0 1 0" range="-90 60" damping="1"/>
        <geom size="0.045 0.18" rgba="0.9 0.5 0.2 1" mass="1"/>
        <body name="shin_R" pos="0 0 -0.36">
          <joint name="knee_R" type="hinge" axis="0 1 0" range="-150 0" damping="0.5"/>
          <geom size="0.035 0.17" rgba="0.9 0.5 0.2 1" mass="0.8"/>
          <geom name="foot_R" type="sphere" pos="0 0 -0.17" size="0.06" rgba="0.3 0.3 0.3 1" mass="0.3"/>
        </body>
      </body>
      <body name="thigh_L" pos="0 0.06 -0.2">
        <joint name="hip_L" type="hinge" axis="0 1 0" range="-90 60" damping="1"/>
        <geom size="0.045 0.18" rgba="0.2 0.8 0.5 1" mass="1"/>
        <body name="shin_L" pos="0 0 -0.36">
          <joint name="knee_L" type="hinge" axis="0 1 0" range="-150 0" damping="0.5"/>
          <geom size="0.035 0.17" rgba="0.2 0.8 0.5 1" mass="0.8"/>
          <geom name="foot_L" type="sphere" pos="0 0 -0.17" size="0.06" rgba="0.3 0.3 0.3 1" mass="0.3"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_R" joint="hip_R" ctrlrange="-50 50" ctrllimited="true"/>
    <motor name="knee_R" joint="knee_R" ctrlrange="-50 50" ctrllimited="true"/>
    <motor name="hip_L" joint="hip_L" ctrlrange="-50 50" ctrllimited="true"/>
    <motor name="knee_L" joint="knee_L" ctrlrange="-50 50" ctrllimited="true"/>
    <motor name="torso_rx" joint="torso_rx" ctrlrange="-30 30" ctrllimited="true"/>
    <motor name="torso_ry" joint="torso_ry" ctrlrange="-30 30" ctrllimited="true"/>
  </actuator>
</mujoco>
"""

# ── Dark theme stylesheet ─────────────────────────────────────
DARK_STYLE = """
QMainWindow, QWidget { background-color: #1a1b26; color: #a9b1d6; font-family: 'Segoe UI', 'Arial', sans-serif; font-size: 12px; }
QMenuBar { background-color: #1a1b26; border-bottom: 1px solid #24283b; padding: 2px; }
QMenuBar::item { padding: 4px 10px; } QMenuBar::item:selected { background-color: #24283b; border-radius: 4px; }
QMenu { background-color: #1a1b26; border: 1px solid #3d59a1; padding: 4px; } QMenu::item { padding: 5px 24px; } QMenu::item:selected { background-color: #3d59a1; border-radius: 3px; } QMenu::separator { height: 1px; background: #24283b; margin: 4px 8px; }
QToolBar { background-color: #16161e; border-bottom: 1px solid #24283b; spacing: 4px; padding: 3px 6px; }
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
QLineEdit { background-color: #16161e; border: 1px solid #3b4261; border-radius: 4px; padding: 4px 8px; color: #a9b1d6; }
QDialog { background-color: #1a1b26; }
QToolTip { background-color: #24283b; color: #c0caf5; border: 1px solid #3d59a1; padding: 4px; border-radius: 4px; }
"""