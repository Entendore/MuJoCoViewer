"""Shared constants, enums, the dark theme stylesheet, and built-in example scenes.

Includes version-safe enum accessors for MuJoCo compatibility across versions.
"""

import os
import mujoco


# ── XML Examples Folder ───────────────────────────────────────
EXAMPLE_XML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xml")


# ═══════════════════════════════════════════════════════════════
#  MuJoCo Version-Safe Enum Accessors
# ═══════════════════════════════════════════════════════════════

def safe_vis_flag(name: str, default=None):
    """Safely get a mjtVisFlag value. Returns *default* if the flag
    doesn't exist in this MuJoCo version."""
    return getattr(mujoco.mjtVisFlag, name, default)


def safe_enum(enum_type_name: str, member_name: str, default=None):
    """Safely get any MuJoCo enum member.
    Example: safe_enum('mjtCatBit', 'mjCAT_ALL', 0)
    """
    enum_type = getattr(mujoco, enum_type_name, None)
    if enum_type is None:
        return default
    return getattr(enum_type, member_name, default)


# Pre-resolved commonly used flags (None if unavailable)
VIS_SHADOW        = safe_vis_flag("mjVIS_SHADOW")
VIS_CONTACTPOINT  = safe_vis_flag("mjVIS_CONTACTPOINT")
VIS_CONTACTFORCE  = safe_vis_flag("mjVIS_CONTACTFORCE")
VIS_JOINT         = safe_vis_flag("mjVIS_JOINT")
VIS_ACTUATOR      = safe_vis_flag("mjVIS_ACTUATOR")
VIS_SELECT        = safe_vis_flag("mjVIS_SELECT")

# Default visualization flags to enable on model load
VIS_FLAG_DEFAULTS: list[tuple[str, object]] = []
for _name in ("mjVIS_SHADOW", "mjVIS_CONTACTPOINT", "mjVIS_CONTACTFORCE",
              "mjVIS_JOINT", "mjVIS_ACTUATOR"):
    _flag = safe_vis_flag(_name)
    if _flag is not None:
        VIS_FLAG_DEFAULTS.append((_name, _flag))


# ── Visualization flags (for the Options panel checkboxes) ────
VIS_FLAGS = []
_vis_flag_names = [
    "mjVIS_JOINT", "mjVIS_ACTUATOR", "mjVIS_CONTACTPOINT",
    "mjVIS_CONTACTFORCE", "mjVIS_COM", "mjVIS_BODYFRAME",
    "mjVIS_BODYPOS", "mjVIS_WIREFRAME", "mjVIS_TRANSPARENT",
    "mjVIS_SHADOW", "mjVIS_LIGHT", "mjVIS_TEXTURE",
    "mjVIS_SPLINE", "mjVIS_INERTIA",
]
for name in _vis_flag_names:
    flag = safe_vis_flag(name)
    if flag is not None:
        VIS_FLAGS.append((name[6:].replace('_', ' ').title(), flag))


# ── Joint type display names ───────────────────────────────────
JOINT_TYPE_NAMES = {}
for name in ["mjJNT_FREE", "mjJNT_BALL", "mjJNT_SLIDE", "mjJNT_HINGE"]:
    val = safe_enum("mjtJoint", name)
    if val is not None:
        JOINT_TYPE_NAMES[val] = name[6:].title()


# ── qpos dimension per joint type ─────────────────────────────
QPOS_DIMS = {}
for _jname, _dim in [("mjJNT_FREE", 7), ("mjJNT_BALL", 4),
                      ("mjJNT_SLIDE", 1), ("mjJNT_HINGE", 1)]:
    _val = safe_enum("mjtJoint", _jname)
    if _val is not None:
        QPOS_DIMS[_val] = _dim


# ── Actuator dynamics type display names ───────────────────────
ACTUATOR_TYPE_NAMES = {}
for name in ["mjDYN_NONE", "mjDYN_INTEGRATOR", "mjDYN_FILTER",
             "mjDYN_MUSCLE", "mjDYN_USER"]:
    val = safe_enum("mjtDyn", name)
    if val is not None:
        ACTUATOR_TYPE_NAMES[int(val)] = name[6:].title()
if not ACTUATOR_TYPE_NAMES:
    ACTUATOR_TYPE_NAMES = {
        0: "None", 1: "Integrator", 2: "Filter", 3: "Muscle", 4: "User",
    }


# ── Sensor type display names ──────────────────────────────────
SENSOR_TYPE_NAMES = {}
if hasattr(mujoco, 'mjtSensor'):
    for attr_name in dir(mujoco.mjtSensor):
        if attr_name.startswith("mjSENS_"):
            val = getattr(mujoco.mjtSensor, attr_name)
            display = attr_name[7:].replace('_', ' ').title()
            SENSOR_TYPE_NAMES[int(val)] = display
if not SENSOR_TYPE_NAMES:
    SENSOR_TYPE_NAMES = {
        0: "None", 1: "Magnetometer", 2: "Gyro", 3: "Accelerometer",
        4: "Velocimeter", 5: "GyroF", 6: "AccelerometerF",
        7: "VelocimeterF", 8: "Force", 9: "Torque", 10: "ForceF",
        11: "TorqueF", 12: "Jointpos", 13: "Jointvel",
        14: "Tendonpos", 15: "Tendonvel", 16: "Actuatorpos",
        17: "Actuatorvel", 18: "Actuatorfrc",
        19: "Balljointang", 20: "Balljointvel",
        21: "Jointlimitpos", 22: "Jointlimitvel",
        23: "Tendonlimitpos", 24: "Tendonlimitvel",
        25: "Framepos", 26: "Framequat", 27: "Framexaxis",
        28: "Frameyaxis", 29: "Framezaxis",
        30: "Framelinvel", 31: "Frameangvel",
        32: "Framelinacc", 33: "Frameangacc",
        34: "Subtreecom", 35: "Subtreelinvel", 36: "Subtreeangmom",
        100: "User",
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


# ── Label modes ───────────────────────────────────────────────
LABEL_MODES = [
    ("None", 0), ("Body", 1), ("Joint", 2), ("Geom", 3),
    ("Site", 4), ("Camera", 5), ("Light", 6), ("All", 7),
]


# ── Max recent files ──────────────────────────────────────────
MAX_RECENT_FILES = 8


# ── Simulation speed table ────────────────────────────────────
SPEED_TABLE = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 10.0]


# ═══════════════════════════════════════════════════════════════
#  Built-in SEED example MuJoCo XML scenes
# ═══════════════════════════════════════════════════════════════
#
#  10 examples covering diverse physics and MuJoCo features:
#    1. Demo Scene       — Multi-mechanism (pendulum + arm + cartpole)
#    2. Cartpole         — Classic underactuated control
#    3. Ant (Quadruped)  — Legged locomotion
#    4. Double Pendulum  — Chaotic dynamics
#    5. Humanoid Stick   — Anthropomorphic biped
#    6. Bouncing Balls   — Free-body collisions
#    7. Robotic Arm      — 4-DOF manipulator with gripper
#    8. Pendulum Wave    — Visual wave pattern from length detuning
#    9. Stacking Blocks  — Contact physics & stability
#   10. Coupled Pendulums— Spring-coupled oscillators via tendons
#
# ═══════════════════════════════════════════════════════════════

SEED_EXAMPLES = {}

# ───────────────────────────────────────────────────────────────
#  1. Demo Scene — multi-mechanism showcase
# ───────────────────────────────────────────────────────────────
SEED_EXAMPLES["Demo Scene"] = """
<mujoco model="demo_scene">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <default>
    <joint damping="0.5" armature="0.01"/>
    <geom friction="0.8 0.02 0.02" condim="3"/>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.35 0.35 0.35" rgb2="0.2 0.2 0.2"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="5 5 0.1" material="grid"/>

    <!-- Double pendulum -->
    <body name="pendulum_base" pos="0 0 2.5">
      <geom type="cylinder" size="0.06 0.04" rgba="0.5 0.5 0.55 1" mass="0.5"/>
      <joint name="pend1" type="hinge" axis="0 1 0" damping="0.2"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.9" size="0.04"
            rgba="0.85 0.35 0.15 1" mass="0.5"/>
      <body name="pend_mid" pos="0 0 -0.9">
        <joint name="pend2" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.7" size="0.035"
              rgba="0.2 0.4 0.85 1" mass="0.3"/>
        <geom name="pend_bob" type="sphere" pos="0 0 -0.7" size="0.09"
              rgba="1 0.75 0.1 1" mass="1"/>
      </body>
    </body>

    <!-- 4-DOF arm -->
    <body name="arm_base" pos="2.5 0 0.3">
      <joint name="arm_yaw" type="hinge" axis="0 0 1" damping="2"/>
      <geom type="cylinder" size="0.1 0.12" rgba="0.45 0.45 0.5 1" mass="2"/>
      <body name="arm_shoulder" pos="0 0 0.25">
        <joint name="arm_pitch" type="hinge" axis="0 1 0" range="-90 45" damping="1"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.6" size="0.05"
              rgba="0.85 0.6 0.2 1" mass="1"/>
        <body name="arm_elbow" pos="0 0 0.6">
          <joint name="arm_elbow" type="hinge" axis="0 1 0" range="-135 0" damping="0.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.45" size="0.04"
                rgba="0.85 0.6 0.2 1" mass="0.6"/>
          <body name="arm_wrist" pos="0 0 0.45">
            <joint name="arm_wrist" type="hinge" axis="0 0 1" damping="0.3"/>
            <geom type="sphere" size="0.05" rgba="0.3 0.3 0.35 1" mass="0.3"/>
          </body>
        </body>
      </body>
    </body>

    <!-- Cart-pole -->
    <body name="cart" pos="-2.5 0 0.25">
      <joint name="cart_x" type="slide" axis="1 0 0" range="-2 2" damping="0.5"/>
      <geom type="box" size="0.22 0.13 0.08" rgba="0.7 0.7 0.25 1" mass="1"/>
      <geom type="cylinder" pos="0.15 0.1 -0.06" size="0.04 0.02"
            rgba="0.3 0.3 0.3 1" mass="0.05"/>
      <geom type="cylinder" pos="-0.15 0.1 -0.06" size="0.04 0.02"
            rgba="0.3 0.3 0.3 1" mass="0.05"/>
      <body name="pole" pos="0 0 0.08">
        <joint name="pole_hinge" type="hinge" axis="0 1 0" damping="0.05"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.8" size="0.025"
              rgba="0.9 0.5 0.2 1" mass="0.3"/>
        <geom name="pole_tip" type="sphere" pos="0 0 0.8" size="0.04"
              rgba="1 0.8 0.2 1" mass="0.1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="cart_force" joint="cart_x" ctrlrange="-50 50" ctrllimited="true"/>
    <motor name="arm_yaw_m" joint="arm_yaw" ctrlrange="-20 20" ctrllimited="true"/>
    <motor name="arm_pitch_m" joint="arm_pitch" ctrlrange="-30 30" ctrllimited="true"/>
    <motor name="arm_elbow_m" joint="arm_elbow" ctrlrange="-20 20" ctrllimited="true"/>
  </actuator>

  <sensor>
    <jointpos name="pend1_pos" joint="pend1"/>
    <jointvel name="pend1_vel" joint="pend1"/>
    <jointpos name="pend2_pos" joint="pend2"/>
    <jointvel name="pend2_vel" joint="pend2"/>
    <jointpos name="cart_pos" joint="cart_x"/>
    <jointvel name="cart_vel" joint="cart_x"/>
    <jointpos name="pole_angle" joint="pole_hinge"/>
    <jointvel name="pole_vel" joint="pole_hinge"/>
    <actuatorpos name="cart_act_pos" actuator="cart_force"/>
    <actuatorfrc name="cart_act_frc" actuator="cart_force"/>
    <framepos name="pend_bob_pos" objtype="geom" objname="pend_bob"/>
    <framepos name="arm_wrist_pos" objtype="body" objname="arm_wrist"/>
    <framepos name="pole_tip_pos" objtype="geom" objname="pole_tip"/>
  </sensor>
</mujoco>
"""

# ───────────────────────────────────────────────────────────────
#  2. Cartpole — classic underactuated control
# ───────────────────────────────────────────────────────────────
SEED_EXAMPLES["Cartpole"] = """
<mujoco model="cartpole">
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <default>
    <geom friction="0.8 0.02 0.02" condim="3"/>
  </default>
  <worldbody>
    <light pos="0 0 5"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.25 0.25 0.3 1"/>
    <!-- Rail visual -->
    <geom name="rail" type="capsule" fromto="-3 0.12 0.2 3 0.12 0.2"
          size="0.015" rgba="0.4 0.4 0.45 1" contype="0" conaffinity="0"/>
    <body name="cart" pos="0 0 0.2">
      <joint name="slider" type="slide" axis="1 0 0" range="-3 3"/>
      <geom type="box" size="0.2 0.12 0.08" rgba="0.75 0.75 0.3 1" mass="1"/>
      <geom type="cylinder" pos="0.12 0.09 -0.06" size="0.035 0.015"
            rgba="0.35 0.35 0.35 1" contype="0" conaffinity="0" mass="0.05"/>
      <geom type="cylinder" pos="-0.12 0.09 -0.06" size="0.035 0.015"
            rgba="0.35 0.35 0.35 1" contype="0" conaffinity="0" mass="0.05"/>
      <body name="pole" pos="0 0 0.12">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.9" size="0.03"
              rgba="0.25 0.7 0.3 1" mass="0.5"/>
        <geom name="pole_tip" type="sphere" pos="0 0 0.9" size="0.045"
              rgba="0.9 0.85 0.2 1" mass="0.15"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="force" joint="slider" ctrlrange="-100 100" ctrllimited="true"/>
  </actuator>
  <sensor>
    <jointpos name="slider_pos" joint="slider"/>
    <jointvel name="slider_vel" joint="slider"/>
    <jointpos name="hinge_pos" joint="hinge"/>
    <jointvel name="hinge_vel" joint="hinge"/>
    <actuatorfrc name="force_applied" actuator="force"/>
    <framepos name="pole_tip_pos" objtype="geom" objname="pole_tip"/>
  </sensor>
</mujoco>
"""

# ───────────────────────────────────────────────────────────────
#  3. Ant (Quadruped) — legged locomotion
# ───────────────────────────────────────────────────────────────
SEED_EXAMPLES["Ant (Quadruped)"] = """
<mujoco model="ant">
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <default>
    <geom friction="0.8 0.005 0.0001" condim="3"/>
    <joint damping="0.5" armature="0.1"/>
  </default>
  <worldbody>
    <light pos="0 0 5"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.25 0.25 0.3 1"/>
    <body name="torso" pos="0 0 0.8">
      <freejoint/>
      <geom name="body" type="ellipsoid" size="0.3 0.2 0.15" rgba="0.85 0.2 0.2 1" mass="3"/>
      <geom name="head" type="sphere" pos="0 0.18 0.08" size="0.07"
            rgba="0.9 0.5 0.2 1" mass="0.3"/>
      <!-- Front-right leg -->
      <body name="FR1" pos="0.22 0.15 -0.1">
        <joint name="hip_FR" type="hinge" axis="0 0 1" range="-40 40"/>
        <geom type="capsule" fromto="0 0 0 0.15 0.15 -0.18" size="0.04"
              rgba="0.25 0.25 0.85 1" mass="0.4"/>
        <body name="FR2" pos="0.15 0.15 -0.18">
          <joint name="ankle_FR" type="hinge" axis="0 1 0" range="-30 70"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.035"
                rgba="0.25 0.25 0.85 1" mass="0.25"/>
          <geom name="FR_foot" type="sphere" pos="0 0 -0.25" size="0.03"
                rgba="0.4 0.4 0.4 1" mass="0.1"/>
        </body>
      </body>
      <!-- Front-left leg -->
      <body name="FL1" pos="-0.22 0.15 -0.1">
        <joint name="hip_FL" type="hinge" axis="0 0 1" range="-40 40"/>
        <geom type="capsule" fromto="0 0 0 -0.15 0.15 -0.18" size="0.04"
              rgba="0.25 0.7 0.3 1" mass="0.4"/>
        <body name="FL2" pos="-0.15 0.15 -0.18">
          <joint name="ankle_FL" type="hinge" axis="0 1 0" range="-30 70"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.035"
                rgba="0.25 0.7 0.3 1" mass="0.25"/>
          <geom name="FL_foot" type="sphere" pos="0 0 -0.25" size="0.03"
                rgba="0.4 0.4 0.4 1" mass="0.1"/>
        </body>
      </body>
      <!-- Back-right leg -->
      <body name="BR1" pos="0.22 -0.15 -0.1">
        <joint name="hip_BR" type="hinge" axis="0 0 1" range="-40 40"/>
        <geom type="capsule" fromto="0 0 0 0.15 -0.15 -0.18" size="0.04"
              rgba="0.85 0.85 0.2 1" mass="0.4"/>
        <body name="BR2" pos="0.15 -0.15 -0.18">
          <joint name="ankle_BR" type="hinge" axis="0 1 0" range="-30 70"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.035"
                rgba="0.85 0.85 0.2 1" mass="0.25"/>
          <geom name="BR_foot" type="sphere" pos="0 0 -0.25" size="0.03"
                rgba="0.4 0.4 0.4 1" mass="0.1"/>
        </body>
      </body>
      <!-- Back-left leg -->
      <body name="BL1" pos="-0.22 -0.15 -0.1">
        <joint name="hip_BL" type="hinge" axis="0 0 1" range="-40 40"/>
        <geom type="capsule" fromto="0 0 0 -0.15 -0.15 -0.18" size="0.04"
              rgba="0.85 0.25 0.8 1" mass="0.4"/>
        <body name="BL2" pos="-0.15 -0.15 -0.18">
          <joint name="ankle_BL" type="hinge" axis="0 1 0" range="-30 70"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.035"
                rgba="0.85 0.25 0.8 1" mass="0.25"/>
          <geom name="BL_foot" type="sphere" pos="0 0 -0.25" size="0.03"
                rgba="0.4 0.4 0.4 1" mass="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_FR_m" joint="hip_FR" ctrlrange="-1 1"/>
    <motor name="ankle_FR_m" joint="ankle_FR" ctrlrange="-1 1"/>
    <motor name="hip_FL_m" joint="hip_FL" ctrlrange="-1 1"/>
    <motor name="ankle_FL_m" joint="ankle_FL" ctrlrange="-1 1"/>
    <motor name="hip_BR_m" joint="hip_BR" ctrlrange="-1 1"/>
    <motor name="ankle_BR_m" joint="ankle_BR" ctrlrange="-1 1"/>
    <motor name="hip_BL_m" joint="hip_BL" ctrlrange="-1 1"/>
    <motor name="ankle_BL_m" joint="ankle_BL" ctrlrange="-1 1"/>
  </actuator>
  <sensor>
    <jointpos name="hip_FR_pos" joint="hip_FR"/>
    <jointpos name="hip_FL_pos" joint="hip_FL"/>
    <jointpos name="hip_BR_pos" joint="hip_BR"/>
    <jointpos name="hip_BL_pos" joint="hip_BL"/>
    <actuatorfrc name="hip_FR_frc" actuator="hip_FR_m"/>
    <actuatorfrc name="hip_FL_frc" actuator="hip_FL_m"/>
    <actuatorfrc name="hip_BR_frc" actuator="hip_BR_m"/>
    <actuatorfrc name="hip_BL_frc" actuator="hip_BL_m"/>
    <framepos name="torso_pos" objtype="body" objname="torso"/>
    <framequat name="torso_quat" objtype="body" objname="torso"/>
    <framelinvel name="torso_vel" objtype="body" objname="torso"/>
  </sensor>
</mujoco>
"""

# ───────────────────────────────────────────────────────────────
#  4. Double Pendulum — chaotic dynamics
# ───────────────────────────────────────────────────────────────
SEED_EXAMPLES["Double Pendulum"] = """
<mujoco model="double_pendulum">
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <default>
    <joint damping="0.03" armature="0.01"/>
    <geom friction="0.5 0.02 0.02"/>
  </default>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.18 0.18 0.22" rgb2="0.12 0.12 0.15"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="4 4"/>
  </asset>
  <worldbody>
    <light pos="0 0 5" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="5 5 0.1" material="floor_mat"/>
    <body name="pillar" pos="0 0 1.8">
      <geom type="cylinder" size="0.05 1.8" rgba="0.45 0.45 0.5 1" mass="3"/>
      <body name="arm1" pos="0 0 0">
        <joint name="j1" type="hinge" axis="0 1 0" damping="0.02"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.85" size="0.035"
              rgba="0.9 0.3 0.2 1" mass="1"/>
        <body name="arm2" pos="0 0 -0.85">
          <joint name="j2" type="hinge" axis="0 1 0" damping="0.01"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.65" size="0.03"
                rgba="0.2 0.5 0.9 1" mass="0.6"/>
          <geom name="bob" type="sphere" pos="0 0 -0.65" size="0.07"
                rgba="1 0.85 0.1 1" mass="1.5"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="perturb1" joint="j1" ctrlrange="-5 5" ctrllimited="true"/>
    <motor name="perturb2" joint="j2" ctrlrange="-3 3" ctrllimited="true"/>
  </actuator>
  <sensor>
    <jointpos name="j1_pos" joint="j1"/>
    <jointvel name="j1_vel" joint="j1"/>
    <jointpos name="j2_pos" joint="j2"/>
    <jointvel name="j2_vel" joint="j2"/>
    <framepos name="bob_pos" objtype="geom" objname="bob"/>
    <framelinvel name="bob_vel" objtype="geom" objname="bob"/>
  </sensor>
</mujoco>
"""

# ───────────────────────────────────────────────────────────────
#  5. Humanoid Stick — simple biped
# ───────────────────────────────────────────────────────────────
SEED_EXAMPLES["Humanoid Stick"] = """
<mujoco model="humanoid_stick">
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <default>
    <joint damping="3" armature="0.1"/>
    <geom friction="0.8 0.02 0.02" condim="3"/>
  </default>
  <worldbody>
    <light pos="0 0 5" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.25 0.25 0.3 1"/>
    <body name="torso" pos="0 0 1.05">
      <freejoint name="root"/>
      <geom name="torso_geom" type="capsule" fromto="0 0 -0.1 0 0 0.35"
            size="0.08" rgba="0.85 0.25 0.25 1" mass="4"/>
      <geom name="head" type="sphere" pos="0 0 0.48" size="0.09"
            rgba="1 0.85 0.7 1" mass="0.8"/>
      <!-- Right arm -->
      <body name="r_upper_arm" pos="0.1 0 0.3">
        <joint name="r_shoulder" type="hinge" axis="0 1 0" range="-180 60"/>
        <geom type="capsule" fromto="0 0 0 0.18 0 -0.08" size="0.03"
              rgba="0.85 0.25 0.25 1" mass="0.4"/>
        <body name="r_forearm" pos="0.18 0 -0.08">
          <joint name="r_elbow" type="hinge" axis="0 1 0" range="-150 0"/>
          <geom type="capsule" fromto="0 0 0 0.14 0 0" size="0.025"
                rgba="0.85 0.25 0.25 1" mass="0.25"/>
        </body>
      </body>
      <!-- Left arm -->
      <body name="l_upper_arm" pos="-0.1 0 0.3">
        <joint name="l_shoulder" type="hinge" axis="0 1 0" range="-180 60"/>
        <geom type="capsule" fromto="0 0 0 -0.18 0 -0.08" size="0.03"
              rgba="0.25 0.4 0.85 1" mass="0.4"/>
        <body name="l_forearm" pos="-0.18 0 -0.08">
          <joint name="l_elbow" type="hinge" axis="0 1 0" range="-150 0"/>
          <geom type="capsule" fromto="0 0 0 -0.14 0 0" size="0.025"
                rgba="0.25 0.4 0.85 1" mass="0.25"/>
        </body>
      </body>
      <!-- Right leg -->
      <body name="r_thigh" pos="0.06 0 -0.1">
        <joint name="r_hip" type="hinge" axis="0 1 0" range="-30 120"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.32" size="0.04"
              rgba="0.85 0.25 0.25 1" mass="1.2"/>
        <body name="r_shin" pos="0 0 -0.32">
          <joint name="r_knee" type="hinge" axis="0 1 0" range="-130 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.32" size="0.035"
                rgba="0.85 0.25 0.25 1" mass="0.8"/>
          <geom name="r_foot" type="box" pos="0 0.03 -0.32"
                size="0.04 0.08 0.02" rgba="0.35 0.25 0.15 1" mass="0.3"/>
        </body>
      </body>
      <!-- Left leg -->
      <body name="l_thigh" pos="-0.06 0 -0.1">
        <joint name="l_hip" type="hinge" axis="0 1 0" range="-30 120"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.32" size="0.04"
              rgba="0.25 0.4 0.85 1" mass="1.2"/>
        <body name="l_shin" pos="0 0 -0.32">
          <joint name="l_knee" type="hinge" axis="0 1 0" range="-130 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.32" size="0.035"
                rgba="0.25 0.4 0.85 1" mass="0.8"/>
          <geom name="l_foot" type="box" pos="0 0.03 -0.32"
                size="0.04 0.08 0.02" rgba="0.35 0.25 0.15 1" mass="0.3"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="r_hip_m" joint="r_hip" ctrlrange="-80 80" ctrllimited="true"/>
    <motor name="l_hip_m" joint="l_hip" ctrlrange="-80 80" ctrllimited="true"/>
    <motor name="r_knee_m" joint="r_knee" ctrlrange="-80 80" ctrllimited="true"/>
    <motor name="l_knee_m" joint="l_knee" ctrlrange="-80 80" ctrllimited="true"/>
    <motor name="r_shoulder_m" joint="r_shoulder" ctrlrange="-30 30" ctrllimited="true"/>
    <motor name="l_shoulder_m" joint="l_shoulder" ctrlrange="-30 30" ctrllimited="true"/>
  </actuator>
  <sensor>
    <framepos name="torso_pos" objtype="body" objname="torso"/>
    <framequat name="torso_quat" objtype="body" objname="torso"/>
    <framelinvel name="torso_vel" objtype="body" objname="torso"/>
    <jointpos name="r_hip_pos" joint="r_hip"/>
    <jointpos name="l_hip_pos" joint="l_hip"/>
    <jointpos name="r_knee_pos" joint="r_knee"/>
    <jointpos name="l_knee_pos" joint="l_knee"/>
    <actuatorfrc name="r_hip_frc" actuator="r_hip_m"/>
    <actuatorfrc name="l_hip_frc" actuator="l_hip_m"/>
  </sensor>
</mujoco>
"""

# ───────────────────────────────────────────────────────────────
#  6. Bouncing Balls — free-body collisions
# ───────────────────────────────────────────────────────────────
SEED_EXAMPLES["Bouncing Balls"] = """
<mujoco model="bouncing_balls">
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <default>
    <geom friction="0.6 0.02 0.02" condim="3"/>
  </default>
  <worldbody>
    <light pos="0 0 8" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.22 0.22 0.28 1"/>
    <!-- Walls -->
    <geom name="wall_x1" type="plane" size="0.01 5 5" pos="1.5 0 2.5"
          rgba="0.3 0.3 0.35 0.3" euler="0 90 0"/>
    <geom name="wall_x2" type="plane" size="0.01 5 5" pos="-1.5 0 2.5"
          rgba="0.3 0.3 0.35 0.3" euler="0 -90 0"/>
    <geom name="wall_y1" type="plane" size="5 0.01 5" pos="0 1.5 2.5"
          rgba="0.3 0.3 0.35 0.3" euler="-90 0 0"/>
    <geom name="wall_y2" type="plane" size="5 0.01 5" pos="0 -1.5 2.5"
          rgba="0.3 0.3 0.35 0.3" euler="90 0 0"/>

    <body name="ball_red" pos="0 0 3">
      <freejoint/>
      <geom type="sphere" size="0.18" rgba="0.9 0.2 0.2 1" mass="1"/>
    </body>
    <body name="ball_green" pos="0.4 0 3.8">
      <freejoint/>
      <geom type="sphere" size="0.14" rgba="0.2 0.75 0.3 1" mass="0.5"/>
    </body>
    <body name="ball_blue" pos="-0.3 0.3 4.5">
      <freejoint/>
      <geom type="sphere" size="0.22" rgba="0.2 0.35 0.9 1" mass="1.5"/>
    </body>
    <body name="ball_yellow" pos="0.6 -0.2 2.5">
      <freejoint/>
      <geom type="sphere" size="0.11" rgba="0.9 0.85 0.15 1" mass="0.3"/>
    </body>
    <body name="ball_purple" pos="-0.5 0.4 5">
      <freejoint/>
      <geom type="sphere" size="0.16" rgba="0.75 0.2 0.8 1" mass="0.7"/>
    </body>
    <body name="ball_orange" pos="0.2 -0.5 3.3">
      <freejoint/>
      <geom type="sphere" size="0.2" rgba="0.95 0.55 0.15 1" mass="1.2"/>
    </body>
  </worldbody>
  <sensor>
    <framepos name="ball_red_pos" objtype="body" objname="ball_red"/>
    <framepos name="ball_blue_pos" objtype="body" objname="ball_blue"/>
    <framelinvel name="ball_red_vel" objtype="body" objname="ball_red"/>
    <framelinvel name="ball_blue_vel" objtype="body" objname="ball_blue"/>
  </sensor>
</mujoco>
"""

# ───────────────────────────────────────────────────────────────
#  7. Robotic Arm — 4-DOF manipulator with gripper
# ───────────────────────────────────────────────────────────────
SEED_EXAMPLES["Robotic Arm"] = """
<mujoco model="robotic_arm">
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <default>
    <joint damping="2" armature="0.1"/>
    <geom friction="0.8 0.02 0.02" condim="3"/>
  </default>
  <worldbody>
    <light pos="0 0 5" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.22 0.22 0.28 1"/>
    <!-- Target object -->
    <body name="target" pos="0.4 0 0.04">
      <freejoint/>
      <geom type="box" size="0.04 0.04 0.04" rgba="0.2 0.7 0.9 1" mass="0.1"/>
    </body>
    <!-- Base -->
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.12 0.1" rgba="0.4 0.4 0.45 1" mass="8"/>
      <!-- Shoulder -->
      <body name="shoulder" pos="0 0 0.1">
        <joint name="base_rotate" type="hinge" axis="0 0 1" damping="3" range="-180 180"/>
        <geom type="cylinder" size="0.06 0.03" rgba="0.55 0.55 0.6 1" mass="1.5"/>
        <!-- Upper arm -->
        <body name="upper_arm" pos="0 0 0.03">
          <joint name="shoulder_pitch" type="hinge" axis="0 1 0" range="-90 90" damping="2"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.55" size="0.04"
                rgba="0.9 0.55 0.15 1" mass="2.5"/>
          <!-- Forearm -->
          <body name="forearm" pos="0 0 0.55">
            <joint name="elbow_pitch" type="hinge" axis="0 1 0" range="-135 0" damping="1.5"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.4" size="0.035"
                  rgba="0.9 0.55 0.15 1" mass="1.5"/>
            <!-- Wrist -->
            <body name="wrist" pos="0 0 0.4">
              <joint name="wrist_rotate" type="hinge" axis="0 0 1"
                     range="-180 180" damping="0.5"/>
              <geom type="sphere" size="0.03" rgba="0.5 0.5 0.55 1" mass="0.4"/>
              <!-- Gripper -->
              <body name="gripper_base" pos="0 0 0.035">
                <geom type="box" size="0.03 0.025 0.015"
                      rgba="0.45 0.45 0.5 1" mass="0.2"/>
                <body name="l_finger" pos="-0.025 0 0.02">
                  <joint name="l_grip" type="slide" axis="0 1 0" range="-0.03 0.03"/>
                  <geom type="box" size="0.008 0.025 0.012"
                        rgba="0.65 0.65 0.7 1" mass="0.05"/>
                </body>
                <body name="r_finger" pos="0.025 0 0.02">
                  <joint name="r_grip" type="slide" axis="0 1 0" range="-0.03 0.03"/>
                  <geom type="box" size="0.008 0.025 0.012"
                        rgba="0.65 0.65 0.7 1" mass="0.05"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="base_m" joint="base_rotate" ctrlrange="-40 40" ctrllimited="true"/>
    <motor name="shoulder_m" joint="shoulder_pitch" ctrlrange="-60 60" ctrllimited="true"/>
    <motor name="elbow_m" joint="elbow_pitch" ctrlrange="-50 50" ctrllimited="true"/>
    <motor name="wrist_m" joint="wrist_rotate" ctrlrange="-15 15" ctrllimited="true"/>
    <motor name="l_grip_m" joint="l_grip" ctrlrange="-1 1" ctrllimited="true"/>
    <motor name="r_grip_m" joint="r_grip" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
  <sensor>
    <jointpos name="base_pos" joint="base_rotate"/>
    <jointpos name="shoulder_pos" joint="shoulder_pitch"/>
    <jointpos name="elbow_pos" joint="elbow_pitch"/>
    <jointpos name="wrist_pos" joint="wrist_rotate"/>
    <framepos name="end_effector_pos" objtype="body" objname="gripper_base"/>
    <framelinvel name="end_effector_vel" objtype="body" objname="gripper_base"/>
    <actuatorfrc name="shoulder_frc" actuator="shoulder_m"/>
    <actuatorfrc name="elbow_frc" actuator="elbow_m"/>
  </sensor>
</mujoco>
"""

# ───────────────────────────────────────────────────────────────
#  8. Pendulum Wave — visual wave from length detuning
# ───────────────────────────────────────────────────────────────
SEED_EXAMPLES["Pendulum Wave"] = """
<mujoco model="pendulum_wave">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <default>
    <joint damping="0.005"/>
    <geom friction="0.5 0.02 0.02"/>
  </default>
  <worldbody>
    <light pos="0 0 5" diffuse="0.85 0.85 0.85"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.15 0.15 0.2 1"/>
    <body name="bar" pos="0 0 2.2">
      <geom type="capsule" fromto="-0.9 0 0 0.9 0 0" size="0.018"
            rgba="0.45 0.45 0.5 1" mass="5"/>
      <body name="p1" pos="-0.8 0 0">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.63" size="0.012"
              rgba="0.9 0.15 0.15 1" mass="0.2"/>
        <geom type="sphere" pos="0 0 -0.63" size="0.025"
              rgba="0.9 0.15 0.15 1" mass="0.1"/>
      </body>
      <body name="p2" pos="-0.6 0 0">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.57" size="0.012"
              rgba="0.85 0.3 0.1 1" mass="0.2"/>
        <geom type="sphere" pos="0 0 -0.57" size="0.025"
              rgba="0.85 0.3 0.1 1" mass="0.1"/>
      </body>
      <body name="p3" pos="-0.4 0 0">
        <joint name="j3" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.52" size="0.012"
              rgba="0.75 0.55 0.1 1" mass="0.2"/>
        <geom type="sphere" pos="0 0 -0.52" size="0.025"
              rgba="0.75 0.55 0.1 1" mass="0.1"/>
      </body>
      <body name="p4" pos="-0.2 0 0">
        <joint name="j4" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.48" size="0.012"
              rgba="0.6 0.7 0.1 1" mass="0.2"/>
        <geom type="sphere" pos="0 0 -0.48" size="0.025"
              rgba="0.6 0.7 0.1 1" mass="0.1"/>
      </body>
      <body name="p5" pos="0 0 0">
        <joint name="j5" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.44" size="0.012"
              rgba="0.2 0.75 0.2 1" mass="0.2"/>
        <geom type="sphere" pos="0 0 -0.44" size="0.025"
              rgba="0.2 0.75 0.2 1" mass="0.1"/>
      </body>
      <body name="p6" pos="0.2 0 0">
        <joint name="j6" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.40" size="0.012"
              rgba="0.1 0.6 0.7 1" mass="0.2"/>
        <geom type="sphere" pos="0 0 -0.40" size="0.025"
              rgba="0.1 0.6 0.7 1" mass="0.1"/>
      </body>
      <body name="p7" pos="0.4 0 0">
        <joint name="j7" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.37" size="0.012"
              rgba="0.15 0.4 0.85 1" mass="0.2"/>
        <geom type="sphere" pos="0 0 -0.37" size="0.025"
              rgba="0.15 0.4 0.85 1" mass="0.1"/>
      </body>
      <body name="p8" pos="0.6 0 0">
        <joint name="j8" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.35" size="0.012"
              rgba="0.35 0.2 0.85 1" mass="0.2"/>
        <geom type="sphere" pos="0 0 -0.35" size="0.025"
              rgba="0.35 0.2 0.85 1" mass="0.1"/>
      </body>
      <body name="p9" pos="0.8 0 0">
        <joint name="j9" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.32" size="0.012"
              rgba="0.7 0.15 0.75 1" mass="0.2"/>
        <geom type="sphere" pos="0 0 -0.32" size="0.025"
              rgba="0.7 0.15 0.75 1" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <sensor>
    <jointpos name="j1_pos" joint="j1"/>
    <jointpos name="j2_pos" joint="j2"/>
    <jointpos name="j3_pos" joint="j3"/>
    <jointpos name="j4_pos" joint="j4"/>
    <jointpos name="j5_pos" joint="j5"/>
    <jointpos name="j6_pos" joint="j6"/>
    <jointpos name="j7_pos" joint="j7"/>
    <jointpos name="j8_pos" joint="j8"/>
    <jointpos name="j9_pos" joint="j9"/>
  </sensor>
</mujoco>
"""

# ───────────────────────────────────────────────────────────────
#  9. Stacking Blocks — contact physics & stability
# ───────────────────────────────────────────────────────────────
SEED_EXAMPLES["Stacking Blocks"] = """
<mujoco model="stacking_blocks">
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <default>
    <geom friction="1.0 0.02 0.02" condim="3"/>
  </default>
  <worldbody>
    <light pos="0 0 5" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.22 0.22 0.28 1"/>
    <!-- Table surface -->
    <geom name="table" type="box" pos="0 0 0.35" size="0.5 0.35 0.02"
          rgba="0.45 0.35 0.25 1"/>
    <geom name="table_leg1" type="cylinder" pos="0.4 0.25 0.175"
          size="0.02 0.175" rgba="0.4 0.3 0.2 1"/>
    <geom name="table_leg2" type="cylinder" pos="-0.4 0.25 0.175"
          size="0.02 0.175" rgba="0.4 0.3 0.2 1"/>
    <geom name="table_leg3" type="cylinder" pos="0.4 -0.25 0.175"
          size="0.02 0.175" rgba="0.4 0.3 0.2 1"/>
    <geom name="table_leg4" type="cylinder" pos="-0.4 -0.25 0.175"
          size="0.02 0.175" rgba="0.4 0.3 0.2 1"/>
    <!-- Stack of blocks -->
    <body name="block1" pos="0 0 0.395">
      <freejoint/>
      <geom type="box" size="0.06 0.06 0.035" rgba="0.9 0.2 0.2 1" mass="0.3"/>
    </body>
    <body name="block2" pos="0 0 0.465">
      <freejoint/>
      <geom type="box" size="0.06 0.06 0.035" rgba="0.2 0.75 0.3 1" mass="0.3"/>
    </body>
    <body name="block3" pos="0 0 0.535">
      <freejoint/>
      <geom type="box" size="0.06 0.06 0.035" rgba="0.2 0.35 0.9 1" mass="0.3"/>
    </body>
    <body name="block4" pos="0 0 0.605">
      <freejoint/>
      <geom type="box" size="0.06 0.06 0.035" rgba="0.9 0.85 0.15 1" mass="0.3"/>
    </body>
    <body name="block5" pos="0 0 0.675">
      <freejoint/>
      <geom type="box" size="0.06 0.06 0.035" rgba="0.75 0.2 0.8 1" mass="0.3"/>
    </body>
    <!-- Sphere to knock them over -->
    <body name="cannonball" pos="-0.6 0 0.42">
      <freejoint/>
      <geom type="sphere" size="0.06" rgba="0.5 0.5 0.55 1" mass="2"/>
    </body>
  </worldbody>
  <sensor>
    <framepos name="block1_pos" objtype="body" objname="block1"/>
    <framepos name="cannonball_pos" objtype="body" objname="cannonball"/>
    <framelinvel name="cannonball_vel" objtype="body" objname="cannonball"/>
  </sensor>
</mujoco>
"""

# ───────────────────────────────────────────────────────────────
#  10. Coupled Pendulums — spring-coupled oscillators via tendons
# ───────────────────────────────────────────────────────────────
SEED_EXAMPLES["Coupled Pendulums"] = """
<mujoco model="coupled_pendulums">
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <default>
    <joint damping="0.08"/>
    <geom friction="0.5 0.02 0.02"/>
  </default>
  <worldbody>
    <light pos="0 0 5" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.2 0.2 0.25 1"/>
    <body name="bar" pos="0 0 2">
      <geom type="capsule" fromto="-0.8 0 0 0.8 0 0" size="0.02"
            rgba="0.45 0.45 0.5 1" mass="5"/>
      <body name="p1" pos="-0.6 0 0">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.8" size="0.015"
              rgba="0.9 0.25 0.2 1" mass="0.3"/>
        <geom name="bob1" type="sphere" pos="0 0 -0.8" size="0.04"
              rgba="0.9 0.25 0.2 1" mass="0.5"/>
        <site name="s1" pos="0 0 -0.8" rgba="0 0 0 0"/>
      </body>
      <body name="p2" pos="-0.3 0 0">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.8" size="0.015"
              rgba="0.3 0.7 0.3 1" mass="0.3"/>
        <geom name="bob2" type="sphere" pos="0 0 -0.8" size="0.04"
              rgba="0.3 0.7 0.3 1" mass="0.5"/>
        <site name="s2" pos="0 0 -0.8" rgba="0 0 0 0"/>
      </body>
      <body name="p3" pos="0 0 0">
        <joint name="j3" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.8" size="0.015"
              rgba="0.2 0.4 0.9 1" mass="0.3"/>
        <geom name="bob3" type="sphere" pos="0 0 -0.8" size="0.04"
              rgba="0.2 0.4 0.9 1" mass="0.5"/>
        <site name="s3" pos="0 0 -0.8" rgba="0 0 0 0"/>
      </body>
      <body name="p4" pos="0.3 0 0">
        <joint name="j4" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.8" size="0.015"
              rgba="0.9 0.85 0.15 1" mass="0.3"/>
        <geom name="bob4" type="sphere" pos="0 0 -0.8" size="0.04"
              rgba="0.9 0.85 0.15 1" mass="0.5"/>
        <site name="s4" pos="0 0 -0.8" rgba="0 0 0 0"/>
      </body>
      <body name="p5" pos="0.6 0 0">
        <joint name="j5" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.8" size="0.015"
              rgba="0.85 0.2 0.8 1" mass="0.3"/>
        <geom name="bob5" type="sphere" pos="0 0 -0.8" size="0.04"
              rgba="0.85 0.2 0.8 1" mass="0.5"/>
        <site name="s5" pos="0 0 -0.8" rgba="0 0 0 0"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="spring12" stiffness="15" damping="0.5">
      <site site="s1"/>
      <site site="s2"/>
    </spatial>
    <spatial name="spring23" stiffness="15" damping="0.5">
      <site site="s2"/>
      <site site="s3"/>
    </spatial>
    <spatial name="spring34" stiffness="15" damping="0.5">
      <site site="s3"/>
      <site site="s4"/>
    </spatial>
    <spatial name="spring45" stiffness="15" damping="0.5">
      <site site="s4"/>
      <site site="s5"/>
    </spatial>
  </tendon>
  <sensor>
    <jointpos name="j1_pos" joint="j1"/>
    <jointpos name="j2_pos" joint="j2"/>
    <jointpos name="j3_pos" joint="j3"/>
    <jointpos name="j4_pos" joint="j4"/>
    <jointpos name="j5_pos" joint="j5"/>
    <jointvel name="j1_vel" joint="j1"/>
    <jointvel name="j2_vel" joint="j2"/>
    <jointvel name="j3_vel" joint="j3"/>
    <tendonpos name="spring12_len" tendon="spring12"/>
    <tendonpos name="spring23_len" tendon="spring23"/>
    <tendonpos name="spring34_len" tendon="spring34"/>
    <tendonpos name="spring45_len" tendon="spring45"/>
  </sensor>
</mujoco>
"""


# ── Dark theme stylesheet ─────────────────────────────────────
DARK_STYLE = """
/* ── Global ── */
QMainWindow, QWidget {
    background-color: #1a1b26; color: #c0caf5;
    font-family: 'Segoe UI', 'Arial', sans-serif; font-size: 12px;
}
QMenuBar {
    background-color: #1a1b26; border-bottom: 1px solid #2a2f45; padding: 2px;
}
QMenuBar::item { padding: 4px 10px; color: #c0caf5; }
QMenuBar::item:selected { background-color: #2a2f45; border-radius: 4px; }
QMenu {
    background-color: #1a1b26; border: 1px solid #3d59a1; padding: 4px;
    color: #c0caf5;
}
QMenu::item { padding: 6px 28px; color: #c0caf5; }
QMenu::item:selected { background-color: #3d59a1; border-radius: 3px; }
QMenu::separator { height: 1px; background: #2a2f45; margin: 4px 8px; }
QToolBar {
    background-color: #16161e; border-bottom: 1px solid #2a2f45;
    spacing: 4px; padding: 3px 6px;
}
QToolButton {
    background-color: #24283b; border: 1px solid #3b4261; border-radius: 4px;
    padding: 5px 10px; color: #c0caf5; font-weight: 500;
}
QToolButton:hover { background-color: #3d59a1; color: #ffffff; }
QToolButton:pressed { background-color: #2b3f7a; }
QToolButton:checked { background-color: #3d59a1; color: #ffffff; }
QPushButton {
    background-color: #24283b; border: 1px solid #3b4261; border-radius: 5px;
    padding: 5px 14px; color: #c0caf5; font-weight: 500;
}
QPushButton:hover { background-color: #3d59a1; color: #ffffff; border-color: #5a7fd6; }
QPushButton:pressed { background-color: #2b3f7a; }
QPushButton:checked { background-color: #3d59a1; color: #ffffff; }
QPushButton:disabled { background-color: #16161e; color: #3b4261; border-color: #1e2235; }
QPushButton[class="danger"] {
    background-color: #5c2b3b; border-color: #f7768e; color: #f7768e;
}
QPushButton[class="danger"]:hover { background-color: #7a3a4e; }
QPushButton[class="success"] {
    background-color: #2b4a2b; border-color: #9ece6a; color: #9ece6a;
}
QPushButton[class="success"]:hover { background-color: #3a6a3a; }
QSlider::groove:horizontal { height: 6px; background: #1e2235; border-radius: 3px; }
QSlider::handle:horizontal {
    width: 16px; height: 16px; background: #7aa2f7; border-radius: 8px; margin: -5px 0;
}
QSlider::sub-page:horizontal { background: #3d59a1; border-radius: 3px; }
QGroupBox {
    border: 1px solid #2a2f45; border-radius: 6px;
    margin-top: 14px; padding: 14px 8px 8px 8px; font-weight: bold; color: #c0caf5;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 10px; padding: 0 6px; color: #7aa2f7;
}
QTreeWidget {
    background-color: #0d0e16; border: 1px solid #2a2f45;
    alternate-background-color: #141623; border-radius: 4px; color: #c0caf5;
}
QTreeWidget::item { padding: 3px 0; color: #c0caf5; }
QTreeWidget::item:selected { background-color: #3d59a1; color: #ffffff; }
QHeaderView::section {
    background-color: #16161e; border: none;
    border-bottom: 1px solid #2a2f45; padding: 4px;
    font-weight: bold; color: #7a85a8;
}
QTableWidget {
    background-color: #0d0e16; border: 1px solid #2a2f45;
    alternate-background-color: #141623; border-radius: 4px;
    gridline-color: #1e2235; color: #c0caf5;
}
QTableWidget::item { padding: 2px 4px; }
QTableWidget::item:selected { background-color: #3d59a1; color: #ffffff; }
QScrollArea { border: none; background: transparent; }
QTabWidget::pane { border: 1px solid #2a2f45; border-radius: 4px; top: -1px; }
QTabBar::tab {
    background-color: #16161e; border: 1px solid #2a2f45;
    padding: 7px 16px; border-top-left-radius: 6px;
    border-top-right-radius: 6px; margin-right: 2px; color: #7a85a8;
}
QTabBar::tab:selected {
    background-color: #1a1b26; border-bottom-color: #1a1b26; color: #7aa2f7;
}
QTabBar::tab:hover { background-color: #24283b; color: #c0caf5; }
QLabel { color: #c0caf5; background: transparent; }
QLabel[class="value"] { color: #7aa2f7; font-family: 'Consolas', 'Courier New', monospace; font-size: 13px; font-weight: bold; }
QLabel[class="dim"] { color: #7a85a8; font-size: 11px; }
QLabel[class="error"] { color: #f7768e; font-weight: bold; }
QLabel[class="success"] { color: #9ece6a; font-weight: bold; }
QLabel[class="warning"] { color: #e0af68; font-weight: bold; }
QStatusBar { background-color: #0d0e16; border-top: 1px solid #2a2f45; color: #7a85a8; font-size: 11px; }
QStatusBar QLabel { color: #7a85a8; padding: 0 4px; }
QComboBox {
    background-color: #24283b; border: 1px solid #3b4261; border-radius: 4px;
    padding: 4px 8px; color: #c0caf5; min-width: 80px;
}
QComboBox QAbstractItemView {
    background-color: #1a1b26; border: 1px solid #3d59a1;
    selection-background-color: #3d59a1; color: #c0caf5;
}
QComboBox::drop-down { border: none; width: 20px; }
QCheckBox { spacing: 8px; background: transparent; color: #c0caf5; }
QCheckBox::indicator {
    width: 16px; height: 16px; border: 1px solid #3b4261;
    border-radius: 4px; background-color: #1e2235;
}
QCheckBox::indicator:checked { background-color: #7aa2f7; border-color: #7aa2f7; }
QSplitter::handle { background-color: #2a2f45; width: 2px; }
QPlainTextEdit {
    background-color: #0d0e16; border: 1px solid #2a2f45; color: #c0caf5;
    font-family: 'Consolas', 'Courier New', monospace; font-size: 12px;
    border-radius: 4px; padding: 6px; selection-background-color: #3d59a1;
}
QProgressBar {
    border: 1px solid #2a2f45; border-radius: 4px;
    background-color: #0d0e16; text-align: center; color: #c0caf5;
    font-weight: bold; font-size: 11px;
}
QProgressBar::chunk { background-color: #3d59a1; border-radius: 3px; }
QSpinBox, QDoubleSpinBox {
    background-color: #1e2235; border: 1px solid #3b4261; border-radius: 4px;
    padding: 3px 6px; color: #c0caf5; font-weight: bold;
}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button {
    background-color: #2a2f45; border: none; width: 16px;
}
QLineEdit {
    background-color: #0d0e16; border: 1px solid #3b4261; border-radius: 4px;
    padding: 4px 8px; color: #c0caf5;
}
QLineEdit[class="search"] {
    background-color: #1a1b26; border: 1px solid #2a2f45; border-radius: 12px;
    padding: 4px 10px 4px 24px; color: #c0caf5; font-size: 11px;
}
QDialog { background-color: #1a1b26; color: #c0caf5; }
QToolTip {
    background-color: #24283b; color: #c0caf5; border: 1px solid #3d59a1;
    padding: 6px; border-radius: 4px; font-size: 12px;
}
QListWidget {
    background-color: #0d0e16; border: 1px solid #2a2f45; border-radius: 4px;
    color: #c0caf5;
}
QListWidget::item { padding: 5px 8px; }
QListWidget::item:selected { background-color: #3d59a1; color: #ffffff; }
QScrollBar:vertical { background-color: #16161e; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background-color: #3b4261; border-radius: 5px; min-height: 20px; }
QScrollBar::handle:vertical:hover { background-color: #5a7fd6; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal { background-color: #16161e; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background-color: #3b4261; border-radius: 5px; min-width: 20px; }
QScrollBar::handle:horizontal:hover { background-color: #5a7fd6; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
"""