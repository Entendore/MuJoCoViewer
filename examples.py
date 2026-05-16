"""Built-in example MuJoCo XML scenes."""

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