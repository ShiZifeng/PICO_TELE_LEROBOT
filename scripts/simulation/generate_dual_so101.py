#!/usr/bin/env python3
"""Generate dual-arm SO-101 URDF and MJCF files from single-arm versions.

Usage:
    python generate_dual_so101.py
"""

import re
import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "so101")
SRC_URDF = os.path.join(ASSETS_DIR, "so101_new_calib.urdf")
SRC_MJCF = os.path.join(ASSETS_DIR, "so101_new_calib.xml")

LEFT_Y = 0.15   # left arm base offset in Y
RIGHT_Y = -0.15  # right arm base offset in Y

# ============================================================
# URDF Generation
# ============================================================

def generate_dual_urdf():
    with open(SRC_URDF, 'r') as f:
        content = f.read()

    # Extract the inner robot content (everything after <robot> and before </robot>)
    # We need links, joints, transmissions, materials
    robot_match = re.search(r'<robot[^>]*>(.*)</robot>', content, re.DOTALL)
    robot_body = robot_match.group(1)

    # Extract materials section
    materials_match = re.search(r'(<!-- Materials -->.*?)(<!-- Link base -->)', robot_body, re.DOTALL)
    materials_section = materials_match.group(1) if materials_match else ""

    # Extract everything from "<!-- Link base -->" to end of robot body
    rest = robot_body[robot_body.find("<!-- Link base -->"):]

    # For each arm, we rename links and joints with a prefix
    # Link names to rename: base_link, shoulder_link, upper_arm_link, lower_arm_link,
    #   wrist_link, gripper_link, gripper_frame_link, moving_jaw_so101_v1_link
    # Joint names to rename: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
    #   wrist_roll, gripper, gripper_frame_joint

    def prefix_arm(text, prefix, y_offset):
        """Add prefix to all link/joint names and offset the base position."""
        t = text

        # Prefix link names in <link name="..."> and parent/child references
        # Order matters: replace longer names first to avoid partial matches
        link_names = [
            "moving_jaw_so101_v1_link",
            "gripper_frame_link",
            "shoulder_link",
            "upper_arm_link",
            "lower_arm_link",
            "gripper_link",
            "wrist_link",
            "base_link",
        ]
        for name in link_names:
            t = t.replace(f'"{name}"', f'"{prefix}{name}"')

        # Prefix joint names (in joint name, parent link, child link, transmission references)
        joint_names = [
            "gripper_frame_joint",
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        for name in joint_names:
            # Joint definitions: <joint name="...">
            t = t.replace(f'name="{name}"', f'name="{prefix}{name}"')
            # Joint references: <joint name="..."> in transmissions
            # Already handled by the above since XML uses name="..."

        # Prefix actuator names (motor1, motor2, ...)
        for i in range(1, 7):
            t = t.replace(f'name="motor{i}"', f'name="{prefix}motor{i}"')

        # No base offset needed in URDF for individual arms - the fixed joint
        # from world link handles positioning

        return t

    left_arm = prefix_arm(rest, "left_", LEFT_Y)
    right_arm = prefix_arm(rest, "right_", RIGHT_Y)

    dual_urdf = f'''<?xml version="1.0" encoding="utf-8"?>
<robot name="dual_so101">

  <!-- World Link (common root) -->
  <link name="world"/>

  <!-- ==================== LEFT ARM ==================== -->
  <joint name="world_to_left_base" type="fixed">
    <parent link="world"/>
    <child link="left_base_link"/>
    <origin xyz="0 {LEFT_Y} 0" rpy="0 0 0"/>
  </joint>
{left_arm}
  <!-- ==================== RIGHT ARM ==================== -->
  <joint name="world_to_right_base" type="fixed">
    <parent link="world"/>
    <child link="right_base_link"/>
    <origin xyz="0 {RIGHT_Y} 0" rpy="0 0 0"/>
  </joint>
{right_arm}
</robot>'''

    out_path = os.path.join(ASSETS_DIR, "dual_so101.urdf")
    with open(out_path, 'w') as f:
        f.write(dual_urdf)
    print(f"Generated: {out_path}")


# ============================================================
# MJCF Generation
# ============================================================

def generate_dual_mjcf():
    with open(SRC_MJCF, 'r') as f:
        content = f.read()

    # Extract sections
    # <default> ... </default> (two default blocks)
    # <worldbody> ... </worldbody>
    # <asset> ... </asset>
    # <actuator> ... </actuator>

    # Find worldbody content
    wb_match = re.search(r'<worldbody>(.*?)</worldbody>', content, re.DOTALL)
    worldbody_content = wb_match.group(1)

    # Find actuator content
    act_match = re.search(r'<actuator>(.*?)</actuator>', content, re.DOTALL)
    actuator_content = act_match.group(1)

    # Find asset content
    asset_match = re.search(r'<asset>(.*?)</asset>', content, re.DOTALL)
    asset_content = asset_match.group(1)

    # Find default blocks
    default_blocks = re.findall(r'<default>.*?</default>', content, re.DOTALL)

    def prefix_mjcf_arm(text, prefix, y_offset):
        """Add prefix to all body/joint/geom/site names in a targeted way.

        Only replaces in name="X" and joint="X" attribute values, never
        in mesh file references or material definitions.
        """
        t = text

        # All MJCF body names (order: longest first to avoid partial match issues)
        body_names = [
            "moving_jaw_so101_v1",
            "upper_arm",
            "lower_arm",
            "shoulder",
            "gripper",
            "wrist",
            "base",
        ]

        # All MJCF joint names
        joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]

        # All MJCF site names
        site_names = [
            "gripperframe",
            "baseframe",
        ]

        # 1. Replace name="X" attributes (bodies, joints, geoms, sites)
        all_names = body_names + joint_names + site_names
        for name in all_names:
            t = t.replace(f'name="{name}"', f'name="{prefix}{name}"')

        # 2. Replace joint="X" references (in actuators)
        for name in joint_names:
            t = t.replace(f'joint="{name}"', f'joint="{prefix}{name}"')

        # 3. Replace childclass="base" body reference
        t = t.replace(f'childclass="{prefix}so101_new_calib"', 'childclass="so101_new_calib"')
        # Fix the childclass - the original says childclass="so101_new_calib", we don't prefix it

        # 4. Add y offset to the root body position
        t = t.replace(
            f'<body name="{prefix}base" pos="0 0 0"',
            f'<body name="{prefix}base" pos="0 {y_offset} 0"',
        )

        return t

    left_wb = prefix_mjcf_arm(worldbody_content, "left_", LEFT_Y)
    right_wb = prefix_mjcf_arm(worldbody_content, "right_", RIGHT_Y)

    # Prefix actuator joint references
    left_act = actuator_content
    right_act = actuator_content
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    for name in joint_names:
        left_act = left_act.replace(f'name="{name}"', f'name="left_{name}"')
        left_act = left_act.replace(f'joint="{name}"', f'joint="left_{name}"')
        right_act = right_act.replace(f'name="{name}"', f'name="right_{name}"')
        right_act = right_act.replace(f'joint="{name}"', f'joint="right_{name}"')

    # Create material colors for left (yellow) and right (blue) arms
    # We add material overrides in the default section

    dual_mjcf = f'''<?xml version="1.0" ?>
<mujoco model="dual_so101">
  <compiler angle="radian" meshdir="assets" autolimits="true"/>

  <!-- Shared defaults (from single-arm model) -->
  <default>
    <default class="so101_new_calib">
      <joint damping="1" frictionloss="0.1" armature="0.005"/>
      <position kp="50"/>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom group="3"/>
      </default>
    </default>
  </default>

  <default>
    <default class="sts3215">
      <geom contype="0" conaffinity="0"/>
      <joint damping="0.60" frictionloss="0.052" armature="0.028"/>
      <position kp="998.22" kv="2.731" forcerange="-2.94 2.94"/>
    </default>
    <default class="backlash">
      <joint damping="0.01" frictionloss="0" armature="0.01" limited="true" range="-0.008726646259971648 0.008726646259971648"/>
    </default>
  </default>

  <worldbody>
    <!-- ==================== LEFT ARM ==================== -->
{left_wb}
    <!-- ==================== RIGHT ARM ==================== -->
{right_wb}
  </worldbody>

  <asset>
{asset_content}
    <!-- Additional materials for left/right arm color differentiation -->
    <material name="left_arm_material" rgba="0.9 0.75 0.1 1.0"/>
    <material name="right_arm_material" rgba="0.2 0.5 0.9 1.0"/>
  </asset>

  <actuator>
    <!-- Left arm actuators -->
{left_act}
    <!-- Right arm actuators -->
{right_act}
  </actuator>
</mujoco>'''

    out_path = os.path.join(ASSETS_DIR, "dual_so101.xml")
    with open(out_path, 'w') as f:
        f.write(dual_mjcf)
    print(f"Generated: {out_path}")


if __name__ == "__main__":
    generate_dual_urdf()
    generate_dual_mjcf()
    print("Done!")
