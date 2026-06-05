import os

import mujoco
import numpy as np
import tyro
from meshcat import transformations as tf

from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import (
    MujocoTeleopController,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


class DualSO101MujocoTeleopController(MujocoTeleopController):
    """MujocoTeleopController for the dual-arm SO-101.

    Both the left and right arms use Placo FK for _get_link_pose on
    gripper_frame_link frames, since the MJCF site orientations don't
    match the URDF frame orientations (same issue as the single-arm version).
    """

    def _get_link_pose(self, ee_name):
        """Get end effector pose.

        For gripper_frame_link variants, uses Placo FK which is always
        consistent with the Placo frame task. Other links use MJCF body lookup.
        """
        # Try direct MJCF body name match first
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
        if body_id != -1:
            return self.mj_data.xpos[body_id].copy(), self.mj_data.xquat[body_id].copy()

        # URDF-only frames: use Placo FK for self-consistent orientation
        if ee_name in ("left_gripper_frame_link", "right_gripper_frame_link"):
            T = self.placo_robot.get_T_world_frame(ee_name)
            xyz = T[:3, 3].copy()
            quat = tf.quaternion_from_matrix(T)
            return xyz, quat

        raise ValueError(f"'{ee_name}' not found as body in MuJoCo model.")


def main(
    xml_path: str = os.path.join(ASSET_PATH, "so101/scene_dual_teleop.xml"),
    robot_urdf_path: str = os.path.join(ASSET_PATH, "so101/dual_so101.urdf"),
    scale_factor: float = 1.5,
    visualize_placo: bool = True,
):
    """
    Main function to run dual-arm SO-101 teleoperation in MuJoCo.

    Two SO-101 6-DOF tabletop robot arms placed 30 cm apart (left at y=+0.15m,
    right at y=-0.15m), each with a parallel jaw gripper.

    Controls:
      - Left grip (side button): activate/deactivate left arm tracking
      - Left trigger: close left gripper (0=open, 1=closed)
      - Left controller: move left end effector (pose tracking)
      - Right grip (side button): activate/deactivate right arm tracking
      - Right trigger: close right gripper (0=open, 1=closed)
      - Right controller: move right end effector (pose tracking)
    """
    config = {
        "left_hand": {
            "link_name": "left_gripper_frame_link",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
            "vis_target": "left_target",
            "gripper_config": {
                "type": "parallel",
                "gripper_trigger": "left_trigger",
                "joint_names": ["left_gripper"],
                "open_pos": [1.74533],  # fully open (~100°)
                "close_pos": [-0.174533],  # fully closed (~-10°)
            },
        },
        "right_hand": {
            "link_name": "right_gripper_frame_link",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "vis_target": "right_target",
            "gripper_config": {
                "type": "parallel",
                "gripper_trigger": "right_trigger",
                "joint_names": ["right_gripper"],
                "open_pos": [1.74533],
                "close_pos": [-0.174533],
            },
        },
    }

    # Create and initialize the teleoperation controller
    controller = DualSO101MujocoTeleopController(
        xml_path=xml_path,
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
        visualize_placo=visualize_placo,
        mj_qpos_init=np.zeros(12),  # 12 joints (6 per arm) at neutral
    )

    # Joint regularization: bias all actuated joints toward zero (neutral pose)
    # Exclude gripper joints (controlled by triggers, not IK)
    joints_task = controller.solver.add_joints_task()
    joints_task.set_joints(
        {
            joint: 0.0
            for joint in controller.placo_robot.joint_names()
            if "gripper" not in joint
        }
    )
    joints_task.configure("joints_regularization", "soft", 1e-3)

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
