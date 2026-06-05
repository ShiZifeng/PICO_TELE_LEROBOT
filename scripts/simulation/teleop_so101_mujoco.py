import os

import mujoco
import numpy as np
import tyro
from meshcat import transformations as tf

from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import (
    MujocoTeleopController,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


class SO101MujocoTeleopController(MujocoTeleopController):
    """MujocoTeleopController with URDF→MJCF body name mapping for SO-101.

    The SO-101 URDF and MJCF use different naming conventions and coordinate
    frame conventions for bodies/links. For MJCF bodies that exist with the
    same name, we look them up directly. For URDF-only frames (like
    gripper_frame_link), we use Placo's forward kinematics to get the correct
    world pose, since the MJCF site orientation does not match the URDF frame.
    """

    def _get_link_pose(self, ee_name):
        """Get end effector pose.

        Uses MJCF body lookup for bodies that exist in both URDF and MJCF.
        Falls back to Placo forward kinematics for URDF-only frames whose
        MJCF site orientation doesn't match.
        """
        # Try direct MJCF body name match first
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
        if body_id != -1:
            return self.mj_data.xpos[body_id].copy(), self.mj_data.xquat[body_id].copy()

        # gripper_frame_link only exists as a frame in the URDF (fixed to gripper_link).
        # The MJCF gripperframe site has a different orientation, so we use Placo FK
        # which is always consistent with the Placo frame task.
        if ee_name == "gripper_frame_link":
            T = self.placo_robot.get_T_world_frame("gripper_frame_link")
            xyz = T[:3, 3].copy()
            quat = tf.quaternion_from_matrix(T)
            return xyz, quat

        raise ValueError(f"'{ee_name}' not found as body in MuJoCo model.")


def main(
    xml_path: str = os.path.join(ASSET_PATH, "so101/scene_teleop.xml"),
    robot_urdf_path: str = os.path.join(ASSET_PATH, "so101/so101_new_calib.urdf"),
    scale_factor: float = 1.5,
    visualize_placo: bool = True,
):
    """
    Main function to run the SO-101 teleoperation in MuJoCo.

    The SO-101 is a 6-DOF tabletop robot arm with a parallel jaw gripper,
    driven by STS3215 servos. This script uses the "new calibration" URDF/MJCF
    where virtual zeros are at the middle of each joint range.

    Controls:
      - Right grip (side button): activate/deactivate arm tracking
      - Right trigger: close gripper (0=open, 1=closed)
      - Right controller: move end effector (pose tracking)
    """
    config = {
        "right_hand": {
            "link_name": "gripper_frame_link",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "vis_target": "right_target",
            "gripper_config": {
                "type": "parallel",
                "gripper_trigger": "right_trigger",
                "joint_names": ["gripper"],
                "open_pos": [1.74533],  # fully open (~100°)
                "close_pos": [-0.174533],  # fully closed (~-10°)
            },
        },
    }

    # Create and initialize the teleoperation controller
    controller = SO101MujocoTeleopController(
        xml_path=xml_path,
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
        visualize_placo=visualize_placo,
        mj_qpos_init=np.zeros(6),  # all 6 joints at neutral (mid-range)
    )

    # Joint regularization: bias actuated joints toward zero (neutral pose)
    # Exclude the gripper joint (controlled by trigger, not IK)
    joints_task = controller.solver.add_joints_task()
    joints_task.set_joints(
        {
            joint: 0.0
            for joint in controller.placo_robot.joint_names()
            if joint != "gripper"
        }
    )
    joints_task.configure("joints_regularization", "soft", 1e-3)

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
