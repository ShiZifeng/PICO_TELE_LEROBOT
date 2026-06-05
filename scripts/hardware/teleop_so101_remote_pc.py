#!/usr/bin/env python3
"""
PC-side XR + IK controller for remote SO-101 teleop and LeRobot recording.

Receives XR poses from PICO headset, solves Placo IK, sends joint targets to
Jetson via ZMQ, receives observations back, and writes LeRobot Datasets.

Usage (on PC):
  # Single arm
  python teleop_so101_remote_pc.py \
    --listen-ip 0.0.0.0 \
    --repo-id local/so101_xr_teleop \
    --mode single

  # Dual arm
  python teleop_so101_remote_pc.py \
    --listen-ip 0.0.0.0 \
    --repo-id local/so101_dual_xr_teleop \
    --mode dual

Controls:
  Space: start/stop recording
  R:     discard current episode
  Q/Esc: quit
"""

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.mjpeg_streamer import MJPEGStreamServer
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
    apply_delta_pose,
    quat_diff_as_angle_axis,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("so101_pc")

SO101_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SO101_GRIPPER_CLOSE_RAD = -0.174533
SO101_GRIPPER_FULL_OPEN_RAD = 1.74533
SO101_GRIPPER_OPEN_RAD = SO101_GRIPPER_CLOSE_RAD + 0.5 * (SO101_GRIPPER_FULL_OPEN_RAD - SO101_GRIPPER_CLOSE_RAD)

JOINT_LIMITS_RAD = {
    "shoulder_pan": (-1.91986, 1.91986), "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69), "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121), "gripper": (-0.174533, 1.74533),
}

XR_FRAME_ROTATIONS = {
    # Exact alignment used by the MuJoCo/Placo simulation controllers.
    "simulation": R_HEADSET_TO_WORLD,
    # OpenXR convention: +X right, +Y up, -Z forward.
    # Robot world: +X forward, +Y left, +Z up.
    "openxr": np.array(
        [
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0],
        ]
    ),
    # Unity/PICO-style convention often seen through app bridges:
    # +X right, +Y up, +Z forward.
    "unity": np.array(
        [
            [0, 0, 1],
            [-1, 0, 0],
            [0, 1, 0],
        ]
    ),
    # PICO XRoboToolkit bridge as observed on hardware.
    "pico": R_HEADSET_TO_WORLD,
    "pico_y_flip": np.array(
        [
            [0, 0, -1],
            [1, 0, 0],
            [0, 1, 0],
        ]
    ),
}


def _find_lerobot_src() -> Optional[str]:
    env_path = os.environ.get("LEROBOT_PATH")
    project_root = Path(__file__).resolve().parents[2]
    bundled_path = project_root / "third_party" / "lerobot-v0.3.3"
    candidates = [
        os.path.join(env_path, "src") if env_path else None,
        env_path,
        str(bundled_path / "src"),
        "/media/shizifeng/projects21/lerobot-v0.3.3/src",
        os.path.expanduser("~/szf_lerobot/lerobot-v0.3.3/src"),
    ]
    for p in candidates:
        if p and os.path.isdir(os.path.join(p, "lerobot")):
            return p
    return None


def _ensure_lerobot_deps():
    """Import LeRobot dataset writer."""
    p = _find_lerobot_src()
    if p and p not in sys.path:
        sys.path.insert(0, p)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import DEFAULT_FEATURES, build_dataset_frame, hw_to_dataset_features
    return LeRobotDataset, DEFAULT_FEATURES, build_dataset_frame, hw_to_dataset_features


def build_manipulator_config(mode: str, control_mode: str = "pose") -> dict:
    """Build XRoboToolkit manipulator_config for single/dual arm."""
    if mode == "dual":
        return {
            "left_hand": {
                "link_name": "left_gripper_frame_link",
                "pose_source": "left_controller",
                "control_trigger": "left_grip",
                "control_mode": control_mode,
                "gripper_config": {"type": "parallel", "gripper_trigger": "left_trigger",
                                   "joint_names": ["left_gripper"], "open_pos": [SO101_GRIPPER_OPEN_RAD], "close_pos": [SO101_GRIPPER_CLOSE_RAD]},
            },
            "right_hand": {
                "link_name": "right_gripper_frame_link",
                "pose_source": "right_controller",
                "control_trigger": "right_grip",
                "control_mode": control_mode,
                "gripper_config": {"type": "parallel", "gripper_trigger": "right_trigger",
                                   "joint_names": ["right_gripper"], "open_pos": [SO101_GRIPPER_OPEN_RAD], "close_pos": [SO101_GRIPPER_CLOSE_RAD]},
            },
        }
    else:
        return {
            "right_hand": {
                "link_name": "gripper_frame_link",
                "pose_source": "right_controller",
                "control_trigger": "right_grip",
                "control_mode": control_mode,
                "gripper_config": {"type": "parallel", "gripper_trigger": "right_trigger",
                                   "joint_names": ["gripper"], "open_pos": [SO101_GRIPPER_OPEN_RAD], "close_pos": [SO101_GRIPPER_CLOSE_RAD]},
            },
        }


class XRIKController:
    """
    XR + Placo IK controller that produces joint targets for SO-101.

    Replaces the leader arm in LeRobot's pipeline.
    """

    def __init__(
        self,
        mode: str = "single",
        scale_factor: float = 1.0,
        control_rate_hz: int = 50,
        max_action_delta_deg: float = 30.0,
        max_target_step_deg: float = 3.0,
        target_filter_alpha: float = 0.35,
        xr_frame: str = "openxr",
        ik_control_mode: str = "pose",
        require_xr_confirm: bool = True,
        xr_confirm_button: str = "A",
        debug_xr_delta: bool = False,
    ):
        import placo
        from meshcat import transformations as tf

        self.mode = mode
        self.scale_factor = scale_factor
        self.dt = 1.0 / control_rate_hz
        self.max_action_delta_deg = max_action_delta_deg
        self.max_target_step_deg = max_target_step_deg
        self.target_filter_alpha = target_filter_alpha
        self.require_xr_confirm = require_xr_confirm
        self.xr_confirm_button = xr_confirm_button
        self.xr_confirmed = not require_xr_confirm
        self._confirm_button_was_down = False
        self._waiting_confirm_logged = False
        self.R_xr_confirm = np.eye(3)
        self.debug_xr_delta = debug_xr_delta
        self._last_xr_debug_t = 0.0

        urdf_path = os.path.join(ASSET_PATH, "so101", "dual_so101.urdf" if mode == "dual" else "so101_new_calib.urdf")
        self.manipulator_config = build_manipulator_config(mode, control_mode=ik_control_mode)

        self.xr_client = XrClient()
        self.R_headset_world = XR_FRAME_ROTATIONS.get(xr_frame, R_HEADSET_TO_WORLD)
        self.placo_robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.placo_robot)
        self.solver.dt = self.dt
        self.solver.mask_fbase(True)
        self.placo_robot.state.q[:7] = np.array([0, 0, 0, 0, 0, 0, 1])
        self.placo_robot.update_kinematics()

        self.effector_task = {}
        self.effector_control_mode = {}
        self.active = {}
        self.ref_ee_xyz = {}
        self.ref_ee_quat = {}
        self.ref_controller_xyz = {}
        self.ref_controller_quat = {}
        self.gripper_pos_target = {}

        for name, config in self.manipulator_config.items():
            link_name = config["link_name"]
            control_mode = config.get("control_mode", "pose")
            self.effector_control_mode[name] = control_mode
            T = self.placo_robot.get_T_world_frame(link_name)
            ee_xyz = T[:3, 3].copy()
            ee_quat = tf.quaternion_from_matrix(T)
            if control_mode == "position":
                self.effector_task[name] = self.solver.add_position_task(link_name, ee_xyz)
            else:
                ee_target = tf.quaternion_matrix(ee_quat)
                ee_target[:3, 3] = ee_xyz
                self.effector_task[name] = self.solver.add_frame_task(link_name, ee_target)
            self.effector_task[name].configure(name, "soft", 1.0)
            m = self.solver.add_manipulability_task(link_name, "both", 1.0)
            m.configure("manipulability", "soft", 1e-2)

            if "gripper_config" in config:
                gc = config["gripper_config"]
                self.gripper_pos_target[name] = dict(zip(gc["joint_names"], gc["open_pos"]))

        # Joint regularization (very soft — don't fight tracking)
        self.joints_task = self.solver.add_joints_task()
        self.joints_task.set_joints({j: 0.0 for j in self.placo_robot.joint_names() if "gripper" not in j})
        self.joints_task.configure("joints_reg", "soft", 1e-6)

        self.sides = ["left", "right"] if mode == "dual" else ["right"]
        self._has_robot_state = False
        self._waiting_for_robot_state_logged = False
        self._task_targets_synced_to_robot = False
        self._latest_observed_motor_deg: Dict[str, float] = {}
        self._last_published_target_deg: Dict[str, float] = {}
        self._filtered_target_deg: Dict[str, float] = {}
        self._last_raw_target_deg: Dict[str, float] = {}
        self._last_filtered_target_deg: Dict[str, float] = {}
        self._last_output_target_deg: Dict[str, float] = {}
        self._safety_paused = False
        self._last_invalid_xr_log_t = 0.0
        self._ik_failure_count = 0
        self._last_ik_failure_log_t = 0.0
        logger.info(
            "Placo IK ready. XR frame=%s, transform=%s, det=%.1f",
            xr_frame,
            self.R_headset_world.tolist(),
            float(np.linalg.det(self.R_headset_world)),
        )
        logger.info("Placo IK ready. control_mode=%s, Joints: %s", ik_control_mode, list(self.placo_robot.joint_names()))
        if self.require_xr_confirm:
            logger.info("Waiting for XR alignment confirmation. Face robot +X/front and press controller button '%s'.", self.xr_confirm_button)

    def _q_index(self, joint_name: str) -> Optional[int]:
        """Return the Pinocchio q-vector index for a 1-DoF joint."""
        if joint_name not in self.placo_robot.model.names:
            return None
        jid = self.placo_robot.model.getJointId(joint_name)
        joint = self.placo_robot.model.joints[jid]
        if getattr(joint, "nq", 0) != 1:
            return None
        return int(joint.idx_q)

    def update_robot_state(self, obs_frame: dict):
        """Update Placo state from real robot observation. Called from main loop."""
        for side in self.sides:
            prefix = f"{side}_" if self.mode == "dual" else ""
            for joint in SO101_JOINT_NAMES:
                obs_key = f"observation.{side}_{joint}.pos"
                value = obs_frame.get(obs_key)
                if value is None:
                    value = obs_frame.get(f"{side}_{joint}.pos")
                if value is not None:
                    value = float(value)
                    if not np.isfinite(value):
                        continue
                    placo_name = f"{prefix}{joint}"
                    motor_name = f"{side}_{joint}"
                    self._latest_observed_motor_deg[motor_name] = value
                    q_idx = self._q_index(placo_name)
                    if q_idx is not None:
                        self.placo_robot.state.q[q_idx] = float(np.radians(value))
        self._has_robot_state = True

    def step(self) -> Dict[str, float]:
        """Run one IK step (matches BaseTeleopController._update_ik)."""
        if not self._has_robot_state:
            if not self._waiting_for_robot_state_logged:
                logger.info("Waiting for first robot observation before publishing IK targets.")
                self._waiting_for_robot_state_logged = True
            return {}

        if self._safety_paused:
            if self._all_controls_released():
                self._reset_xr_references()
                self._safety_paused = False
                logger.warning("Safety pause cleared. Re-grip to resume teleop.")
            else:
                return {}

        # 1. Update kinematics from current robot state (set by update_robot_state)
        self.placo_robot.update_kinematics()
        if not self._task_targets_synced_to_robot:
            self._hold_all_effectors_at_current_pose()
            self._task_targets_synced_to_robot = True
            logger.info("Synced IK task targets to first robot observation.")
            return {}

        self._update_xr_confirmation()
        if not self.xr_confirmed:
            if not self._waiting_confirm_logged:
                logger.info("XR alignment not confirmed yet. Press controller button '%s' before using grip.", self.xr_confirm_button)
                self._waiting_confirm_logged = True
            self._reset_xr_references()
            self._hold_all_effectors_at_current_pose()
            return self._hold_command()

        # 2. Process XR input and update frame task targets
        any_active = False
        for name, config in self.manipulator_config.items():
            xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            self.active[name] = xr_grip_val > 0.9
            any_active = any_active or self.active[name]

            if self.active[name]:
                if self.ref_ee_xyz.get(name) is None:
                    link = config["link_name"]
                    T = self.placo_robot.get_T_world_frame(link)
                    self.ref_ee_xyz[name] = T[:3, 3].copy()
                    self.ref_ee_quat[name] = self._mat_to_quat(T)
                    logger.info("[%s] Activated.", name)

                xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
                if not self._is_valid_xr_pose(xr_pose):
                    self._log_invalid_xr_pose(name, xr_pose)
                    continue
                delta_xyz, delta_rot = self._process_xr(xr_pose, name)
                if self.effector_control_mode[name] == "position":
                    target_xyz = self.ref_ee_xyz[name] + delta_xyz
                    if not np.all(np.isfinite(target_xyz)):
                        self._log_invalid_xr_pose(name, xr_pose)
                        continue
                    self.effector_task[name].target_world = target_xyz
                else:
                    target_xyz, target_quat = apply_delta_pose(
                        self.ref_ee_xyz[name], self.ref_ee_quat[name], delta_xyz, delta_rot)
                    target_pose = self._quat_to_mat(target_quat)
                    target_pose[:3, 3] = target_xyz
                    if not np.all(np.isfinite(target_pose)):
                        self._log_invalid_xr_pose(name, xr_pose)
                        continue
                    self.effector_task[name].T_world_frame = target_pose
            else:
                if self.ref_ee_xyz.get(name) is not None:
                    logger.info("[%s] Deactivated.", name)
                    self.ref_ee_xyz[name] = None
                    self.ref_ee_quat[name] = None
                    self.ref_controller_xyz[name] = None
                    self.ref_controller_quat[name] = None
                self._hold_effector_at_current_pose(name)

        # 3. Gripper
        for name, config in self.manipulator_config.items():
            if "gripper_config" not in config:
                continue
            gc = config["gripper_config"]
            trigger = self.xr_client.get_key_value_by_name(gc["gripper_trigger"])
            for jname, open_p, close_p in zip(gc["joint_names"], gc["open_pos"], gc["close_pos"]):
                self.gripper_pos_target[name][jname] = open_p + (close_p - open_p) * trigger

        if not any_active:
            self._ik_failure_count = 0
            self._clear_arm_target_cache()
            targets = self._gripper_to_motor_dict()
            self._last_raw_target_deg = dict(targets)
            filtered = self._filter_target(targets)
            clipped = self._clip_target_step(filtered)
            self._last_output_target_deg = dict(clipped)
            return clipped

        # 4. Solve (always, like base controller does)
        q_before_solve = self.placo_robot.state.q.copy()
        try:
            self.solver.solve(True)
        except RuntimeError as e:
            self._handle_ik_failure(q_before_solve, f"RuntimeError: {e}")
            return {}
        except Exception as e:
            self._handle_ik_failure(q_before_solve, f"{type(e).__name__}: {e}")
            return {}

        if not np.all(np.isfinite(self.placo_robot.state.q)):
            self._handle_ik_failure(q_before_solve, "solver produced non-finite q")
            return {}

        self._restore_inactive_arm_joints(q_before_solve)
        self._ik_failure_count = 0
        targets = self._ik_to_motor_dict()
        self._last_raw_target_deg = dict(targets)
        if self._target_is_too_far(targets):
            return {}
        filtered = self._filter_target(targets)
        clipped = self._clip_target_step(filtered)
        self._last_output_target_deg = dict(clipped)
        return clipped

    def _handle_ik_failure(self, q_before_solve: np.ndarray, reason: str):
        self._ik_failure_count += 1
        self.placo_robot.state.q = q_before_solve
        self.placo_robot.update_kinematics()
        self._hold_all_effectors_at_current_pose()

        now = time.perf_counter()
        if self._ik_failure_count <= 3 or now - self._last_ik_failure_log_t > 1.0:
            self._last_ik_failure_log_t = now
            logger.warning("IK solve failed (#%d): %s", self._ik_failure_count, reason)

        if self._ik_failure_count >= 3:
            self._safety_paused = True
            self._reset_xr_references()
            logger.error("Safety pause: IK failed repeatedly. Release all grip buttons to reset.")

    def _target_is_too_far(self, targets: Dict[str, float]) -> bool:
        if not self.max_action_delta_deg or self.max_action_delta_deg <= 0:
            return False

        worst_name = None
        worst_delta = 0.0
        for name, target_deg in targets.items():
            if name.endswith("_gripper"):
                continue
            observed_deg = self._latest_observed_motor_deg.get(name)
            if observed_deg is None:
                continue
            delta = abs(float(target_deg) - float(observed_deg))
            if delta > worst_delta:
                worst_name = name
                worst_delta = delta

        if worst_delta <= self.max_action_delta_deg:
            return False

        self._safety_paused = True
        self._reset_xr_references()
        self._hold_all_effectors_at_current_pose()
        logger.error(
            "Safety pause: target jump too large on %s (%.1f deg > %.1f deg). "
            "Release all grip buttons to reset.",
            worst_name,
            worst_delta,
            self.max_action_delta_deg,
        )
        return True

    def _filter_target(self, targets: Dict[str, float]) -> Dict[str, float]:
        """Low-pass filter IK joint targets before step clipping."""
        alpha = float(self.target_filter_alpha)
        if alpha <= 0.0 or alpha >= 1.0:
            self._filtered_target_deg = dict(targets)
            self._last_filtered_target_deg = dict(targets)
            return targets

        filtered = {}
        for name, target_deg in targets.items():
            target_deg = float(target_deg)
            if name.endswith("_gripper"):
                filtered[name] = target_deg
                continue

            previous_deg = self._filtered_target_deg.get(name)
            if previous_deg is None:
                previous_deg = self._last_published_target_deg.get(name)
            if previous_deg is None:
                previous_deg = self._latest_observed_motor_deg.get(name)
            if previous_deg is None:
                filtered[name] = target_deg
            else:
                filtered[name] = float(previous_deg + alpha * (target_deg - float(previous_deg)))

        self._filtered_target_deg = dict(filtered)
        self._last_filtered_target_deg = dict(filtered)
        return filtered

    def _clip_target_step(self, targets: Dict[str, float]) -> Dict[str, float]:
        """Limit per-IK-tick joint target movement before publishing."""
        if not self.max_target_step_deg or self.max_target_step_deg <= 0:
            self._last_published_target_deg = dict(targets)
            return targets

        max_step = float(self.max_target_step_deg)
        clipped = {}
        for name, target_deg in targets.items():
            target_deg = float(target_deg)
            previous_deg = self._last_published_target_deg.get(name)
            if previous_deg is None:
                previous_deg = self._latest_observed_motor_deg.get(name)

            if previous_deg is None or name.endswith("_gripper"):
                clipped[name] = target_deg
                continue

            delta = target_deg - float(previous_deg)
            clipped[name] = float(previous_deg + np.clip(delta, -max_step, max_step))

        self._last_published_target_deg = dict(clipped)
        return clipped

    def _all_controls_released(self) -> bool:
        for config in self.manipulator_config.values():
            if self.xr_client.get_key_value_by_name(config["control_trigger"]) > 0.9:
                return False
        return True

    def _update_xr_confirmation(self):
        if not self.require_xr_confirm:
            return
        try:
            is_down = bool(self.xr_client.get_button_state_by_name(self.xr_confirm_button))
        except Exception as e:
            now = time.perf_counter()
            if now - self._last_invalid_xr_log_t > 1.0:
                self._last_invalid_xr_log_t = now
                logger.warning("Failed to read XR confirm button '%s': %s", self.xr_confirm_button, e)
            return
        if is_down and not self._confirm_button_was_down:
            self._confirm_xr_alignment()
        self._confirm_button_was_down = is_down

    def _confirm_xr_alignment(self):
        headset_pose = self.xr_client.get_pose_by_name("headset")
        if not self._is_valid_xr_pose(headset_pose):
            self._log_invalid_xr_pose("headset", headset_pose)
            return

        quat_wxyz = [headset_pose[6], headset_pose[3], headset_pose[4], headset_pose[5]]
        headset_R = self._transform_xr_quat_to_matrix(quat_wxyz, apply_confirmation=False)
        forward = headset_R @ np.array([1.0, 0.0, 0.0])
        forward[2] = 0.0
        norm = np.linalg.norm(forward)
        if not np.isfinite(norm) or norm < 1e-6:
            logger.warning("Cannot confirm XR alignment: headset forward vector is invalid.")
            return

        forward /= norm
        yaw = float(np.arctan2(forward[1], forward[0]))
        c = float(np.cos(-yaw))
        s = float(np.sin(-yaw))
        self.R_xr_confirm = np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.xr_confirmed = True
        self._waiting_confirm_logged = False
        self._reset_xr_references()
        self._hold_all_effectors_at_current_pose()
        logger.info("XR alignment confirmed with button '%s' (headset yaw %.1f deg).", self.xr_confirm_button, np.degrees(yaw))

    def _reset_xr_references(self):
        for name in self.manipulator_config:
            self.ref_ee_xyz[name] = None
            self.ref_ee_quat[name] = None
            self.ref_controller_xyz[name] = None
            self.ref_controller_quat[name] = None
            self.active[name] = False
        self._last_published_target_deg = {}
        self._filtered_target_deg = {}

    def _hold_all_effectors_at_current_pose(self):
        for name in self.manipulator_config:
            self._hold_effector_at_current_pose(name)

    def _hold_effector_at_current_pose(self, name: str):
        config = self.manipulator_config[name]
        T = self.placo_robot.get_T_world_frame(config["link_name"])
        if self.effector_control_mode.get(name) == "position":
            self.effector_task[name].target_world = T[:3, 3].copy()
        else:
            self.effector_task[name].T_world_frame = T.copy()

    def _restore_inactive_arm_joints(self, q_reference: np.ndarray):
        restored = False
        for name in self.manipulator_config:
            if self.active.get(name, False):
                continue
            side = name.removesuffix("_hand")
            prefix = f"{side}_" if self.mode == "dual" else ""
            for joint in SO101_JOINT_NAMES:
                if joint == "gripper":
                    continue
                q_idx = self._q_index(f"{prefix}{joint}")
                if q_idx is not None:
                    self.placo_robot.state.q[q_idx] = q_reference[q_idx]
                    restored = True
        if restored:
            self.placo_robot.update_kinematics()

    def _ik_to_motor_dict(self) -> Dict[str, float]:
        """Convert Placo IK output (radians) → {motor_name: degrees}."""
        targets = {}
        for side in self.sides:
            prefix = f"{side}_" if self.mode == "dual" else ""
            for joint in SO101_JOINT_NAMES:
                placo_name = f"{prefix}{joint}"
                motor_name = f"{side}_{joint}"
                try:
                    q_idx = self._q_index(placo_name)
                    rad = float(self.placo_robot.state.q[q_idx]) if q_idx is not None else 0.0
                except Exception:
                    rad = 0.0
                targets[motor_name] = float(np.degrees(rad))
        for hand_name, gripper_target in self.gripper_pos_target.items():
            side = hand_name.removesuffix("_hand")
            for placo_name, rad in gripper_target.items():
                motor_name = placo_name if self.mode == "dual" else f"{side}_{placo_name}"
                targets[motor_name] = float(np.degrees(float(rad)))
        return targets

    def _gripper_to_motor_dict(self) -> Dict[str, float]:
        """Convert gripper targets only, so triggers work without arm grip."""
        targets = {}
        for hand_name, gripper_target in self.gripper_pos_target.items():
            side = hand_name.removesuffix("_hand")
            for placo_name, rad in gripper_target.items():
                motor_name = placo_name if self.mode == "dual" else f"{side}_{placo_name}"
                targets[motor_name] = float(np.degrees(float(rad)))
        return targets

    @staticmethod
    def _hold_command() -> Dict[str, bool]:
        return {"__hold__": True}

    def _clear_arm_target_cache(self):
        self._last_published_target_deg = {
            name: value
            for name, value in self._last_published_target_deg.items()
            if name.endswith("_gripper")
        }
        self._filtered_target_deg = {
            name: value
            for name, value in self._filtered_target_deg.items()
            if name.endswith("_gripper")
        }

    def target_debug_snapshot(self) -> Dict[str, Any]:
        return {
            "raw_target_deg": dict(self._last_raw_target_deg),
            "filtered_target_deg": dict(self._last_filtered_target_deg),
            "published_target_deg": dict(self._last_output_target_deg),
            "observed_deg": dict(self._latest_observed_motor_deg),
            "active": dict(self.active),
            "xr_confirmed": bool(self.xr_confirmed),
            "safety_paused": bool(self._safety_paused),
        }

    def _process_xr(self, xr_pose, src_name):
        raw_xyz = np.array(xr_pose[:3])
        ctrl_xyz = raw_xyz.copy()
        ctrl_quat = [xr_pose[6], xr_pose[3], xr_pose[4], xr_pose[5]]
        ctrl_xyz = self.R_headset_world @ ctrl_xyz
        ctrl_xyz = self.R_xr_confirm @ ctrl_xyz
        ctrl_quat = self._transform_xr_quat(ctrl_quat)

        if self.ref_controller_xyz.get(src_name) is None:
            self.ref_controller_xyz[src_name] = ctrl_xyz
            self.ref_controller_quat[src_name] = ctrl_quat
            return np.zeros(3), np.zeros(3)

        raw_delta_xyz = raw_xyz - (self.R_headset_world.T @ self.R_xr_confirm.T @ self.ref_controller_xyz[src_name])
        delta_xyz = (ctrl_xyz - self.ref_controller_xyz[src_name]) * self.scale_factor
        delta_rot = quat_diff_as_angle_axis(self.ref_controller_quat[src_name], ctrl_quat)
        if self.debug_xr_delta:
            now = time.perf_counter()
            if now - self._last_xr_debug_t > 1.0:
                self._last_xr_debug_t = now
                logger.info(
                    "[%s] XR raw delta xyz=%s -> robot delta xyz=%s, delta rot axis-angle=%s",
                    src_name,
                    np.round(raw_delta_xyz, 3).tolist(),
                    np.round(delta_xyz, 3).tolist(),
                    np.round(delta_rot, 3).tolist(),
                )
        return delta_xyz, delta_rot

    def _is_valid_xr_pose(self, xr_pose) -> bool:
        pose = np.asarray(xr_pose, dtype=float)
        if pose.shape[0] < 7 or not np.all(np.isfinite(pose[:7])):
            return False
        quat_xyzw = pose[3:7]
        norm = np.linalg.norm(quat_xyzw)
        return 1e-6 < norm < 10.0

    def _log_invalid_xr_pose(self, name: str, xr_pose):
        now = time.perf_counter()
        if now - self._last_invalid_xr_log_t > 1.0:
            self._last_invalid_xr_log_t = now
            logger.warning("[%s] Invalid XR pose/quaternion, skipping this frame: %s", name, xr_pose)

    def _transform_xr_quat(self, quat_wxyz):
        """Transform XR orientation into robot world, including handedness flips."""
        from meshcat import transformations as tf

        robot_R = self._transform_xr_quat_to_matrix(quat_wxyz, apply_confirmation=True)
        robot_T = np.eye(4)
        robot_T[:3, :3] = robot_R
        return tf.quaternion_from_matrix(robot_T)

    def _transform_xr_quat_to_matrix(self, quat_wxyz, apply_confirmation: bool):
        from meshcat import transformations as tf

        quat_wxyz = np.asarray(quat_wxyz, dtype=float)
        quat_norm = np.linalg.norm(quat_wxyz)
        if not np.isfinite(quat_norm) or quat_norm <= 1e-6:
            return np.eye(3)
        quat_wxyz = quat_wxyz / quat_norm
        raw_T = tf.quaternion_matrix(quat_wxyz)
        robot_R = self.R_headset_world @ raw_T[:3, :3] @ self.R_headset_world.T
        if apply_confirmation:
            robot_R = self.R_xr_confirm @ robot_R
        if not np.all(np.isfinite(robot_R)):
            return np.eye(3)
        return robot_R

    @staticmethod
    def _mat_to_quat(T):
        from meshcat import transformations as tf
        return tf.quaternion_from_matrix(T)

    @staticmethod
    def _quat_to_mat(q):
        from meshcat import transformations as tf
        return tf.quaternion_matrix(q)


def decode_frame(frame: dict) -> dict:
    """Decode JPEG-encoded images in a LeRobot frame dict."""
    out = {}
    for k, v in frame.items():
        if isinstance(v, dict) and v.get("__remote_image_encoding__") == "jpg":
            img = cv2.imdecode(np.frombuffer(v["data"], dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                out[k] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            out[k] = v
    return out


def make_preview(frame: dict, image_keys: list, status: str, target_h: int = 240) -> Optional[np.ndarray]:
    """Build preview image from camera frames."""
    images = []
    for key in image_keys:
        img = frame.get(key)
        if img is None:
            continue
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]
        scale = target_h / h
        img_bgr = cv2.resize(img_bgr, (max(1, int(w * scale)), target_h))
        cv2.putText(img_bgr, key.split(".")[-1], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        images.append(img_bgr)
    if not images:
        return None
    preview = cv2.hconcat(images)
    bar = np.zeros((36, preview.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)
    return cv2.vconcat([preview, bar])


def get_image_keys(features: dict) -> list:
    return [k for k, ft in features.items() if ft.get("dtype") in ("image", "video")]


def main():
    import argparse
    import zmq

    parser = argparse.ArgumentParser("PC-side XR + IK + Recording for SO-101")
    parser.add_argument("--repo-id", default="local/so101_xr_teleop")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--listen-ip", default="0.0.0.0")
    parser.add_argument("--obs-port", type=int, default=5570, help="Port receiving observations from Jetson")
    parser.add_argument("--target-port", type=int, default=5580, help="Port sending joint targets to Jetson")
    parser.add_argument("--command-port", type=int, default=5571, help="Port sending commands to Jetson")
    parser.add_argument("--mode", default="single", choices=["single", "dual"])
    parser.add_argument("--scale-factor", type=float, default=1.0, help="XR motion scaling")
    parser.add_argument("--control-rate-hz", type=int, default=50, help="IK solver frequency")
    parser.add_argument(
        "--xr-frame",
        default="openxr",
        choices=sorted(XR_FRAME_ROTATIONS.keys()),
        help="XR tracking coordinate convention: openxr uses -Z forward, unity uses +Z forward",
    )
    parser.add_argument(
        "--ik-control-mode",
        default="pose",
        choices=["pose", "position"],
        help="Use full pose IK or position-only IK for debugging axis alignment",
    )
    parser.add_argument("--require-xr-confirm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--xr-confirm-button", default="A", help="XR controller button used to confirm headset-yaw alignment")
    parser.add_argument("--xr-record-button", default="X", help="XR controller button used to start/stop recording")
    parser.add_argument("--debug-xr-delta", action="store_true", help="Log raw and robot-frame controller deltas")
    parser.add_argument(
        "--max-action-delta-deg",
        type=float,
        default=30.0,
        help="Pause publishing if any non-gripper target differs from observation by more than this many degrees (0=disable)",
    )
    parser.add_argument(
        "--max-target-step-deg",
        type=float,
        default=3.0,
        help="Clip each non-gripper joint target change to this many degrees per IK tick before publishing (0=disable)",
    )
    parser.add_argument(
        "--target-filter-alpha",
        type=float,
        default=0.35,
        help="Low-pass filter alpha for non-gripper IK targets before clipping (0<alpha<1, smaller=smoother/slower, 1=disable)",
    )
    parser.add_argument(
        "--ik-target-log",
        type=Path,
        default=None,
        help="Path for IK target JSONL log (default: logs/ik_targets_<timestamp>.jsonl)",
    )
    parser.add_argument("--no-ik-target-log", action="store_true", help="Disable IK target JSONL logging")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--preview-height", type=int, default=240)
    parser.add_argument("--mjpeg-port", type=int, default=8080, help="MJPEG streaming port for XR (0=disable)")
    parser.add_argument("--mjpeg-quality", type=int, default=70, help="MJPEG JPEG quality (10-100)")
    args = parser.parse_args()

    LeRobotDataset, DEFAULT_FEATURES, build_dataset_frame, hw_to_dataset_features = _ensure_lerobot_deps()

    # ── ZMQ Setup ──
    ctx = zmq.Context()

    # Receive observations from Jetson (PULL — same protocol as LeRobot receiver)
    obs_socket = ctx.socket(zmq.PULL)
    obs_socket.setsockopt(zmq.RCVHWM, 1)
    obs_socket.bind(f"tcp://{args.listen_ip}:{args.obs_port}")

    # Send joint targets to Jetson (PUB)
    target_socket = ctx.socket(zmq.PUB)
    target_socket.setsockopt(zmq.SNDHWM, 1)
    target_socket.bind(f"tcp://{args.listen_ip}:{args.target_port}")

    # Send commands to Jetson (PUB)
    cmd_socket = ctx.socket(zmq.PUB)
    cmd_socket.setsockopt(zmq.SNDHWM, 1)
    cmd_socket.bind(f"tcp://{args.listen_ip}:{args.command_port}")

    logger.info("Listening for observations on tcp://%s:%d", args.listen_ip, args.obs_port)
    logger.info("Publishing joint targets on tcp://%s:%d", args.listen_ip, args.target_port)

    ik_target_log_fh = None
    if not args.no_ik_target_log:
        ik_target_log_path = args.ik_target_log
        if ik_target_log_path is None:
            ik_target_log_path = Path("logs") / f"ik_targets_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        ik_target_log_path.parent.mkdir(parents=True, exist_ok=True)
        ik_target_log_fh = open(ik_target_log_path, "a", buffering=1)
        logger.info("IK target log: %s", ik_target_log_path)

    # ── MJPEG streamer (for PICO headset video) ──
    mjpeg = None
    if args.mjpeg_port > 0:
        mjpeg = MJPEGStreamServer(port=args.mjpeg_port, jpeg_quality=args.mjpeg_quality)
        mjpeg.start()
        logger.info("MJPEG streaming: %s", mjpeg.url)
        logger.info("  PICO browser → %s", mjpeg.url)

    # ── XR + IK ──
    # Store controller in a function that re-validates on each access.
    # Workaround for C++ SDK memory corruption that overwrites Python objects.
    class _IKRef:
        def __init__(self, controller):
            self._ctrl = controller
        def get(self):
            if type(self._ctrl).__name__ == 'str':
                logger.error("BUG: ik ref corrupted (got str=%r), attempting recovery", self._ctrl)
                raise RuntimeError("IK controller corrupted — restart required")
            return self._ctrl

    _ik_ref = _IKRef(XRIKController(
        mode=args.mode,
        scale_factor=args.scale_factor,
        control_rate_hz=args.control_rate_hz,
        max_action_delta_deg=args.max_action_delta_deg,
        max_target_step_deg=args.max_target_step_deg,
        target_filter_alpha=args.target_filter_alpha,
        xr_frame=args.xr_frame,
        ik_control_mode=args.ik_control_mode,
        require_xr_confirm=args.require_xr_confirm,
        xr_confirm_button=args.xr_confirm_button,
        debug_xr_delta=args.debug_xr_delta,
    ))
    logger.info(
        "IK target smoothing: alpha=%.2f, max_step=%.2f deg/tick",
        args.target_filter_alpha,
        args.max_target_step_deg,
    )

    # ── Recording state ──
    dataset = None
    task = None
    image_keys = []
    recording = False
    episode_frame_count = 0
    saved_episodes = 0
    fps_display = 0.0
    last_fps_t = time.perf_counter()
    recv_frames = 0
    xr_record_button_was_down = False
    last_xr_record_button_error_t = 0.0

    # ── IK thread ──
    latest_target: Dict[str, float] = {}
    ik_stop = threading.Event()

    def ik_loop(ik_ref):
        nonlocal latest_target
        ik_error_count = 0
        ik_tick_count = 0
        ik_send_count = 0
        ik_late_count = 0
        ik_last_stats_t = time.perf_counter()
        ik_max_step_ms = 0.0
        ik_seq = 0
        period = 1.0 / args.control_rate_hz
        while not ik_stop.is_set():
            t0 = time.perf_counter()
            try:
                ctrl = ik_ref.get()
                target = ctrl.step()
                latest_target = target if target else {}
                ik_error_count = 0
            except Exception as e:
                latest_target = {}
                ik_error_count += 1
                if ik_error_count <= 1:
                    import traceback
                    logger.warning("IK error (#%d): %s", ik_error_count, e)
                    logger.warning("Full traceback:\n%s", traceback.format_exc())
                elif ik_error_count % 100 == 0:
                    logger.warning("IK error (#%d): %s (suppressed)", ik_error_count, e)
            # Empty target means startup wait, safety pause, or IK failure.
            if latest_target:
                try:
                    target_socket.send_pyobj(latest_target, flags=zmq.NOBLOCK)
                    ik_send_count += 1
                    if ik_target_log_fh is not None:
                        snapshot = ctrl.target_debug_snapshot()
                        ik_seq += 1
                        record = {
                            "seq": ik_seq,
                            "wall_time": time.time(),
                            "monotonic_time": time.perf_counter(),
                            "control_rate_hz": args.control_rate_hz,
                            "recording": bool(recording),
                            "episode_frame_count": int(episode_frame_count),
                            "target_sent_deg": dict(latest_target),
                            **snapshot,
                        }
                        ik_target_log_fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                except zmq.Again:
                    pass
                except Exception as e:
                    logger.warning("Failed to write IK target log: %s", e)
            elapsed = time.perf_counter() - t0
            ik_tick_count += 1
            ik_max_step_ms = max(ik_max_step_ms, elapsed * 1000.0)
            if elapsed > period:
                ik_late_count += 1
            now = time.perf_counter()
            if now - ik_last_stats_t >= 1.0:
                dt = now - ik_last_stats_t
                logger.info(
                    "IK fps: %.1f tick/s, %.1f publish/s, max_step=%.2fms, late=%d",
                    ik_tick_count / dt,
                    ik_send_count / dt,
                    ik_max_step_ms,
                    ik_late_count,
                )
                ik_tick_count = 0
                ik_send_count = 0
                ik_late_count = 0
                ik_max_step_ms = 0.0
                ik_last_stats_t = now
            time.sleep(max(0, period - elapsed))

    ik_thread = threading.Thread(target=ik_loop, args=(_ik_ref,), daemon=True)
    ik_thread.start()

    # ── Keyboard listener ──
    keyboard_events = {"toggle_recording": False, "discard_episode": False, "stop": False}

    def on_press(key):
        try:
            from pynput import keyboard as kb
            if key == kb.Key.space:
                keyboard_events["toggle_recording"] = True
            elif hasattr(key, 'char') and key.char == 'r':
                keyboard_events["discard_episode"] = True
            elif key == kb.Key.esc or (hasattr(key, 'char') and key.char == 'q'):
                keyboard_events["stop"] = True
        except Exception:
            pass

    try:
        from pynput import keyboard as kb
        listener = kb.Listener(on_press=on_press)
        listener.start()
    except Exception:
        logger.warning("pynput not available; keyboard control disabled.")
        listener = None

    logger.info("Controls: XR %s or Space=start/stop recording, R=discard, Q/Esc=quit", args.xr_record_button)

    # ── Main loop: receive observations ──
    try:
        while True:
            msg = obs_socket.recv_pyobj()
            if dataset is not None and not recording:
                try:
                    while True:
                        msg = obs_socket.recv_pyobj(flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
            msg_type = msg.get("type")

            # Log unexpected message types
            if msg_type not in ("setup", "frame", "episode_end", "done"):
                logger.warning("Unknown message type: %s, keys=%s", msg_type, list(msg.keys())[:5])

            if msg_type == "setup":
                logger.info("Received setup: fps=%d, robot_type=%s, features=%d",
                            msg.get("fps"), msg.get("robot_type"), len(msg.get("features", {})))
                task = msg.get("task", "XR teleop")
                features = msg["features"]
                features.update(DEFAULT_FEATURES)
                image_keys = get_image_keys(features)
                logger.info("Image keys: %s", image_keys)

                root = args.root or Path.cwd() / "datasets" / args.repo_id.replace("/", "_")
                try:
                    if root.exists():
                        if (root / "meta/info.json").is_file():
                            # Resume: load existing dataset, continue adding episodes
                            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
                            from lerobot.datasets.compute_stats import aggregate_stats
                            from lerobot.datasets.utils import load_episodes, load_episodes_stats, load_info, load_tasks, EPISODES_PATH, EPISODES_STATS_PATH, TASKS_PATH
                            meta = LeRobotDatasetMetadata.__new__(LeRobotDatasetMetadata)
                            meta.repo_id = args.repo_id
                            meta.root = root
                            meta.revision = None
                            meta.info = load_info(root)
                            meta.tasks, meta.task_to_task_index = load_tasks(root) if (root / TASKS_PATH).is_file() else ({}, {})
                            meta.episodes = load_episodes(root) if (root / EPISODES_PATH).is_file() else {}
                            meta.episodes_stats = load_episodes_stats(root) if (root / EPISODES_STATS_PATH).is_file() else {}
                            meta.stats = aggregate_stats(list(meta.episodes_stats.values())) if meta.episodes_stats else {}
                            dataset = LeRobotDataset.__new__(LeRobotDataset)
                            dataset.meta = meta
                            dataset.repo_id = meta.repo_id
                            dataset.root = meta.root
                            dataset.revision = None
                            dataset.tolerance_s = 1e-4
                            dataset.image_writer = None
                            dataset.batch_encoding_size = 1
                            dataset.episodes_since_last_encoding = 0
                            dataset.episodes = None
                            dataset.hf_dataset = dataset.create_hf_dataset()
                            dataset.image_transforms = None
                            dataset.delta_timestamps = None
                            dataset.delta_indices = None
                            dataset.episode_data_index = None
                            dataset.video_backend = None
                            dataset.episode_buffer = dataset.create_episode_buffer()
                            logger.info("Resumed dataset: %s (%d episodes)", dataset.root, dataset.meta.episodes.get("total", 0))
                        else:
                            # Partial/invalid dataset directory, remove and recreate
                            import shutil
                            shutil.rmtree(root)
                            dataset = LeRobotDataset.create(
                                repo_id=args.repo_id, fps=msg["fps"], root=str(root),
                                robot_type=msg.get("robot_type", "so101_follower"),
                                features=features, use_videos=True,
                            )
                            logger.info("Dataset created (after cleanup): %s", dataset.root)
                    else:
                        dataset = LeRobotDataset.create(
                            repo_id=args.repo_id, fps=msg["fps"], root=str(root),
                            robot_type=msg.get("robot_type", "so101_follower"),
                            features=features, use_videos=True,
                        )
                        logger.info("Dataset created: %s (fps=%d)", dataset.root, msg["fps"])
                except Exception as e:
                    logger.error("Failed to create/resume dataset: %s", e, exc_info=True)
                    raise

            elif msg_type == "frame":
                if dataset is None:
                    logger.warning("Frame before setup, skipping")
                    continue

                # Push raw JPEG bytes to MJPEG streamer (before decode, zero-cost)
                if mjpeg is not None and image_keys:
                    raw_frame = msg["frame"]
                    for ik in image_keys:
                        img_data = raw_frame.get(ik)
                        if isinstance(img_data, dict) and img_data.get("__remote_image_encoding__") == "jpg":
                            short_name = ik.replace("observation.images.", "")
                            mjpeg.update_raw(short_name, img_data["data"])

                frame = decode_frame(msg["frame"])
                recv_frames += 1

                try:
                    xr_record_button_down = bool(_ik_ref.get().xr_client.get_button_state_by_name(args.xr_record_button))
                    if xr_record_button_down and not xr_record_button_was_down:
                        keyboard_events["toggle_recording"] = True
                        logger.info("XR button '%s' pressed: toggle recording.", args.xr_record_button)
                    xr_record_button_was_down = xr_record_button_down
                except Exception as e:
                    now = time.perf_counter()
                    if now - last_xr_record_button_error_t > 1.0:
                        last_xr_record_button_error_t = now
                        logger.warning("Failed to read XR record button '%s': %s", args.xr_record_button, e)

                # Feed real robot state to IK controller (matches _update_robot_state)
                try:
                    _ik_ref.get().update_robot_state(frame)
                except Exception:
                    pass
                now = time.perf_counter()
                if now - last_fps_t >= 1.0:
                    fps_display = recv_frames / (now - last_fps_t)
                    recv_frames = 0
                    last_fps_t = now

                if recording:
                    dataset.add_frame(frame, task=task)
                    episode_frame_count += 1

                if args.display:
                    try:
                        state = "REC" if recording else "PREVIEW"
                        status = (f"{state} | frames:{episode_frame_count} | saved:{saved_episodes} | "
                                  f"fps:{fps_display:.1f} | XR {args.xr_record_button}/space rec, r discard, q/Esc quit")
                        preview = make_preview(frame, image_keys, status, args.preview_height)
                        if preview is not None:
                            cv2.imshow("SO-101 XR Teleop", preview)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord(" "):
                            keyboard_events["toggle_recording"] = True
                        elif key == ord("r"):
                            keyboard_events["discard_episode"] = True
                        elif key in (ord("q"), 27):
                            keyboard_events["stop"] = True
                    except cv2.error:
                        logger.warning("OpenCV GUI not available, disabling display")
                        args.display = False

                if keyboard_events["toggle_recording"]:
                    keyboard_events["toggle_recording"] = False
                    if recording:
                        if episode_frame_count > 0:
                            dataset.save_episode()
                            saved_episodes += 1
                            logger.info("Saved episode %d", dataset.num_episodes - 1)
                        recording = False
                        episode_frame_count = 0
                    else:
                        dataset.clear_episode_buffer()
                        episode_frame_count = 0
                        recording = True
                        logger.info("Started episode %d", dataset.num_episodes)

                if keyboard_events["discard_episode"]:
                    keyboard_events["discard_episode"] = False
                    dataset.clear_episode_buffer()
                    episode_frame_count = 0
                    recording = False
                    logger.info("Discarded current episode.")

                if keyboard_events["stop"]:
                    if recording and episode_frame_count > 0:
                        dataset.save_episode()
                        saved_episodes += 1
                    cmd_socket.send_string("stop")
                    break

            elif msg_type == "done":
                if recording and episode_frame_count > 0:
                    dataset.save_episode()
                    saved_episodes += 1
                break

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        ik_stop.set()
        if ik_thread.is_alive():
            ik_thread.join(timeout=2)
        if listener:
            listener.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if mjpeg:
            mjpeg.stop()
        if ik_target_log_fh is not None:
            ik_target_log_fh.close()
        cmd_socket.send_string("stop")
        target_socket.close(0)
        obs_socket.close(0)
        cmd_socket.close(0)
        ctx.term()
        logger.info("Shutdown complete. Total saved episodes: %d", saved_episodes)


if __name__ == "__main__":
    main()
