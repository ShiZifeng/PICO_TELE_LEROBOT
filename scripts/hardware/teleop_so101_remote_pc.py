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
from datetime import datetime
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.h264_tcp_streamer import H264TCPStreamer
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
    "elbow_flex": (-1.50, 1.50), "wrist_flex": (-2.26893, 2.26893),
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
        control_rate_hz: int = 60,
        max_action_delta_deg: float = 60.0,
        max_target_step_deg: float = 3.0,
        target_filter_alpha: float = 0.05,
        wrist_pitch_scale: float = 1.0,
        wrist_roll_scale: float = -1.0,
        wrist_roll_speed_deg: float = 45.0,
        wrist_roll_stick_axis: str = "x",
        wrist_roll_stick_deadzone: float = 0.08,
        max_wrist_pitch_deg: float = 130.0,
        max_wrist_reach_m: float = 0.285,
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
        self.wrist_pitch_scale = wrist_pitch_scale
        self.wrist_roll_scale = wrist_roll_scale
        self.wrist_roll_speed_rad = np.radians(wrist_roll_speed_deg)
        self.wrist_roll_stick_axis = wrist_roll_stick_axis
        self.wrist_roll_stick_deadzone = wrist_roll_stick_deadzone
        self.max_wrist_pitch_rad = np.radians(max_wrist_pitch_deg)
        self.max_wrist_reach_m = max_wrist_reach_m
        self.require_xr_confirm = require_xr_confirm
        self.xr_confirm_button = xr_confirm_button
        self.xr_confirmed = not require_xr_confirm
        self._confirm_button_was_down = False
        self._waiting_confirm_logged = False
        self._last_confirm_debug_t = 0.0
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
        self.task_link_name = {}
        self.active = {}
        self.ref_ee_xyz = {}
        self.ref_ee_quat = {}
        self.ref_controller_xyz = {}
        self.ref_controller_quat = {}
        self._snapped_robot_q: Optional[np.ndarray] = None  # frozen joint state at grip moment
        self.ref_wrist_pitch_elevation_rad = {}
        self.ref_wrist_pitch_horizontal_world = {}
        self.desired_wrist_pitch_elevation_rad = {}
        self.last_wrist_pitch_elevation_rad = {}
        self.direct_wrist_roll_target_rad = {}
        self.wrist_xyz_target_world = {}
        self.gripper_pos_target = {}

        for name, config in self.manipulator_config.items():
            control_mode = config.get("control_mode", "pose")
            self.effector_control_mode[name] = control_mode
            link_name = self._task_link_name(name, config)
            self.task_link_name[name] = link_name
            T = self.placo_robot.get_T_world_frame(link_name)
            ee_xyz = T[:3, 3].copy()
            ee_quat = tf.quaternion_from_matrix(T)
            if control_mode in ("position", "position_wrist"):
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
        self._last_workspace_clip_log_t = 0.0

        # ── Homing state ──
        # Y or B button triggers smooth homing to a neutral resting pose.
        self._homing_active = False
        self._homing_start_time = 0.0
        self._homing_duration = 2.0  # seconds (smoothstep eased)
        self._homing_start_positions_deg: Dict[str, float] = {}
        self._homing_button_was_down = False
        self._homing_button_name = "Y"  # also checks B as alternative
        # Neutral home pose (motor degrees). Same for both arms.
        self._home_pose_deg = {
            "shoulder_pan": 0.0,
            "shoulder_lift": -15.0,
            "elbow_flex": 90.0,
            "wrist_flex": 45.0,
            "wrist_roll": 0.0,
            "gripper": 50.0,
        }

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
        """Update Placo state from real robot observation. Supports both grouped
        (observation.state array) and per-joint (observation.xxx.pos) formats."""
        state_arr = obs_frame.get("observation.state")
        if state_arr is not None:
            # LeRobot grouped format: observation.state is (12,) array
            all_joints = []
            for side in self.sides:
                for joint in SO101_JOINT_NAMES:
                    all_joints.append(f"{side}_{joint}")
            for joint_name, value in zip(all_joints, state_arr):
                value = float(value)
                if not np.isfinite(value):
                    continue
                side, joint = joint_name.split("_", 1)
                prefix = f"{side}_"
                placo_name = f"{prefix}{joint}"
                self._latest_observed_motor_deg[joint_name] = value
                q_idx = self._q_index(placo_name)
                if q_idx is not None:
                    self.placo_robot.state.q[q_idx] = float(np.radians(value))
        else:
            # Legacy per-joint format
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

    def _start_homing(self):
        """Begin smooth homing from current observed positions to neutral pose."""
        self._homing_active = True
        self._homing_start_time = time.perf_counter()
        self._reset_xr_references()
        self._safety_paused = False
        self._ik_failure_count = 0
        # Snapshot current observed joint positions as homing start
        self._homing_start_positions_deg = dict(self._latest_observed_motor_deg)
        # Clear target caches so filter/clip start fresh from current position
        self._last_published_target_deg = {}
        self._filtered_target_deg = {}
        self._last_raw_target_deg = {}
        self._last_filtered_target_deg = {}
        self._last_output_target_deg = {}

    def _homing_step(self) -> Dict[str, float]:
        """Produce one tick of homing interpolation (smoothstep eased)."""
        elapsed = time.perf_counter() - self._homing_start_time
        duration = self._homing_duration

        if elapsed >= duration:
            # Homing complete — hold at exact home position
            self._homing_active = False
            targets = {}
            for side in self.sides:
                prefix = f"{side}_" if self.mode == "dual" else ""
                for joint, home_deg in self._home_pose_deg.items():
                    targets[f"{side}_{joint}"] = home_deg
            logger.info("Homing complete (%.1fs). Holding at home pose.", elapsed)
            self._last_raw_target_deg = dict(targets)
            self._last_output_target_deg = dict(targets)
            return targets

        # Smoothstep easing: t^2 * (3 - 2t) → smooth accel/decel
        t = elapsed / duration
        t = max(0.0, min(1.0, t))
        alpha = t * t * (3.0 - 2.0 * t)

        targets = {}
        for side in self.sides:
            for joint, home_deg in self._home_pose_deg.items():
                motor_name = f"{side}_{joint}"
                start_deg = self._homing_start_positions_deg.get(motor_name, home_deg)
                targets[motor_name] = float(start_deg + (home_deg - start_deg) * alpha)

        self._last_raw_target_deg = dict(targets)
        filtered = self._filter_target(targets)
        clipped = self._clip_target_step(filtered)
        clipped = self._clip_motor_joint_limits(clipped)
        self._last_output_target_deg = dict(clipped)
        return clipped

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

        # ── Homing check (Y or B button edge) ──
        homing_pressed = False
        try:
            homing_pressed = (
                self.xr_client.get_button_state_by_name(self._homing_button_name)
                or self.xr_client.get_button_state_by_name("B")
            )
        except Exception:
            pass
        if homing_pressed and not self._homing_button_was_down and not self._homing_active:
            self._start_homing()
            logger.info("Homing started (target: %s)", self._home_pose_deg)
        self._homing_button_was_down = homing_pressed

        if self._homing_active:
            return self._homing_step()

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
            now = time.perf_counter()
            if now - self._last_confirm_debug_t > 3.0:
                self._last_confirm_debug_t = now
                logger.info("Waiting for %s button (currently pressed: %s)", self.xr_confirm_button, self._confirm_button_was_down)
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
                    # Snapshot current robot joint state to freeze reference
                    self._snapped_robot_q = self.placo_robot.state.q.copy()
                    link = self.task_link_name[name]
                    T = self._fk_from_snapped(link)
                    self.ref_ee_xyz[name] = T[:3, 3].copy()
                    self.ref_ee_quat[name] = self._mat_to_quat(T)
                    if self.effector_control_mode[name] == "position_wrist":
                        self._capture_wrist_pitch_reference(name)
                        self._capture_wrist_roll_reference(name)
                    logger.debug("[%s] Activated.", name)

                xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
                if not self._is_valid_xr_pose(xr_pose):
                    self._log_invalid_xr_pose(name, xr_pose)
                    continue
                delta_xyz, delta_rot, ctrl_quat = self._process_xr(xr_pose, name)
                if self.effector_control_mode[name] in ("position", "position_wrist"):
                    target_xyz = self.ref_ee_xyz[name] + delta_xyz
                    if not np.all(np.isfinite(target_xyz)):
                        self._log_invalid_xr_pose(name, xr_pose)
                        continue
                    if self.effector_control_mode[name] == "position_wrist":
                        target_xyz = self._clamp_position_wrist_target(name, target_xyz)
                    self.effector_task[name].target_world = target_xyz
                    if self.effector_control_mode[name] == "position_wrist":
                        self.wrist_xyz_target_world[name] = target_xyz
                        self._update_desired_wrist_pitch(name, ctrl_quat)
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
                    logger.debug("[%s] Deactivated.", name)
                    self.ref_ee_xyz[name] = None
                    self.ref_ee_quat[name] = None
                    self.ref_controller_xyz[name] = None
                    self.ref_controller_quat[name] = None
                    self._snapped_robot_q = None
                    self.ref_wrist_pitch_elevation_rad.pop(name, None)
                    self.ref_wrist_pitch_horizontal_world.pop(name, None)
                    self.desired_wrist_pitch_elevation_rad.pop(name, None)
                    self.last_wrist_pitch_elevation_rad.pop(name, None)
                    self.direct_wrist_roll_target_rad.pop(name, None)
                    self.wrist_xyz_target_world.pop(name, None)
                self._hold_effector_at_current_pose(name)

        # 3. Gripper
        for name, config in self.manipulator_config.items():
            if "gripper_config" not in config:
                continue
            gc = config["gripper_config"]
            trigger = self.xr_client.get_key_value_by_name(gc["gripper_trigger"])
            for jname, open_p, close_p in zip(gc["joint_names"], gc["open_pos"], gc["close_pos"]):
                self.gripper_pos_target[name][jname] = open_p + (close_p - open_p) * trigger

        any_wrist_roll_active = self._update_wrist_roll_joystick_controls()

        if not any_active:
            self._ik_failure_count = 0
            self._clear_arm_target_cache()
            targets = self._gripper_to_motor_dict()
            if any_wrist_roll_active:
                targets.update(self._wrist_roll_to_motor_dict())
            self._last_raw_target_deg = dict(targets)
            filtered = self._filter_target(targets)
            clipped = self._clip_target_step(filtered)
            clipped = self._clip_motor_joint_limits(clipped)
            self._last_output_target_deg = dict(clipped)
            return clipped

        # 4. Solve (always, like base controller does)
        q_before_solve = self.placo_robot.state.q.copy()
        try:
            if self._uses_position_wrist_4dof_solver():
                self._solve_position_wrist_4dof()
            else:
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
        filtered = self._filter_target(targets)
        clipped = self._clip_target_step(filtered)
        clipped = self._clip_motor_joint_limits(clipped)
        self._last_output_target_deg = dict(clipped)
        if self._target_is_too_far(clipped):
            return {}
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

    def _clip_motor_joint_limits(self, targets: Dict[str, float]) -> Dict[str, float]:
        clipped = {}
        for motor_name, target_deg in targets.items():
            target_deg = float(target_deg)
            joint_suffix = self._joint_suffix(motor_name)
            if joint_suffix in JOINT_LIMITS_RAD and joint_suffix != "gripper":
                lo, hi = JOINT_LIMITS_RAD[joint_suffix]
                target_deg = float(np.clip(target_deg, np.degrees(lo), np.degrees(hi)))
            clipped[motor_name] = target_deg
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
        self._snapped_robot_q = None
        self.ref_wrist_pitch_elevation_rad = {}
        self.ref_wrist_pitch_horizontal_world = {}
        self.desired_wrist_pitch_elevation_rad = {}
        self.last_wrist_pitch_elevation_rad = {}
        self.direct_wrist_roll_target_rad = {}
        self.wrist_xyz_target_world = {}
        self._last_published_target_deg = {}
        self._filtered_target_deg = {}

    def _hold_all_effectors_at_current_pose(self):
        for name in self.manipulator_config:
            self._hold_effector_at_current_pose(name)

    def _fk_from_snapped(self, link_name: str):
        """Compute FK of link from the snapped (frozen) robot state."""
        import placo
        saved_q = self.placo_robot.state.q.copy()
        try:
            self.placo_robot.state.q[:] = self._snapped_robot_q
            self.placo_robot.update_kinematics()
            return self.placo_robot.get_T_world_frame(link_name)
        finally:
            self.placo_robot.state.q[:] = saved_q
            self.placo_robot.update_kinematics()

    def _hold_effector_at_current_pose(self, name: str):
        if self._snapped_robot_q is not None:
            T = self._fk_from_snapped(self.task_link_name[name])
        else:
            T = self.placo_robot.get_T_world_frame(self.task_link_name[name])
        if self.effector_control_mode.get(name) in ("position", "position_wrist"):
            self.effector_task[name].target_world = T[:3, 3].copy()
        else:
            self.effector_task[name].T_world_frame = T.copy()

    def _task_link_name(self, name: str, config: dict) -> str:
        if config.get("control_mode") != "position_wrist":
            return config["link_name"]
        side = name.removesuffix("_hand")
        prefix = f"{side}_" if self.mode == "dual" else ""
        return f"{prefix}wrist_link"

    def _shoulder_anchor_link_name(self, name: str) -> str:
        side = name.removesuffix("_hand")
        prefix = f"{side}_" if self.mode == "dual" else ""
        return f"{prefix}shoulder_link"

    def _clamp_position_wrist_target(self, name: str, target_xyz: np.ndarray) -> np.ndarray:
        max_reach = float(self.max_wrist_reach_m)
        if max_reach <= 0.0:
            return target_xyz

        anchor = self.placo_robot.get_T_world_frame(self._shoulder_anchor_link_name(name))[:3, 3]
        delta = np.asarray(target_xyz, dtype=float) - anchor
        reach = float(np.linalg.norm(delta))
        if not np.isfinite(reach) or reach <= max_reach or reach < 1e-6:
            return target_xyz

        clipped = anchor + delta * (max_reach / reach)
        now = time.perf_counter()
        if now - self._last_workspace_clip_log_t > 1.0:
            self._last_workspace_clip_log_t = now
            logger.warning(
                "[%s] Wrist target clipped by reach limit: %.3f m -> %.3f m. "
                "Reduce arm extension or increase --max-wrist-reach-m if needed.",
                name,
                reach,
                max_reach,
            )
        return clipped

    def _pitch_task_link_name(self, name: str) -> str:
        return self.manipulator_config[name]["link_name"]

    def _wrist_roll_joint_name(self, name: str) -> str:
        side = name.removesuffix("_hand")
        prefix = f"{side}_" if self.mode == "dual" else ""
        return f"{prefix}wrist_roll"

    def _wrist_flex_joint_name(self, name: str) -> str:
        side = name.removesuffix("_hand")
        prefix = f"{side}_" if self.mode == "dual" else ""
        return f"{prefix}wrist_flex"

    def _wrist_pitch_reference_horizontal(self, name: str) -> Optional[np.ndarray]:
        q_idx = self._q_index(self._wrist_flex_joint_name(name))
        if q_idx is None:
            return None

        q_saved = float(self.placo_robot.state.q[q_idx])
        self.placo_robot.state.q[q_idx] = 0.0
        self.placo_robot.update_kinematics()
        T = self.placo_robot.get_T_world_frame(self._pitch_task_link_name(name))
        axis_world = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
        self.placo_robot.state.q[q_idx] = q_saved
        self.placo_robot.update_kinematics()

        norm = np.linalg.norm(axis_world)
        if not np.isfinite(norm) or norm < 1e-6:
            return None
        horizontal = axis_world / norm
        horizontal[2] = 0.0
        horizontal_norm = np.linalg.norm(horizontal)
        if not np.isfinite(horizontal_norm) or horizontal_norm < 1e-6:
            return None
        return horizontal / horizontal_norm

    def _tool_pitch_elevation(self, name: str, hint: Optional[float] = None) -> Optional[float]:
        T = self.placo_robot.get_T_world_frame(self._pitch_task_link_name(name))
        axis_world = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
        norm = np.linalg.norm(axis_world)
        if not np.isfinite(norm) or norm < 1e-6:
            return None
        axis_world = axis_world / norm
        ref_horizontal = self.ref_wrist_pitch_horizontal_world.get(name)
        if ref_horizontal is None:
            ref_horizontal = axis_world.copy()
            ref_horizontal[2] = 0.0
            ref_norm = np.linalg.norm(ref_horizontal)
            if not np.isfinite(ref_norm) or ref_norm < 1e-6:
                return float(np.arcsin(np.clip(axis_world[2], -1.0, 1.0)))
            ref_horizontal = ref_horizontal / ref_norm
        return float(np.arctan2(axis_world[2], float(np.dot(axis_world, ref_horizontal))))

    def _current_tool_pitch_elevation(self, name: str) -> Optional[float]:
        elevation = self._tool_pitch_elevation(name)
        if elevation is not None:
            self.last_wrist_pitch_elevation_rad[name] = elevation
        return elevation

    def _capture_wrist_pitch_reference(self, name: str):
        ref_horizontal = self._wrist_pitch_reference_horizontal(name)
        if ref_horizontal is None:
            return
        self.ref_wrist_pitch_horizontal_world[name] = ref_horizontal
        elevation = self._current_tool_pitch_elevation(name)
        if elevation is None:
            return
        self.ref_wrist_pitch_elevation_rad[name] = elevation
        self.desired_wrist_pitch_elevation_rad[name] = elevation

    def _update_desired_wrist_pitch(self, name: str, ctrl_quat):
        ref_quat = self.ref_controller_quat.get(name)
        ref_elevation = self.ref_wrist_pitch_elevation_rad.get(name)
        if ref_quat is None or ref_elevation is None:
            return

        _roll, pitch = self._controller_delta_roll_pitch(ref_quat, ctrl_quat)
        target = ref_elevation - pitch * self.wrist_pitch_scale
        self.desired_wrist_pitch_elevation_rad[name] = float(np.clip(target, -self.max_wrist_pitch_rad, self.max_wrist_pitch_rad))

    def _capture_wrist_roll_reference(self, name: str):
        q_idx = self._q_index(self._wrist_roll_joint_name(name))
        if q_idx is None:
            return
        self.direct_wrist_roll_target_rad[name] = float(self.placo_robot.state.q[q_idx])

    def _stick_value(self, name: str) -> float:
        side = name.removesuffix("_hand")
        axis = self.wrist_roll_stick_axis.lower()
        try:
            joystick = self.xr_client.get_joystick_state(side)
            idx = 1 if axis == "y" else 0
            value = float(joystick[idx])
            if np.isfinite(value):
                return float(np.clip(value, -1.0, 1.0))
        except Exception:
            pass

        candidates = [
            f"{side}_thumbstick_{axis}",
            f"{side}_joystick_{axis}",
            f"{side}_stick_{axis}",
            f"{side}_axis_{axis}",
            f"{side}_primary2d_{axis}",
        ]
        for key_name in candidates:
            try:
                value = float(self.xr_client.get_key_value_by_name(key_name))
            except Exception:
                continue
            if np.isfinite(value) and abs(value) > 1e-6:
                return float(np.clip(value, -1.0, 1.0))
        return 0.0

    def _update_wrist_roll_joystick_controls(self) -> bool:
        any_active = False
        for name in self.manipulator_config:
            if self.effector_control_mode.get(name) != "position_wrist":
                continue
            any_active = self._update_wrist_roll_from_stick(name) or any_active
        return any_active

    def _update_wrist_roll_from_stick(self, name: str) -> bool:
        q_idx = self._q_index(self._wrist_roll_joint_name(name))
        if q_idx is None:
            return False

        target = self.direct_wrist_roll_target_rad.get(name)
        if target is None:
            target = float(self.placo_robot.state.q[q_idx])

        stick = self._stick_value(name)
        if abs(stick) < self.wrist_roll_stick_deadzone:
            return False
        target += stick * self.wrist_roll_scale * self.wrist_roll_speed_rad * self.dt
        target = self._clip_joint_rad("wrist_roll", target)
        self.direct_wrist_roll_target_rad[name] = target
        self.placo_robot.state.q[q_idx] = target
        return True

    def _uses_position_wrist_4dof_solver(self) -> bool:
        return any(
            self.active.get(name, False) and self.effector_control_mode.get(name) == "position_wrist"
            for name in self.manipulator_config
        )

    def _position_wrist_joint_names(self, name: str) -> list[str]:
        side = name.removesuffix("_hand")
        prefix = f"{side}_" if self.mode == "dual" else ""
        return [
            f"{prefix}shoulder_pan",
            f"{prefix}shoulder_lift",
            f"{prefix}elbow_flex",
            f"{prefix}wrist_flex",
        ]

    def _position_wrist_task_vector(self, name: str, elevation_hint: Optional[float] = None) -> Optional[np.ndarray]:
        elevation = self._tool_pitch_elevation(name, hint=elevation_hint)
        if elevation is None:
            return None
        T = self.placo_robot.get_T_world_frame(self.task_link_name[name])
        return np.array([T[0, 3], T[1, 3], T[2, 3], elevation], dtype=float)

    @staticmethod
    def _joint_suffix(joint_name: str) -> str:
        for suffix in SO101_JOINT_NAMES:
            if joint_name.endswith(suffix):
                return suffix
        return joint_name

    def _solve_position_wrist_4dof(self):
        for name in self.manipulator_config:
            if self.effector_control_mode.get(name) != "position_wrist" or not self.active.get(name, False):
                continue

            target_xyz = self.wrist_xyz_target_world.get(name)
            target_elevation = self.desired_wrist_pitch_elevation_rad.get(name)
            if target_xyz is None or target_elevation is None:
                continue

            joint_names = self._position_wrist_joint_names(name)
            q_indices = [self._q_index(joint_name) for joint_name in joint_names]
            if any(q_idx is None for q_idx in q_indices):
                raise RuntimeError(f"Missing 4DOF wrist IK joints for {name}: {joint_names}")
            q_indices = [int(q_idx) for q_idx in q_indices]
            target = np.array([target_xyz[0], target_xyz[1], target_xyz[2], target_elevation], dtype=float)

            for joint_name, q_idx in zip(joint_names, q_indices):
                suffix = self._joint_suffix(joint_name)
                if suffix in JOINT_LIMITS_RAD:
                    self.placo_robot.state.q[q_idx] = self._clip_joint_rad(suffix, float(self.placo_robot.state.q[q_idx]))

            for _ in range(24):
                self.placo_robot.update_kinematics()
                current = self._position_wrist_task_vector(name, elevation_hint=target_elevation)
                if current is None or not np.all(np.isfinite(current)):
                    raise RuntimeError(f"Invalid 4DOF wrist IK state for {name}")
                error = target - current
                if np.linalg.norm(error[:3]) < 5e-4 and abs(error[3]) < np.radians(0.5):
                    break

                jacobian = np.zeros((4, 4), dtype=float)
                eps = 1e-4
                q0 = self.placo_robot.state.q.copy()
                for col, q_idx in enumerate(q_indices):
                    self.placo_robot.state.q[q_idx] = q0[q_idx] + eps
                    self.placo_robot.update_kinematics()
                    plus = self._position_wrist_task_vector(name, elevation_hint=target_elevation)
                    if plus is None or not np.all(np.isfinite(plus)):
                        raise RuntimeError(f"Invalid 4DOF wrist IK Jacobian for {name}")
                    jacobian[:, col] = (plus - current) / eps
                    self.placo_robot.state.q[q_idx] = q0[q_idx]
                self.placo_robot.update_kinematics()

                damping = 1e-3
                lhs = jacobian @ jacobian.T + (damping * damping) * np.eye(4)
                dq = jacobian.T @ np.linalg.solve(lhs, error)
                if not np.all(np.isfinite(dq)):
                    raise RuntimeError(f"Non-finite 4DOF wrist IK update for {name}")
                dq = np.clip(dq, -0.12, 0.12)

                for joint_name, q_idx, delta in zip(joint_names, q_indices, dq):
                    suffix = self._joint_suffix(joint_name)
                    q_next = float(self.placo_robot.state.q[q_idx] + delta)
                    if suffix in JOINT_LIMITS_RAD:
                        q_next = self._clip_joint_rad(suffix, q_next)
                    self.placo_robot.state.q[q_idx] = q_next

        self.placo_robot.update_kinematics()
        for name in self.manipulator_config:
            if self.effector_control_mode.get(name) == "position_wrist" and self.active.get(name, False):
                self._current_tool_pitch_elevation(name)

    @staticmethod
    def _clip_joint_rad(joint_suffix: str, value: float) -> float:
        lo, hi = JOINT_LIMITS_RAD[joint_suffix]
        return float(np.clip(value, lo, hi))

    @staticmethod
    def _controller_delta_roll_pitch(ref_quat, ctrl_quat) -> tuple[float, float]:
        from meshcat import transformations as tf

        ref_R = tf.quaternion_matrix(ref_quat)[:3, :3]
        ctrl_R = tf.quaternion_matrix(ctrl_quat)[:3, :3]
        delta_R = ref_R.T @ ctrl_R
        delta_T = np.eye(4)
        delta_T[:3, :3] = delta_R
        roll, pitch, _yaw = tf.euler_from_matrix(delta_T, axes="sxyz")
        continuous_pitch = np.arctan2(delta_R[0, 2], delta_R[0, 0])
        return float(roll), float(continuous_pitch)

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

    def _wrist_roll_to_motor_dict(self) -> Dict[str, float]:
        targets = {}
        for hand_name, rad in self.direct_wrist_roll_target_rad.items():
            side = hand_name.removesuffix("_hand")
            motor_name = f"{side}_wrist_roll"
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
            "desired_wrist_pitch_elevation_deg": {
                name: float(np.degrees(rad))
                for name, rad in self.desired_wrist_pitch_elevation_rad.items()
            },
            "direct_wrist_roll_target_deg": {
                name: float(np.degrees(rad))
                for name, rad in self.direct_wrist_roll_target_rad.items()
            },
            "wrist_xyz_target_world": {
                name: np.round(xyz, 6).tolist()
                for name, xyz in self.wrist_xyz_target_world.items()
            },
            "wrist_reach_m": {
                name: float(np.linalg.norm(
                    np.asarray(xyz) - self.placo_robot.get_T_world_frame(self._shoulder_anchor_link_name(name))[:3, 3]
                ))
                for name, xyz in self.wrist_xyz_target_world.items()
            },
            "observed_deg": dict(self._latest_observed_motor_deg),
            "active": dict(self.active),
            "xr_confirmed": bool(self.xr_confirmed),
            "safety_paused": bool(self._safety_paused),
            "homing_active": bool(self._homing_active),
            "homing_progress": float(
                min(1.0, (time.perf_counter() - self._homing_start_time) / self._homing_duration)
            ) if self._homing_active else 0.0,
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
            return np.zeros(3), np.zeros(3), ctrl_quat

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
        return delta_xyz, delta_rot, ctrl_quat

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


def _save_episode_quiet(dataset):
    """Save episode with suppressed video encoder output and progress bar."""
    import os, sys, threading, time as _time
    done = [False]

    def _spinner():
        chars = "|/-\\"
        i = 0
        while not done[0]:
            sys.stderr.write(f"\r  Encoding videos... {chars[i % len(chars)]}")
            sys.stderr.flush()
            i += 1
            _time.sleep(0.15)
        sys.stderr.write("\r  Encoding videos... done     \n")
        sys.stderr.flush()

    t = threading.Thread(target=_spinner, daemon=True)
    old_env = os.environ.get("SVT_LOG")
    os.environ["SVT_LOG"] = "0"
    try:
        t.start()
        dataset.save_episode()
    finally:
        done[0] = True
        t.join(timeout=1)
        if old_env is not None:
            os.environ["SVT_LOG"] = old_env
        else:
            os.environ.pop("SVT_LOG", None)


def decode_frame(frame: dict) -> dict:
    """Decode JPEG-encoded images in a LeRobot frame dict."""
    out = {}
    for k, v in frame.items():
        if isinstance(v, dict) and v.get("__remote_image_encoding__") == "jpg":
            img = cv2.imdecode(np.frombuffer(v["data"], dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                out[k] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(v, (float, int)):
            out[k] = np.array([v], dtype=np.float32)
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
    parser.add_argument("--state-port", type=int, default=5572, help="Port receiving lightweight 60Hz joint state from Jetson (0=disable)")
    parser.add_argument("--target-port", type=int, default=5580, help="Port sending joint targets to Jetson")
    parser.add_argument("--command-port", type=int, default=5571, help="Port sending commands to Jetson")
    parser.add_argument("--mode", default="single", choices=["single", "dual"])
    parser.add_argument("--scale-factor", type=float, default=1.0, help="XR motion scaling")
    parser.add_argument("--control-rate-hz", type=int, default=60, help="IK solver frequency (aligned with Jetson control_fps)")
    parser.add_argument("--debug-vis", action="store_true", help="Show real-time joint target + safety visualization")
    parser.add_argument(
        "--xr-frame",
        default="openxr",
        choices=sorted(XR_FRAME_ROTATIONS.keys()),
        help="XR tracking coordinate convention: openxr uses -Z forward, unity uses +Z forward",
    )
    parser.add_argument(
        "--ik-control-mode",
        default="pose",
        choices=["pose", "position", "position_wrist"],
        help="Use full pose IK, gripper-frame position IK, or wrist-link xyz + wrist elevation IK",
    )
    parser.add_argument("--wrist-pitch-scale", type=float, default=1.0, help="Scale for controller pitch -> wrist/tool elevation in position_wrist mode (use -1 to invert)")
    parser.add_argument("--wrist-roll-scale", type=float, default=-1.0, help="Scale for joystick -> wrist_roll velocity in position_wrist mode (use opposite sign to invert)")
    parser.add_argument("--wrist-roll-speed-deg", type=float, default=45.0, help="Max wrist_roll joystick velocity in deg/s")
    parser.add_argument("--wrist-roll-stick-axis", default="x", choices=["x", "y"], help="Joystick axis used for wrist_roll velocity")
    parser.add_argument("--wrist-roll-stick-deadzone", type=float, default=0.08, help="Joystick deadzone for wrist_roll velocity")
    parser.add_argument("--max-wrist-pitch-deg", type=float, default=130.0, help="Max absolute wrist/tool pitch target in position_wrist mode")
    parser.add_argument("--max-wrist-reach-m", type=float, default=0.285, help="Position-wrist max shoulder_link->wrist_link reach in meters (0=disable)")
    parser.add_argument("--require-xr-confirm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--xr-confirm-button", default="A", help="XR controller button used to confirm headset-yaw alignment")
    parser.add_argument("--xr-record-button", default="X", help="XR controller button used to start/stop recording")
    parser.add_argument("--debug-xr-delta", action="store_true", help="Log raw and robot-frame controller deltas")
    parser.add_argument(
        "--max-action-delta-deg",
        type=float,
        default=60.0,
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
        default=0.05,
        help="Low-pass filter alpha for non-gripper IK targets before clipping (0<alpha<1, smaller=smoother/slower, 1=disable)",
    )
    parser.add_argument(
        "--ik-target-log",
        type=Path,
        default=None,
        help="Enable IK target JSONL logging to this path (disabled by default)",
    )
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--preview-height", type=int, default=240)
    parser.add_argument("--mjpeg-port", type=int, default=0, help="MJPEG streaming port for browser (0=disable)")
    parser.add_argument("--mjpeg-quality", type=int, default=70, help="MJPEG JPEG quality (10-100)")
    parser.add_argument("--h264-port", type=int, default=12345, help="H.264 TCP video port (0=disable)")
    parser.add_argument("--h264-control-port", type=int, default=13579, help="H.264 TCP control port")
    parser.add_argument("--h264-width", type=int, default=1280, help="H.264 output width")
    parser.add_argument("--h264-height", type=int, default=720, help="H.264 output height")
    parser.add_argument("--h264-fps", type=int, default=15, help="H.264 output FPS")
    parser.add_argument("--h264-bitrate", type=int, default=8_000_000, help="H.264 bitrate in bps")
    parser.add_argument("--record-every-n", type=int, default=1, help="Record every Nth frame (1=all at Jetson fps, typically 15Hz)")
    parser.add_argument("--record-image-writer-threads", type=int, default=12, help="Async image writer threads")
    args = parser.parse_args()

    LeRobotDataset, DEFAULT_FEATURES, build_dataset_frame, hw_to_dataset_features = _ensure_lerobot_deps()

    # ── ZMQ Setup ──
    ctx = zmq.Context()

    # Receive observations from Jetson (PULL — same protocol as LeRobot receiver)
    obs_socket = ctx.socket(zmq.PULL)
    obs_socket.setsockopt(zmq.RCVHWM, 1)
    obs_socket.bind(f"tcp://{args.listen_ip}:{args.obs_port}")

    state_socket = None
    if args.state_port > 0:
        state_socket = ctx.socket(zmq.PULL)
        state_socket.setsockopt(zmq.RCVHWM, 1)
        state_socket.setsockopt(zmq.RCVTIMEO, 1000)
        state_socket.bind(f"tcp://{args.listen_ip}:{args.state_port}")

    # Send joint targets to Jetson (PUB)
    target_socket = ctx.socket(zmq.PUB)
    target_socket.setsockopt(zmq.SNDHWM, 500)  # buffer ~2s at 250Hz to avoid drops
    try: target_socket.setsockopt(50, 1)  # ZMQ_TCP_NODELAY
    except Exception: pass
    target_socket.bind(f"tcp://{args.listen_ip}:{args.target_port}")

    # Send commands to Jetson (PUB)
    cmd_socket = ctx.socket(zmq.PUB)
    cmd_socket.setsockopt(zmq.SNDHWM, 1)
    cmd_socket.bind(f"tcp://{args.listen_ip}:{args.command_port}")

    logger.info("Listening for observations on tcp://%s:%d", args.listen_ip, args.obs_port)
    if state_socket is not None:
        logger.info("Listening for 60Hz joint state on tcp://%s:%d", args.listen_ip, args.state_port)
    logger.info("Publishing joint targets on tcp://%s:%d", args.listen_ip, args.target_port)

    ik_target_log_fh = None
    if args.ik_target_log is not None:
        ik_target_log_path = args.ik_target_log
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

    # ── H.264 TCP streamer (for PICO APK Remote Vision) ──
    h264 = None
    if args.h264_port > 0:
        h264 = H264TCPStreamer(
            control_port=args.h264_control_port,
            video_port=args.h264_port,
            width=args.h264_width,
            height=args.h264_height,
            fps=args.h264_fps,
            bitrate=args.h264_bitrate,
        )
        h264.start()
        logger.info("H.264 TCP streaming: %s", h264.url)
        logger.info("  PICO APK Remote Vision → ZEDMINI → Listen")

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
        wrist_pitch_scale=args.wrist_pitch_scale,
        wrist_roll_scale=args.wrist_roll_scale,
        wrist_roll_speed_deg=args.wrist_roll_speed_deg,
        wrist_roll_stick_axis=args.wrist_roll_stick_axis,
        wrist_roll_stick_deadzone=args.wrist_roll_stick_deadzone,
        max_wrist_pitch_deg=args.max_wrist_pitch_deg,
        max_wrist_reach_m=args.max_wrist_reach_m,
        xr_frame=args.xr_frame,
        ik_control_mode=args.ik_control_mode,
        require_xr_confirm=args.require_xr_confirm,
        xr_confirm_button=args.xr_confirm_button,
        debug_xr_delta=args.debug_xr_delta,
    ))
    logger.info(
        "IK target smoothing: alpha=%.2f, max_step=%.2f deg/tick, wrist_pitch_scale=%.2f, max_wrist_pitch=%.1f deg, wrist_roll_scale=%.2f, wrist_roll_speed=%.1f deg/s, max_wrist_reach=%.3f m",
        args.target_filter_alpha,
        args.max_target_step_deg,
        args.wrist_pitch_scale,
        args.max_wrist_pitch_deg,
        args.wrist_roll_scale,
        args.wrist_roll_speed_deg,
        args.max_wrist_reach_m,
    )
    if args.ik_control_mode == "position_wrist":
        logger.info(
            "Position-wrist mode: IK tracks wrist_link xyz + scalar wrist/tool elevation; joystick controls wrist_roll velocity.",
        )

    ik_controller_lock = threading.RLock()
    state_stop = threading.Event()
    state_thread = None
    if state_socket is not None:
        def state_loop():
            state_count = 0
            last_log_t = time.perf_counter()
            while not state_stop.is_set():
                try:
                    msg = state_socket.recv_pyobj()
                except zmq.Again:
                    continue
                except Exception as e:
                    if not state_stop.is_set():
                        logger.warning("State receiver error: %s", e)
                    continue
                if not isinstance(msg, dict) or msg.get("type") != "state":
                    continue
                frame = msg.get("frame", {})
                try:
                    with ik_controller_lock:
                        _ik_ref.get().update_robot_state(frame)
                    state_count += 1
                except Exception as e:
                    logger.warning("update_robot_state from state stream failed: %s", e)
                now = time.perf_counter()
                if now - last_log_t >= 5.0:
                    logger.debug("State stream: %.1f msg/s", state_count / (now - last_log_t))
                    state_count = 0
                    last_log_t = now

        state_thread = threading.Thread(target=state_loop, daemon=True, name="state-recv")
        state_thread.start()

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

    # ── Debug visualizer ──
    debug_viz = None
    if args.debug_vis:
        import os, sys
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        from debug_ik_visualizer import DebugIKVisualizer
        viz_joints = []
        for side in (["left", "right"] if args.mode == "dual" else ["right"]):
            for j in SO101_JOINT_NAMES:
                viz_joints.append(f"{side}_{j}")
        viz_save = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs",
                               f"ik_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        debug_viz = DebugIKVisualizer(viz_joints, history_s=5.0, update_hz=10, save_path=viz_save)
        debug_viz.start()
        logger.info("Debug IK visualizer started (%d joints)", len(viz_joints))

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
                with ik_controller_lock:
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
                    # Debug visualization (always when enabled)
                    if debug_viz is not None:
                        with ik_controller_lock:
                            snapshot = ctrl.target_debug_snapshot()
                        debug_viz.update(snapshot)
                    # IK target JSONL logging (when enabled)
                    if ik_target_log_fh is not None:
                        if debug_viz is None:
                            with ik_controller_lock:
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
            if now - ik_last_stats_t >= 5.0:
                dt = now - ik_last_stats_t
                logger.debug(
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
    _saving = False  # guard against double-trigger during video encoding

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

    logger.info("Controls: XR %s/Space=rec, R=discard, Q/Esc=quit, Y/B=home arms", args.xr_record_button)

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
                # Auto-align H.264 FPS to match Jetson observation rate
                if h264 is not None and msg.get("fps"):
                    h264._fps = int(msg["fps"])
                    logger.info("H.264 FPS aligned to Jetson: %d", h264._fps)
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
                            dataset.start_image_writer(num_threads=args.record_image_writer_threads)
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
                                image_writer_threads=args.record_image_writer_threads,
                            )
                            logger.info("Dataset created (after cleanup): %s", dataset.root)
                    else:
                        dataset = LeRobotDataset.create(
                            repo_id=args.repo_id, fps=msg["fps"], root=str(root),
                            robot_type=msg.get("robot_type", "so101_follower"),
                            features=features, use_videos=True,
                            image_writer_threads=args.record_image_writer_threads,
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

                # Push BGR frames to H.264 TCP streamer
                if h264 is not None and image_keys:
                    raw_frame = msg["frame"]
                    for ik in image_keys:
                        img_data = raw_frame.get(ik)
                        if isinstance(img_data, dict) and img_data.get("__remote_image_encoding__") == "jpg":
                            bgr = cv2.imdecode(np.frombuffer(img_data["data"], dtype=np.uint8), cv2.IMREAD_COLOR)
                            if bgr is not None:
                                short_name = ik.replace("observation.images.", "")
                                h264.update(short_name, bgr)

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

                # Feed real robot state to IK only when the dedicated 60Hz state stream is disabled.
                # With state stream enabled, the 15Hz video/record frame would otherwise overwrite
                # fresher joint state and make IK debug look stair-stepped again.
                if state_socket is None:
                    try:
                        with ik_controller_lock:
                            _ik_ref.get().update_robot_state(frame)
                    except Exception as e:
                        now = time.perf_counter()
                        if now - last_fps_t >= 5.0:
                            logger.warning("update_robot_state failed: %s", e)
                now = time.perf_counter()
                if now - last_fps_t >= 1.0:
                    fps_display = recv_frames / (now - last_fps_t)
                    recv_frames = 0
                    last_fps_t = now

                if recording:
                    if episode_frame_count % args.record_every_n == 0:
                        missing = [ik for ik in image_keys if ik not in frame]
                        if not missing:
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

                if keyboard_events["toggle_recording"] and not _saving:
                    keyboard_events["toggle_recording"] = False
                    if recording:
                        if episode_frame_count > 0:
                            _saving = True
                            fps = dataset.fps if dataset else 15
                            logger.info("Saving episode %d (%d frames, %.1fs) ...",
                                        dataset.num_episodes, episode_frame_count,
                                        episode_frame_count / max(fps, 1))
                            _save_episode_quiet(dataset)
                            saved_episodes += 1
                            logger.info("Episode %d saved ✓", dataset.num_episodes - 1)
                            _saving = False
                        recording = False
                        episode_frame_count = 0
                    else:
                        dataset.clear_episode_buffer()
                        episode_frame_count = 0
                        recording = True
                        logger.info("Started episode %d", dataset.num_episodes)

                if keyboard_events["discard_episode"] and not _saving:
                    keyboard_events["discard_episode"] = False
                    dataset.clear_episode_buffer()
                    episode_frame_count = 0
                    recording = False
                    logger.info("Discarded current episode.")
                    if recording and episode_frame_count > 0:
                        logger.info("Saving final episode %d ...", dataset.num_episodes)
                        _save_episode_quiet(dataset)
                        saved_episodes += 1
                        logger.info("Episode %d saved ✓", dataset.num_episodes - 1)
                    cmd_socket.send_string("stop")
                    break

            elif msg_type == "done":
                if recording and episode_frame_count > 0:
                    logger.info("Saving final episode %d ...", dataset.num_episodes)
                    _save_episode_quiet(dataset)
                    saved_episodes += 1
                    logger.info("Episode %d saved ✓", dataset.num_episodes - 1)
                break

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        state_stop.set()
        if state_thread is not None and state_thread.is_alive():
            state_thread.join(timeout=2)
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
        if h264:
            h264.stop()
        if debug_viz:
            debug_viz.stop()
        if ik_target_log_fh is not None:
            ik_target_log_fh.close()
        cmd_socket.send_string("stop")
        target_socket.close(0)
        obs_socket.close(0)
        if state_socket is not None:
            state_socket.close(0)
        cmd_socket.close(0)
        ctx.term()
        logger.info("Shutdown complete. Total saved episodes: %d", saved_episodes)


if __name__ == "__main__":
    main()
