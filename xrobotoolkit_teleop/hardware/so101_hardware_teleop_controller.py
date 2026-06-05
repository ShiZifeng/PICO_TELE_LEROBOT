"""
SO-101 Hardware Teleop Controller

Integrates XRoboToolkit's XR-based teleop (PICO headset + Placo IK) with
real SO-101 hardware via LeRobot's FeetechMotorsBus (STS3215 servos).

Supports:
  - Single arm (right hand) and dual arm (left + right) modes
  - XR pose → Placo IK → STS3215 servo position commands
  - Data recording in LeRobot-compatible format
  - OpenCV camera capture for wrist/head cameras
  - B-button toggle for recording episodes
"""

import os
import sys
import time
import threading
from typing import Any, Dict, Optional

import cv2
import numpy as np

from xrobotoolkit_teleop.common.base_hardware_teleop_controller import HardwareTeleopController
from xrobotoolkit_teleop.hardware.interface.base_camera import BaseCameraInterface
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

# SO-101 joint names and motor IDs (STS3215 servos on Feetech bus)
SO101_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SO101_ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SO101_MOTOR_IDS = {"shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3, "wrist_flex": 4, "wrist_roll": 5, "gripper": 6}

# URDF joint limits (radians) — used for radian↔motor unit conversion
SO101_JOINT_LIMITS_RAD = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": (-0.174533, 1.74533),  # closed → open
}

# Community-tested PID gains from LeRobot
SO101_PID_GAINS = {
    "shoulder_pan": (12, 0, 6),
    "shoulder_lift": (15, 8, 10),
    "elbow_flex": (14, 6, 12),
    "wrist_flex": (15, 0, 14),
    "wrist_roll": (12, 0, 12),
    "gripper": (12, 0, 12),
}


def _find_lerobot_path() -> Optional[str]:
    """Try to find the LeRobot package in common locations."""
    env_path = os.environ.get("LEROBOT_PATH")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    bundled_path = os.path.join(project_root, "third_party", "lerobot-v0.3.3")
    candidates = [
        os.path.join(env_path, "src") if env_path else None,
        env_path,
        os.path.join(bundled_path, "src"),
        "/media/shizifeng/projects21/lerobot-v0.3.3/src",
        os.path.expanduser("~/szf_lerobot/lerobot-v0.3.3/src"),
    ]
    for p in candidates:
        if p and os.path.isdir(os.path.join(p, "lerobot")):
            return p
    # Try importing directly (if installed via pip)
    try:
        import lerobot
        return None  # Already importable
    except ImportError:
        pass
    return None


class OpenCVCameraInterface(BaseCameraInterface):
    """Simple OpenCV camera interface compatible with HardwareTeleopController."""

    def __init__(self, camera_configs: Dict[str, Dict[str, Any]]):
        self.camera_configs = camera_configs
        self.caps: Dict[str, cv2.VideoCapture] = {}
        self.frames: Dict[str, Dict[str, np.ndarray]] = {}
        self._running = False

    def start(self):
        for name, cfg in self.camera_configs.items():
            idx = cfg.get("index", 0)
            cap = cv2.VideoCapture(idx)
            if cfg.get("width"):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
            if cfg.get("height"):
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
            if cfg.get("fps"):
                cap.set(cv2.CAP_PROP_FPS, cfg["fps"])
            self.caps[name] = cap
            self.frames[name] = {}
        self._running = True

    def stop(self):
        self._running = False
        for cap in self.caps.values():
            cap.release()

    def update_frames(self):
        if not self._running:
            return
        for name, cap in self.caps.items():
            ret, frame = cap.read()
            if ret:
                self.frames[name]["color"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_frames(self) -> Dict[str, Dict[str, np.ndarray]]:
        return self.frames

    def get_compressed_frames(self, quality: int = 85) -> Dict[str, Dict[str, bytes]]:
        result = {}
        for name, frame_dict in self.frames.items():
            result[name] = {}
            if "color" in frame_dict:
                img = cv2.cvtColor(frame_dict["color"], cv2.COLOR_RGB2BGR)
                _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                result[name]["color"] = buf.tobytes()
        return result


class SO101HardwareTeleopController(HardwareTeleopController):
    """
    Hardware teleop controller for SO-101 robot arm(s).

    Uses XR headset + Placo IK for teleop, and LeRobot's FeetechMotorsBus
    for real hardware communication with STS3215 servos.

    Supports single-arm (right hand) and dual-arm (left + right) modes.
    """

    def __init__(
        self,
        robot_urdf_path: str = None,
        mode: str = "single",  # "single" or "dual"
        ports: Dict[str, str] = None,  # {"right": "/dev/right_follower"} or {"left": ..., "right": ...}
        camera_configs: Dict[str, Dict[str, Any]] = None,
        scale_factor: float = 1.0,
        control_rate_hz: int = 50,
        enable_log_data: bool = True,
        log_dir: str = "logs/so101",
        log_freq: float = 30,
        enable_camera: bool = False,
        camera_fps: int = 30,
        use_degrees: bool = True,
        max_relative_target: Optional[float] = None,
        **kwargs,
    ):
        """
        Args:
            robot_urdf_path: Path to SO-101 URDF for Placo IK.
            mode: "single" (one arm, right hand) or "dual" (left + right).
            ports: Dict mapping arm side to serial port, e.g. {"right": "/dev/ttyUSB0"}.
            camera_configs: Dict of camera name → {index, width, height, fps}.
            scale_factor: XR motion scaling.
            control_rate_hz: IK + control loop frequency.
            enable_log_data: Enable data recording.
            log_dir: Directory for recorded data.
            log_freq: Recording frequency.
            enable_camera: Enable camera capture.
            camera_fps: Camera capture FPS.
            use_degrees: Use degrees for motor commands (True) or RANGE_M100_100 (False).
            max_relative_target: Safety clamp for per-step motor movement.
        """
        self._mode = mode
        self._ports = ports or {}
        self._camera_cfgs = camera_configs or {}
        self._use_degrees = use_degrees
        self._max_relative_target = max_relative_target
        self._robot_buses: Dict[str, Any] = {}  # side → FeetechMotorsBus
        self._lerobot_imported = False

        if robot_urdf_path is None:
            if mode == "dual":
                robot_urdf_path = os.path.join(ASSET_PATH, "so101", "dual_so101.urdf")
            else:
                robot_urdf_path = os.path.join(ASSET_PATH, "so101", "so101_new_calib.urdf")

        # Build manipulator config for the base class
        if mode == "dual":
            manipulator_config = {
                "left_hand": {
                    "link_name": "left_gripper_frame_link",
                    "pose_source": "left_controller",
                    "control_trigger": "left_grip",
                    "gripper_config": {
                        "type": "parallel",
                        "gripper_trigger": "left_trigger",
                        "joint_names": ["left_gripper"],
                        "open_pos": [1.74533],
                        "close_pos": [-0.174533],
                    },
                },
                "right_hand": {
                    "link_name": "right_gripper_frame_link",
                    "pose_source": "right_controller",
                    "control_trigger": "right_grip",
                    "gripper_config": {
                        "type": "parallel",
                        "gripper_trigger": "right_trigger",
                        "joint_names": ["right_gripper"],
                        "open_pos": [1.74533],
                        "close_pos": [-0.174533],
                    },
                },
            }
        else:
            manipulator_config = {
                "right_hand": {
                    "link_name": "gripper_frame_link",
                    "pose_source": "right_controller",
                    "control_trigger": "right_grip",
                    "gripper_config": {
                        "type": "parallel",
                        "gripper_trigger": "right_trigger",
                        "joint_names": ["gripper"],
                        "open_pos": [1.74533],
                        "close_pos": [-0.174533],
                    },
                },
            }

        super().__init__(
            robot_urdf_path=robot_urdf_path,
            manipulator_config=manipulator_config,
            R_headset_world=R_HEADSET_TO_WORLD,
            floating_base=False,
            scale_factor=scale_factor,
            visualize_placo=False,
            control_rate_hz=control_rate_hz,
            enable_log_data=enable_log_data,
            log_dir=log_dir,
            log_freq=log_freq,
            enable_camera=enable_camera,
            camera_fps=camera_fps,
            q_init=None,
        )

    def _ensure_lerobot(self):
        """Import LeRobot motor modules."""
        if self._lerobot_imported:
            return
        lerobot_path = _find_lerobot_path()
        if lerobot_path and lerobot_path not in sys.path:
            sys.path.insert(0, lerobot_path)
        try:
            from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
            from lerobot.motors import Motor, MotorNormMode

            self._FeetechMotorsBus = FeetechMotorsBus
            self._OperatingMode = OperatingMode
            self._Motor = Motor
            self._MotorNormMode = MotorNormMode
            self._lerobot_imported = True
        except ImportError as e:
            raise ImportError(
                "LeRobot package not found. Install it with:\n"
                "  pip install -e /path/to/lerobot-v0.3.3\n"
                f"Original error: {e}"
            )

    # ============================================================
    # Abstract method implementations
    # ============================================================

    def _robot_setup(self):
        """Initialize SO-101 hardware via FeetechMotorsBus."""
        self._ensure_lerobot()

        sides = ["right"] if self._mode == "single" else ["left", "right"]
        norm_mode = self._MotorNormMode.DEGREES if self._use_degrees else self._MotorNormMode.RANGE_M100_100

        for side in sides:
            port = self._ports.get(side)
            if not port:
                raise ValueError(f"No port specified for {side} arm. Provide --ports.")

            motors = {}
            for joint_name in SO101_JOINT_NAMES:
                motor_name = f"{side}_{joint_name}" if self._mode == "dual" else joint_name
                # Gripper uses RANGE_0_100 for LeRobot convention
                if joint_name == "gripper":
                    m_norm = self._MotorNormMode.RANGE_0_100
                else:
                    m_norm = norm_mode
                motors[motor_name] = self._Motor(SO101_MOTOR_IDS[joint_name], "sts3215", m_norm)

            bus = self._FeetechMotorsBus(port=port, motors=motors)
            bus.connect()

            # Configure PID gains and operating mode
            with bus.torque_disabled():
                bus.configure_motors()
                for motor_name in motors:
                    joint_base = joint_name if self._mode == "single" else "_".join(motor_name.split("_")[1:])
                    # Map back to base joint name for PID lookup
                    lookup = motor_name.replace(f"{side}_", "") if self._mode == "dual" else motor_name
                    p, i, d = SO101_PID_GAINS.get(lookup, (12, 0, 12))
                    bus.write("P_Coefficient", motor_name, p)
                    bus.write("I_Coefficient", motor_name, i)
                    bus.write("D_Coefficient", motor_name, d)
                    bus.write("Operating_Mode", motor_name, self._OperatingMode.POSITION.value)

            self._robot_buses[side] = bus
            print(f"[SO101] {side} arm connected on {port}")

        print("[SO101] Robot setup complete.")

    def _initialize_camera(self):
        """Initialize OpenCV cameras."""
        if not self.enable_camera or not self._camera_cfgs:
            self.camera_interface = None
            return
        self.camera_interface = OpenCVCameraInterface(self._camera_cfgs)
        self.camera_interface.start()
        print(f"[SO101] Cameras initialized: {list(self._camera_cfgs.keys())}")

    def _update_robot_state(self):
        """Read current joint positions from hardware into Placo."""
        placo_q = self.placo_robot.state.q.copy()

        if self._mode == "dual":
            for side in ["left", "right"]:
                bus = self._robot_buses.get(side)
                if bus is None:
                    continue
                try:
                    present = bus.sync_read("Present_Position")
                except Exception:
                    continue
                for i, joint_name in enumerate(SO101_JOINT_NAMES):
                    motor_name = f"{side}_{joint_name}"
                    if motor_name in present:
                        rad = self._motor_to_radians(joint_name, present[motor_name])
                        placo_joint = f"{side}_{joint_name}"
                        if placo_joint in self.placo_robot.model.names:
                            jid = self.placo_robot.model.getJointId(placo_joint)
                            if jid < len(placo_q):
                                placo_q[jid] = rad
        else:
            bus = self._robot_buses.get("right")
            if bus is not None:
                try:
                    present = bus.sync_read("Present_Position")
                except Exception:
                    present = {}
                for i, joint_name in enumerate(SO101_JOINT_NAMES):
                    if joint_name in present:
                        rad = self._motor_to_radians(joint_name, present[joint_name])
                        if joint_name in self.placo_robot.model.names:
                            jid = self.placo_robot.model.getJointId(joint_name)
                            if jid < len(placo_q):
                                placo_q[jid] = rad

        self.placo_robot.state.q = placo_q
        self.placo_robot.update_kinematics()

    def _send_command(self):
        """Send Placo IK joint targets to hardware."""
        for side, bus in self._robot_buses.items():
            goal = {}
            for joint_name in SO101_JOINT_NAMES:
                motor_name = f"{side}_{joint_name}" if self._mode == "dual" else joint_name
                placo_joint = f"{side}_{joint_name}" if self._mode == "dual" else joint_name

                if placo_joint in self.placo_robot.model.names:
                    jid = self.placo_robot.model.getJointId(placo_joint)
                    rad = float(self.placo_robot.state.q[jid])
                else:
                    continue

                motor_val = self._radians_to_motor(joint_name, rad)
                goal[motor_name] = motor_val

            # Safety clamping
            if self._max_relative_target is not None:
                try:
                    present = bus.sync_read("Present_Position")
                except Exception:
                    present = {}
                for motor_name, g in list(goal.items()):
                    p = present.get(motor_name)
                    if p is not None:
                        delta = abs(g - p)
                        if delta > self._max_relative_target:
                            goal[motor_name] = p + np.sign(g - p) * self._max_relative_target

            try:
                bus.sync_write("Goal_Position", goal)
            except Exception as e:
                print(f"[SO101] Error sending command to {side} arm: {e}")

    def _get_robot_state_for_logging(self) -> Dict:
        """Return robot state for .pkl recording."""
        entry = {"qpos": {}, "qpos_des": {}, "gripper_target": {}}

        for side in ["right"] if self._mode == "single" else ["left", "right"]:
            bus = self._robot_buses.get(side)
            qpos = []
            qpos_des = []
            for joint_name in SO101_JOINT_NAMES:
                motor_name = f"{side}_{joint_name}" if self._mode == "dual" else joint_name
                placo_joint = f"{side}_{joint_name}" if self._mode == "dual" else joint_name
                # Actual position
                if bus is not None:
                    try:
                        present = bus.sync_read("Present_Position")
                        motor_val = present.get(motor_name, 0.0)
                    except Exception:
                        motor_val = 0.0
                    qpos.append(self._motor_to_radians(joint_name, motor_val))
                else:
                    qpos.append(0.0)
                # Desired position
                if placo_joint in self.placo_robot.model.names:
                    jid = self.placo_robot.model.getJointId(placo_joint)
                    qpos_des.append(float(self.placo_robot.state.q[jid]))
                else:
                    qpos_des.append(0.0)

            entry["qpos"][f"{side}_arm"] = np.array(qpos, dtype=np.float32)
            entry["qpos_des"][f"{side}_arm"] = np.array(qpos_des, dtype=np.float32)

            # Gripper target from XR trigger
            gripper_key = f"{side}_hand"
            if gripper_key in self.gripper_pos_target:
                entry["gripper_target"][f"{side}_arm"] = self.gripper_pos_target[gripper_key].copy()

        return entry

    def _get_camera_frame_for_logging(self) -> Dict:
        """Return compressed camera frames for logging."""
        if not self.camera_interface:
            return {}
        return self.camera_interface.get_compressed_frames()

    def _shutdown_robot(self):
        """Graceful shutdown."""
        for side, bus in self._robot_buses.items():
            try:
                bus.disconnect(disable_torque=True)
                print(f"[SO101] {side} arm disconnected.")
            except Exception as e:
                print(f"[SO101] Error disconnecting {side} arm: {e}")

    # ============================================================
    # Unit conversion helpers
    # ============================================================

    def _radians_to_motor(self, joint_name: str, rad: float) -> float:
        """Convert Placo IK output (radians) to motor units."""
        limits = SO101_JOINT_LIMITS_RAD[joint_name]
        if self._use_degrees:
            return float(np.degrees(rad))
        else:
            # RANGE_M100_100: map [lower, upper] → [-100, 100]
            lower, upper = limits
            mid = (upper + lower) / 2
            half = (upper - lower) / 2
            if half == 0:
                return 0.0
            normalized = (rad - mid) / half * 100
            return float(np.clip(normalized, -100, 100))

    def _motor_to_radians(self, joint_name: str, motor_val: float) -> float:
        """Convert motor units to radians for Placo."""
        limits = SO101_JOINT_LIMITS_RAD[joint_name]
        if self._use_degrees:
            return float(np.radians(motor_val))
        else:
            lower, upper = limits
            mid = (upper + lower) / 2
            half = (upper - lower) / 2
            return float(mid + motor_val / 100.0 * half)
