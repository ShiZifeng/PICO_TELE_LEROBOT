#!/usr/bin/env python3
"""
SO-101 Hardware Teleoperation with XR + Placo IK

Connects PICO XR headset teleop with real SO-101 hardware via STS3215 servos.

Usage:
  # Single arm (right hand)
  python scripts/hardware/teleop_so101_hardware.py \\
    --ports '{"right": "/dev/right_follower"}'

  # Dual arm
  python scripts/hardware/teleop_so101_hardware.py \\
    --mode dual \\
    --ports '{"left": "/dev/left_follower", "right": "/dev/right_follower"}'

  # With cameras
  python scripts/hardware/teleop_so101_hardware.py \\
    --ports '{"right": "/dev/right_follower"}' \\
    --enable-camera \\
    --camera-configs '{"wrist": {"index": 0, "width": 640, "height": 480, "fps": 30}}'

Controls:
  - Right/Left grip: activate arm tracking
  - Right/Left trigger: close gripper
  - B button: start/stop recording
  - Right axis click: discard recording
  - Ctrl+C: exit
"""

import json
import os
from typing import Any, Dict, Optional

import tyro

from xrobotoolkit_teleop.hardware.so101_hardware_teleop_controller import (
    SO101HardwareTeleopController,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


def main(
    mode: str = "single",  # "single" or "dual"
    ports_json: str = '{"right": "/dev/right_follower"}',
    robot_urdf_path: Optional[str] = None,
    camera_configs_json: str = "{}",
    scale_factor: float = 1.0,
    control_rate_hz: int = 50,
    enable_log_data: bool = True,
    log_dir: str = "logs/so101",
    log_freq: float = 30,
    enable_camera: bool = False,
    camera_fps: int = 30,
    use_degrees: bool = True,
    max_relative_target: Optional[float] = None,
):
    """
    Main function to run SO-101 hardware teleoperation.

    Args:
        mode: "single" (right arm only) or "dual" (left + right arms).
        ports_json: JSON string mapping arm side to serial port.
            Example: '{"right": "/dev/right_follower"}' or
                     '{"left": "/dev/left_follower", "right": "/dev/right_follower"}'.
        robot_urdf_path: Path to SO-101 URDF. Auto-detected if not provided.
        camera_configs_json: JSON string for camera configs.
            Example: '{"wrist": {"index": 0, "width": 640, "height": 480, "fps": 30}}'.
        scale_factor: XR motion scaling factor.
        control_rate_hz: IK + control loop frequency (Hz).
        enable_log_data: Enable .pkl data recording.
        log_dir: Directory for recorded .pkl files.
        log_freq: Recording frequency (Hz).
        enable_camera: Enable camera capture.
        camera_fps: Camera capture FPS.
        use_degrees: Use degrees for motor commands (True) or RANGE_M100_100 (False).
        max_relative_target: Safety clamp for per-step motor movement (in motor units).
            e.g., 5.0 limits each servo to move at most 5 degrees (or 5 in RANGE_M100_100) per step.
    """
    ports = json.loads(ports_json)
    camera_configs = json.loads(camera_configs_json) if enable_camera else {}

    # Auto-detect URDF path
    if robot_urdf_path is None:
        if mode == "dual":
            robot_urdf_path = os.path.join(ASSET_PATH, "so101", "dual_so101.urdf")
        else:
            robot_urdf_path = os.path.join(ASSET_PATH, "so101", "so101_new_calib.urdf")

    print("=" * 60)
    print("SO-101 Hardware Teleoperation")
    print("=" * 60)
    print(f"  Mode: {mode}")
    print(f"  Ports: {ports}")
    print(f"  URDF: {robot_urdf_path}")
    print(f"  Scale factor: {scale_factor}")
    print(f"  Control rate: {control_rate_hz} Hz")
    print(f"  Log data: {enable_log_data} → {log_dir}")
    print(f"  Log freq: {log_freq} Hz")
    print(f"  Cameras: {list(camera_configs.keys()) if camera_configs else 'disabled'}")
    print(f"  Motor units: {'degrees' if use_degrees else 'RANGE_M100_100'}")
    if max_relative_target:
        print(f"  Safety clamp: ±{max_relative_target}")
    print("=" * 60)

    controller = SO101HardwareTeleopController(
        robot_urdf_path=robot_urdf_path,
        mode=mode,
        ports=ports,
        camera_configs=camera_configs,
        scale_factor=scale_factor,
        control_rate_hz=control_rate_hz,
        enable_log_data=enable_log_data,
        log_dir=log_dir,
        log_freq=log_freq,
        enable_camera=enable_camera,
        camera_fps=camera_fps,
        use_degrees=use_degrees,
        max_relative_target=max_relative_target,
    )

    print("\nStarting teleoperation...")
    print("  - Hold grip button to activate arm tracking")
    print("  - Press trigger to close gripper")
    print("  - Press B to start/stop recording")
    print("  - Press right axis click to discard recording")
    print("  - Press Ctrl+C to exit\n")

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
