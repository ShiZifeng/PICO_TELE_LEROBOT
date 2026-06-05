#!/usr/bin/env python

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import MotorCalibration
from lerobot.robots.robot import Robot
from lerobot.robots.so101_follower import SO101Follower
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig

from .config_so101_follower_dual import SO101FollowerDualConfig

logger = logging.getLogger(__name__)


def split_dual_calibration(
    calibration: dict[str, MotorCalibration], side: str
) -> dict[str, MotorCalibration]:
    prefix = f"{side}_"
    return {key.removeprefix(prefix): value for key, value in calibration.items() if key.startswith(prefix)}


def apply_calibration(arm: SO101Follower, calibration: dict[str, MotorCalibration]) -> None:
    arm.calibration = calibration
    arm.bus.calibration = calibration


def connect_camera_with_retries(camera, attempts: int = 5, retry_sleep_s: float = 0.5) -> None:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            camera.connect()
            return
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Failed to connect %s on attempt %s/%s: %s", camera, attempt, attempts, exc
            )
            try:
                camera.disconnect()
            except Exception:
                pass
            time.sleep(retry_sleep_s)

    raise RuntimeError(f"Failed to connect {camera} after {attempts} attempts.") from last_error


class SO101FollowerDual(Robot):
    config_class = SO101FollowerDualConfig
    name = "so101_follower_dual"

    def __init__(self, config: SO101FollowerDualConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = SO101FollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            use_degrees=config.left_arm_use_degrees,
            cameras={},
        )
        right_arm_config = SO101FollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
            max_relative_target=config.right_arm_max_relative_target,
            use_degrees=config.right_arm_use_degrees,
            cameras={},
        )

        self.left_arm = SO101Follower(left_arm_config)
        self.right_arm = SO101Follower(right_arm_config)

        if self.calibration:
            apply_calibration(self.left_arm, split_dual_calibration(self.calibration, "left"))
            apply_calibration(self.right_arm, split_dual_calibration(self.calibration, "right"))

        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.left_arm.bus.is_connected
            and self.right_arm.bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)
        for cam in self.cameras.values():
            connect_camera_with_retries(cam)

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}
        obs_dict.update({f"left_{key}": value for key, value in self.left_arm.get_observation().items()})
        obs_dict.update({f"right_{key}": value for key, value in self.right_arm.get_observation().items()})

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug("%s read %s: %.1fms", self, cam_key, dt_ms)

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        sent_left = self.left_arm.send_action(left_action)
        sent_right = self.right_arm.send_action(right_action)
        return {f"left_{key}": value for key, value in sent_left.items()} | {
            f"right_{key}": value for key, value in sent_right.items()
        }

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()
