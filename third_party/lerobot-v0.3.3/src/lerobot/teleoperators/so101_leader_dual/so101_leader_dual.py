#!/usr/bin/env python

from functools import cached_property
from typing import Any

from lerobot.motors import MotorCalibration
from lerobot.teleoperators.so101_leader import SO101Leader
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.teleoperator import Teleoperator

from .config_so101_leader_dual import SO101LeaderDualConfig


def split_dual_calibration(
    calibration: dict[str, MotorCalibration], side: str
) -> dict[str, MotorCalibration]:
    prefix = f"{side}_"
    return {key.removeprefix(prefix): value for key, value in calibration.items() if key.startswith(prefix)}


def apply_calibration(arm: SO101Leader, calibration: dict[str, MotorCalibration]) -> None:
    arm.calibration = calibration
    arm.bus.calibration = calibration


class SO101LeaderDual(Teleoperator):
    config_class = SO101LeaderDualConfig
    name = "so101_leader_dual"

    def __init__(self, config: SO101LeaderDualConfig):
        super().__init__(config)
        self.config = config

        self.left_arm = SO101Leader(
            SO101LeaderConfig(
                id=f"{config.id}_left" if config.id else None,
                calibration_dir=config.calibration_dir,
                port=config.left_arm_port,
                use_degrees=config.left_arm_use_degrees,
            )
        )
        self.right_arm = SO101Leader(
            SO101LeaderConfig(
                id=f"{config.id}_right" if config.id else None,
                calibration_dir=config.calibration_dir,
                port=config.right_arm_port,
                use_degrees=config.right_arm_use_degrees,
            )
        )

        if self.calibration:
            apply_calibration(self.left_arm, split_dual_calibration(self.calibration, "left"))
            apply_calibration(self.right_arm, split_dual_calibration(self.calibration, "right"))

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

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

    def get_action(self) -> dict[str, Any]:
        action = {}
        action.update({f"left_{key}": value for key, value in self.left_arm.get_action().items()})
        action.update({f"right_{key}": value for key, value in self.right_arm.get_action().items()})
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        left_feedback = {
            key.removeprefix("left_"): value for key, value in feedback.items() if key.startswith("left_")
        }
        right_feedback = {
            key.removeprefix("right_"): value for key, value in feedback.items() if key.startswith("right_")
        }
        if left_feedback:
            self.left_arm.send_feedback(left_feedback)
        if right_feedback:
            self.right_arm.send_feedback(right_feedback)

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
