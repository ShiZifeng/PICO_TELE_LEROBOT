#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("so101_leader_dual")
@dataclass
class SO101LeaderDualConfig(TeleoperatorConfig):
    left_arm_port: str
    right_arm_port: str
    left_arm_use_degrees: bool = False
    right_arm_use_degrees: bool = False
