#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("remote_follower")
@dataclass
class RemoteFollowerConfig(RobotConfig):
    # IP address or hostname of the Jetson running `lerobot.robots.remote_follower.remote_host`.
    remote_ip: str

    # Real robot type running on the Jetson. This is only used on the PC to build the dataset schema.
    remote_robot_type: str = "so101_follower"

    # Camera schema must match the cameras configured on the Jetson host.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    port_zmq_cmd: int = 5560
    port_zmq_observations: int = 5561

    connect_timeout_s: int = 5
    command_timeout_ms: int = 1000
    observation_timeout_ms: int = 1000
