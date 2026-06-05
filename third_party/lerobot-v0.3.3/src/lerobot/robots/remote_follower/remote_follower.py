#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging
from typing import Any

import cv2
import numpy as np

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.robots.robot import Robot

from .config_remote_follower import RemoteFollowerConfig


SO_ARM_MOTOR_FEATURES = {
    "shoulder_pan.pos": float,
    "shoulder_lift.pos": float,
    "elbow_flex.pos": float,
    "wrist_flex.pos": float,
    "wrist_roll.pos": float,
    "gripper.pos": float,
}


class RemoteFollower(Robot):
    config_class = RemoteFollowerConfig
    name = "remote_follower"

    def __init__(self, config: RemoteFollowerConfig):
        import zmq

        super().__init__(config)
        self.config = config
        self._zmq = zmq
        self.robot_type = config.remote_robot_type

        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None
        self._is_connected = False
        self._last_observation = None

    @property
    def observation_features(self) -> dict:
        return {**self._motor_features, **self._camera_features}

    @property
    def action_features(self) -> dict:
        return self._motor_features

    @property
    def _motor_features(self) -> dict[str, type]:
        if self.config.remote_robot_type in {"so100_follower", "so101_follower"}:
            return SO_ARM_MOTOR_FEATURES
        raise ValueError(
            f"remote_follower does not know how to build features for "
            f"remote_robot_type={self.config.remote_robot_type!r}."
        )

    @property
    def _camera_features(self) -> dict[str, tuple[int, int, int]]:
        return {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        zmq = self._zmq
        self.zmq_context = zmq.Context()

        self.zmq_cmd_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_cmd_socket.setsockopt(zmq.SNDTIMEO, self.config.command_timeout_ms)
        self.zmq_cmd_socket.setsockopt(zmq.RCVTIMEO, self.config.command_timeout_ms)
        self.zmq_cmd_socket.connect(f"tcp://{self.config.remote_ip}:{self.config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.setsockopt(zmq.RCVTIMEO, self.config.observation_timeout_ms)
        self.zmq_observation_socket.connect(
            f"tcp://{self.config.remote_ip}:{self.config.port_zmq_observations}"
        )

        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)
        socks = dict(poller.poll(self.config.connect_timeout_s * 1000))
        if self.zmq_observation_socket not in socks:
            self.disconnect()
            raise DeviceNotConnectedError(
                f"Timed out waiting for remote follower observations from {self.config.remote_ip}."
            )

        self._is_connected = True
        self._last_observation = self._decode_observation_images(self.zmq_observation_socket.recv_pyobj())

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def _decode_observation_images(self, observation: dict[str, Any]) -> dict[str, Any]:
        decoded = dict(observation)
        for key, value in observation.items():
            if not isinstance(value, dict) or value.get("__remote_image_encoding__") != "jpg":
                continue
            frame = cv2.imdecode(np.frombuffer(value["data"], dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to JPEG decode camera frame '{key}'.")
            decoded[key] = frame
        return decoded

    def get_observation(self) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError("RemoteFollower is not connected.")

        try:
            while True:
                observation = self.zmq_observation_socket.recv_pyobj(flags=self._zmq.NOBLOCK)
                self._last_observation = self._decode_observation_images(observation)
        except self._zmq.Again:
            pass

        if self._last_observation is None:
            raise DeviceNotConnectedError("No observation received from remote follower.")

        return self._last_observation

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._is_connected:
            raise DeviceNotConnectedError("RemoteFollower is not connected.")

        try:
            self.zmq_cmd_socket.send_pyobj(action)
            return self.zmq_cmd_socket.recv_pyobj()
        except self._zmq.Again as exc:
            logging.error("Timed out sending action to remote follower.")
            raise DeviceNotConnectedError("Timed out sending action to remote follower.") from exc

    def disconnect(self) -> None:
        if self.zmq_observation_socket is not None:
            self.zmq_observation_socket.close(0)
            self.zmq_observation_socket = None
        if self.zmq_cmd_socket is not None:
            self.zmq_cmd_socket.close(0)
            self.zmq_cmd_socket = None
        if self.zmq_context is not None:
            self.zmq_context.term()
            self.zmq_context = None
        self._is_connected = False
