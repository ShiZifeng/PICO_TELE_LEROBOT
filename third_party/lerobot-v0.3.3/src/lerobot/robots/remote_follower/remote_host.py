#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import (  # noqa: F401
    bi_so100_follower,
    hope_jr,
    koch_follower,
    lekiwi,
    so100_follower,
    so101_follower,
    so101_follower_dual,
)
from lerobot.robots.config import RobotConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging


@dataclass
class RemoteFollowerHostConfig:
    robot: RobotConfig
    port_zmq_cmd: int = 5560
    port_zmq_observations: int = 5561
    fps: int = 30
    jpeg_quality: int = 80
    compress_images: bool = True
    watchdog_timeout_ms: int = 500


def encode_observation_images(
    observation: dict[str, Any], camera_keys: set[str], jpeg_quality: int
) -> dict[str, Any]:
    encoded = dict(observation)
    for key in camera_keys:
        if key not in encoded:
            continue
        ok, buffer = cv2.imencode(".jpg", encoded[key], [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            raise RuntimeError(f"Failed to JPEG encode camera frame '{key}'.")
        encoded[key] = {
            "__remote_image_encoding__": "jpg",
            "shape": encoded[key].shape,
            "data": buffer,
        }
    return encoded


def decode_observation_images(observation: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(observation)
    for key, value in observation.items():
        if not isinstance(value, dict) or value.get("__remote_image_encoding__") != "jpg":
            continue
        frame = cv2.imdecode(np.frombuffer(value["data"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to JPEG decode camera frame '{key}'.")
        decoded[key] = frame
    return decoded


@parser.wrap()
def main(cfg: RemoteFollowerHostConfig) -> None:
    import zmq

    init_logging()
    logging.info(
        "Starting remote follower host for robot=%s on cmd_port=%s observation_port=%s fps=%s",
        cfg.robot.type,
        cfg.port_zmq_cmd,
        cfg.port_zmq_observations,
        cfg.fps,
    )

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    camera_keys = {key for key, ft in robot.observation_features.items() if isinstance(ft, tuple)}

    context = zmq.Context()
    cmd_socket = context.socket(zmq.REP)
    cmd_socket.bind(f"tcp://*:{cfg.port_zmq_cmd}")
    observation_socket = context.socket(zmq.PUSH)
    observation_socket.setsockopt(zmq.CONFLATE, 1)
    observation_socket.bind(f"tcp://*:{cfg.port_zmq_observations}")

    poller = zmq.Poller()
    poller.register(cmd_socket, zmq.POLLIN)
    last_cmd_time = time.perf_counter()

    try:
        while True:
            start = time.perf_counter()

            socks = dict(poller.poll(0))
            if cmd_socket in socks:
                action = cmd_socket.recv_pyobj()
                sent_action = robot.send_action(action)
                cmd_socket.send_pyobj(sent_action)
                last_cmd_time = time.perf_counter()

            if time.perf_counter() - last_cmd_time > cfg.watchdog_timeout_ms / 1000:
                if hasattr(robot, "stop_base"):
                    robot.stop_base()
                last_cmd_time = time.perf_counter()

            observation = robot.get_observation()
            if cfg.compress_images:
                observation = encode_observation_images(observation, camera_keys, cfg.jpeg_quality)
            try:
                observation_socket.send_pyobj(observation, flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.debug("Dropping observation because no remote recorder is connected.")

            busy_wait(1 / cfg.fps - (time.perf_counter() - start))
    except KeyboardInterrupt:
        logging.info("Stopping remote follower host.")
    finally:
        observation_socket.close(0)
        cmd_socket.close(0)
        context.term()
        robot.disconnect()


if __name__ == "__main__":
    main()
