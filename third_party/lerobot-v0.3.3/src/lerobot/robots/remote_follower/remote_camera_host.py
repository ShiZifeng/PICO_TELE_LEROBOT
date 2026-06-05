#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.configs import parser
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging


@dataclass
class RemoteCameraHostConfig:
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    port_zmq_observations: int = 5561
    fps: int = 25
    read_timeout_ms: int = 1000
    jpeg_quality: int = 80
    compress_images: bool = True


def encode_images(observation: dict[str, Any], jpeg_quality: int) -> dict[str, Any]:
    encoded = {}
    for key, image in observation.items():
        image = np.ascontiguousarray(image)
        if image.ndim == 3 and image.shape[2] == 3:
            # LeRobot cameras return RGB by default, while OpenCV JPEG encoding expects BGR.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            raise RuntimeError(f"Failed to JPEG encode camera frame '{key}'.")
        encoded[key] = {
            "__remote_image_encoding__": "jpg",
            "shape": image.shape,
            "data": buffer,
        }
    return encoded


@parser.wrap()
def main(cfg: RemoteCameraHostConfig) -> None:
    import zmq

    init_logging()
    logging.info(
        "Starting remote camera host on observation_port=%s fps=%s cameras=%s",
        cfg.port_zmq_observations,
        cfg.fps,
        list(cfg.cameras),
    )

    cameras = make_cameras_from_configs(cfg.cameras)
    for camera in cameras.values():
        camera.connect()

    context = zmq.Context()
    observation_socket = context.socket(zmq.PUSH)
    observation_socket.setsockopt(zmq.CONFLATE, 1)
    observation_socket.bind(f"tcp://*:{cfg.port_zmq_observations}")

    try:
        while True:
            start = time.perf_counter()
            observation = {}
            for name, camera in cameras.items():
                try:
                    observation[name] = camera.async_read(timeout_ms=cfg.read_timeout_ms)
                except TimeoutError as exc:
                    logging.warning("Skipping camera %s for this frame: %s", name, exc)
                except Exception as exc:
                    logging.exception("Error reading camera %s; skipping it for this frame.", name)

            if not observation:
                busy_wait(1 / cfg.fps - (time.perf_counter() - start))
                continue

            if cfg.compress_images:
                observation = encode_images(observation, cfg.jpeg_quality)

            try:
                observation_socket.send_pyobj(observation, flags=zmq.NOBLOCK)
            except zmq.Again:
                logging.debug("Dropping camera observation because no preview client is connected.")

            busy_wait(1 / cfg.fps - (time.perf_counter() - start))
    except KeyboardInterrupt:
        logging.info("Stopping remote camera host.")
    finally:
        observation_socket.close(0)
        context.term()
        for camera in cameras.values():
            if camera.is_connected:
                camera.disconnect()


if __name__ == "__main__":
    main()
