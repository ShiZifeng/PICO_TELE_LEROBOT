#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import logging
import sys
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.orbbec.configuration_orbbec import OrbbecCameraConfig
from lerobot.cameras.utils import get_cv2_rotation
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)


class OrbbecCamera(Camera):
    def __init__(self, config: OrbbecCameraConfig):
        super().__init__(config)
        self.config = config
        self.serial_number = config.serial_number
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
        self.timeout_ms = config.timeout_ms
        self.rotation = get_cv2_rotation(config.rotation)

        self.pipeline = None
        self.pipeline_config = None
        self._sdk = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock = Lock()
        self.latest_frame: np.ndarray | None = None
        self.new_frame_event = Event()

        if config.sdk_path is not None:
            sdk_path = str(Path(config.sdk_path).expanduser())
            if sdk_path not in sys.path:
                sys.path.append(sdk_path)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number or 'default'})"

    @property
    def is_connected(self) -> bool:
        return self.pipeline is not None

    def _import_sdk(self):
        if self._sdk is None:
            try:
                import pyorbbecsdk as sdk
            except ImportError as exc:
                raise ImportError(
                    "pyorbbecsdk is not importable. Install it in the active environment or set "
                    "`sdk_path` to the directory containing pyorbbecsdk."
                ) from exc
            self._sdk = sdk
        return self._sdk

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        try:
            import pyorbbecsdk as sdk
        except ImportError:
            return []

        cameras = []
        context = sdk.Context()
        device_list = context.query_devices()
        for index in range(device_list.get_count()):
            device = device_list[index]
            info = device.get_device_info()
            color_profiles = []
            try:
                pipeline = sdk.Pipeline(device)
                profile_list = pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
                for profile_index in range(profile_list.get_count()):
                    profile = profile_list.get_stream_profile_by_index(profile_index)
                    if type(profile).__name__ != "VideoStreamProfile":
                        continue
                    color_profiles.append(
                        {
                            "format": str(profile.get_format()),
                            "width": profile.get_width(),
                            "height": profile.get_height(),
                            "fps": profile.get_fps(),
                        }
                    )
            except Exception as exc:
                logger.warning("Could not enumerate Orbbec color profiles for %s: %s", info.get_name(), exc)

            cameras.append(
                {
                    "name": info.get_name(),
                    "type": "Orbbec",
                    "id": info.get_serial_number(),
                    "pid": info.get_pid(),
                    "color_stream_profiles": color_profiles,
                    "default_stream_profile": color_profiles[0] if color_profiles else None,
                }
            )
        return cameras

    def _select_profile(self):
        sdk = self._import_sdk()
        profile_list = self.pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
        if self.width is not None and self.height is not None and self.fps is not None:
            try:
                return profile_list.get_video_stream_profile(
                    self.width, self.height, sdk.OBFormat.RGB, self.fps
                )
            except sdk.OBError:
                logger.warning(
                    "%s could not use RGB %sx%s@%s; falling back to default profile.",
                    self,
                    self.width,
                    self.height,
                    self.fps,
                )
        return profile_list.get_default_video_stream_profile()

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        sdk = self._import_sdk()
        if self.serial_number:
            context = sdk.Context()
            device = context.query_devices().get_device_by_serial_number(self.serial_number)
            self.pipeline = sdk.Pipeline(device)
        else:
            self.pipeline = sdk.Pipeline()
        self.pipeline_config = sdk.Config()

        profile = self._select_profile()
        self.pipeline_config.enable_stream(profile)
        self.pipeline.start(self.pipeline_config)

        if self.width is None:
            self.width = profile.get_width()
        if self.height is None:
            self.height = profile.get_height()
        if self.fps is None:
            self.fps = profile.get_fps()

        if warmup:
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.read()
                time.sleep(0.1)

        logger.info("%s connected.", self)

    def _frame_to_bgr_image(self, frame) -> np.ndarray | None:
        sdk = self._import_sdk()
        width = frame.get_width()
        height = frame.get_height()
        color_format = frame.get_format()
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)

        if color_format == sdk.OBFormat.RGB:
            image = data.reshape((height, width, 3))
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if color_format == sdk.OBFormat.BGR:
            return data.reshape((height, width, 3))
        if color_format == sdk.OBFormat.YUYV:
            image = data.reshape((height, width, 2))
            return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
        if color_format == sdk.OBFormat.UYVY:
            image = data.reshape((height, width, 2))
            return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
        if color_format == sdk.OBFormat.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        if color_format == sdk.OBFormat.NV12:
            yuv = data.reshape((height * 3 // 2, width))
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        if color_format == sdk.OBFormat.NV21:
            yuv = data.reshape((height * 3 // 2, width))
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
        if color_format == sdk.OBFormat.I420:
            yuv = data.reshape((height * 3 // 2, width))
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

        raise RuntimeError(f"Unsupported Orbbec color format: {color_format}")

    def _postprocess_image(self, image: np.ndarray, color_mode: ColorMode | None = None) -> np.ndarray:
        requested_color_mode = self.color_mode if color_mode is None else color_mode
        if requested_color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(f"Invalid color mode '{requested_color_mode}'.")

        h, w, c = image.shape
        if c != 3:
            raise RuntimeError(f"{self} frame channels={c}; expected 3.")
        if self.width is not None and self.height is not None and (w != self.width or h != self.height):
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured width={self.width} or height={self.height}."
            )

        processed = image
        if requested_color_mode == ColorMode.RGB:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        if self.rotation is not None:
            processed = cv2.rotate(processed, self.rotation)
        return processed

    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        frames = self.pipeline.wait_for_frames(self.timeout_ms)
        if frames is None:
            raise TimeoutError(f"Timed out waiting for frames from {self}.")
        color_frame = frames.get_color_frame()
        if color_frame is None:
            raise RuntimeError(f"{self} did not return a color frame.")

        image = self._frame_to_bgr_image(color_frame)
        if image is None:
            raise RuntimeError(f"{self} failed to convert color frame.")
        return self._postprocess_image(image, color_mode)

    def _read_loop(self):
        while not self.stop_event.is_set():
            try:
                frame = self.read()
                with self.frame_lock:
                    self.latest_frame = frame
                self.new_frame_event.set()
            except DeviceNotConnectedError:
                break
            except Exception as exc:
                logger.warning("Error reading frame in background thread for %s: %s", self, exc)

    def _start_read_thread(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop", daemon=True)
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 1000) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timed out waiting for frame from {self} after {timeout_ms} ms.")

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")
        return frame

    def disconnect(self) -> None:
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is not None:
            self._stop_read_thread()
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
            self.pipeline_config = None

        logger.info("%s disconnected.", self)
