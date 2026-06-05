#!/usr/bin/env python

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
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    lekiwi,
    make_robot_from_config,
    so100_follower,
    so101_follower,
    so101_follower_dual,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    bi_so100_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
    so101_leader_dual,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging


@dataclass
class RemoteRecordSenderConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig
    receiver_ip: str
    port: int = 5570
    command_port: int = 5571
    fps: int = 15
    control_fps: int = 60
    interpolate_actions: bool = True
    num_episodes: int = 1
    episode_time_s: float = 10
    reset_time_s: float = 5
    task: str = "remote recording"
    jpeg_quality: int = 80
    keyboard_control: bool = False


def encode_frame_images(frame: dict[str, Any], features: dict[str, dict], jpeg_quality: int) -> dict[str, Any]:
    encoded = {}
    for key, value in frame.items():
        if features[key]["dtype"] not in ["image", "video"]:
            encoded[key] = value
            continue
        image = np.ascontiguousarray(value)
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            raise RuntimeError(f"Failed to JPEG encode frame key '{key}'.")
        encoded[key] = {"__remote_image_encoding__": "jpg", "data": buffer}
    return encoded


def interpolate_action(
    previous_action: dict[str, Any] | None, target_action: dict[str, Any], alpha: float
) -> dict[str, Any]:
    if previous_action is None:
        return target_action

    action = {}
    for key, target_value in target_action.items():
        previous_value = previous_action.get(key, target_value)
        try:
            action[key] = previous_value + (target_value - previous_value) * alpha
        except TypeError:
            action[key] = target_value
    return action


def send_interpolated_action(
    robot,
    previous_action: dict[str, Any] | None,
    target_action: dict[str, Any],
    control_fps: int,
    duration_s: float,
    interpolate: bool,
) -> dict[str, Any]:
    if not interpolate or previous_action is None or duration_s <= 0 or control_fps <= 0:
        return robot.send_action(target_action)

    num_steps = max(1, int(round(duration_s * control_fps)))
    sent_action = target_action
    step_s = duration_s / num_steps
    for step_idx in range(1, num_steps + 1):
        step_start = time.perf_counter()
        alpha = step_idx / num_steps
        sent_action = robot.send_action(interpolate_action(previous_action, target_action, alpha))
        busy_wait(step_s - (time.perf_counter() - step_start))
    return sent_action


def teleop_control(robot, teleop, fps: int, control_fps: int, duration_s: float, interpolate: bool) -> None:
    previous_action = None
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s:
        loop_start = time.perf_counter()
        action = teleop.get_action()
        previous_action = send_interpolated_action(
            robot,
            previous_action,
            action,
            control_fps=control_fps,
            duration_s=max(0, 1 / fps - (time.perf_counter() - loop_start)),
            interpolate=interpolate,
        )
        busy_wait(1 / fps - (time.perf_counter() - loop_start))


def stream_frame(
    robot,
    teleop,
    dataset_features: dict[str, dict],
    jpeg_quality: int,
    previous_action: dict[str, Any] | None,
    control_fps: int,
    control_duration_s: float,
    interpolate_actions: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    observation = robot.get_observation()
    action = teleop.get_action()
    sent_action = send_interpolated_action(
        robot,
        previous_action,
        action,
        control_fps=control_fps,
        duration_s=control_duration_s,
        interpolate=interpolate_actions,
    )

    observation_frame = build_dataset_frame(dataset_features, observation, prefix="observation")
    action_frame = build_dataset_frame(dataset_features, sent_action, prefix="action")
    frame = encode_frame_images({**observation_frame, **action_frame}, dataset_features, jpeg_quality)
    return frame, sent_action


@parser.wrap()
def main(cfg: RemoteRecordSenderConfig) -> None:
    import zmq

    init_logging()
    logging.info("Starting remote record sender to %s:%s", cfg.receiver_ip, cfg.port)
    logging.info(
        "Remote sender sync fps=%s, local control_fps=%s, interpolate_actions=%s",
        cfg.fps,
        cfg.control_fps,
        cfg.interpolate_actions,
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    action_features = hw_to_dataset_features(robot.action_features, "action", use_video=True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
    dataset_features = {**action_features, **obs_features}

    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect(f"tcp://{cfg.receiver_ip}:{cfg.port}")

    command_socket = None
    if cfg.keyboard_control:
        command_socket = context.socket(zmq.SUB)
        command_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        command_socket.connect(f"tcp://{cfg.receiver_ip}:{cfg.command_port}")

    socket.send_pyobj(
        {
            "type": "setup",
            "fps": cfg.fps,
            "features": dataset_features,
            "robot_type": robot.name,
            "task": cfg.task,
        }
    )

    robot.connect()
    teleop.connect()

    try:
        previous_action = None
        if cfg.keyboard_control:
            logging.info(
                "Streaming frames for PC keyboard-controlled recording. Stop from PC with q/Esc."
            )
            while True:
                loop_start = time.perf_counter()

                if command_socket is not None:
                    try:
                        command = command_socket.recv_string(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        command = None
                    if command == "stop":
                        break

                control_duration_s = max(0, 1 / cfg.fps - (time.perf_counter() - loop_start))
                frame, previous_action = stream_frame(
                    robot,
                    teleop,
                    dataset_features,
                    cfg.jpeg_quality,
                    previous_action=previous_action,
                    control_fps=cfg.control_fps,
                    control_duration_s=control_duration_s,
                    interpolate_actions=cfg.interpolate_actions,
                )
                socket.send_pyobj({"type": "frame", "frame": frame})

                busy_wait(1 / cfg.fps - (time.perf_counter() - loop_start))
        else:
            for episode_idx in range(cfg.num_episodes):
                logging.info("Recording remote episode %s", episode_idx)
                previous_action = None
                start = time.perf_counter()
                while time.perf_counter() - start < cfg.episode_time_s:
                    loop_start = time.perf_counter()

                    control_duration_s = max(0, 1 / cfg.fps - (time.perf_counter() - loop_start))
                    frame, previous_action = stream_frame(
                        robot,
                        teleop,
                        dataset_features,
                        cfg.jpeg_quality,
                        previous_action=previous_action,
                        control_fps=cfg.control_fps,
                        control_duration_s=control_duration_s,
                        interpolate_actions=cfg.interpolate_actions,
                    )
                    socket.send_pyobj({"type": "frame", "frame": frame})

                    busy_wait(1 / cfg.fps - (time.perf_counter() - loop_start))

                socket.send_pyobj({"type": "episode_end"})

                if episode_idx < cfg.num_episodes - 1 and cfg.reset_time_s > 0:
                    logging.info("Resetting for %.1fs", cfg.reset_time_s)
                    teleop_control(
                        robot,
                        teleop,
                        fps=cfg.fps,
                        control_fps=cfg.control_fps,
                        duration_s=cfg.reset_time_s,
                        interpolate=cfg.interpolate_actions,
                    )

        socket.send_pyobj({"type": "done"})
    except KeyboardInterrupt:
        socket.send_pyobj({"type": "done"})
    finally:
        robot.disconnect()
        teleop.disconnect()
        if command_socket is not None:
            command_socket.close(0)
        socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
