#!/usr/bin/env python3
"""
Jetson-side SO-101 remote hardware controller.

Receives joint targets from PC (XR + Placo IK), controls SO-101 hardware
via LeRobot SO101FollowerDual (with calibration), captures cameras, and
sends observations back to PC in LeRobot-compatible format.

Usage (on Jetson):
  python teleop_so101_remote_jetson.py \
    --pc-ip 192.168.50.101 \
    --robot-id mobile_follower_arm_dual \
    --ports '{"left": "/dev/left_follower_mobile", "right": "/dev/right_follower_mobile"}' \
    --cameras '{"left_arm": {"type": "opencv", "index": 10, ...}}' \
    --fps 60 --control-fps 60
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("so101_jetson")

SO101_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _find_lerobot_src() -> Optional[str]:
    env_path = os.environ.get("LEROBOT_PATH")
    project_root = Path(__file__).resolve().parents[2]
    bundled_path = project_root / "third_party" / "lerobot-v0.3.3"
    for p in [
        os.path.join(env_path, "src") if env_path else None,
        env_path,
        str(bundled_path / "src"),
        os.path.expanduser("~/szf_lerobot/lerobot-v0.3.3/src"),
        "/media/shizifeng/projects21/lerobot-v0.3.3/src",
    ]:
        if p and os.path.isdir(os.path.join(p, "lerobot")):
            return p
    return None


def _ensure_lerobot_path():
    p = _find_lerobot_src()
    if p and p not in sys.path:
        sys.path.insert(0, p)


_ensure_lerobot_path()


def open_cameras(camera_configs: dict) -> dict:
    caps = {}
    for name, cfg in camera_configs.items():
        idx = cfg.get("index", 0)
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.get("width", 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.get("height", 480))
        cap.set(cv2.CAP_PROP_FPS, cfg.get("fps", 30))
        caps[name] = cap
        logger.info("[camera] %s opened (index=%s, %dx%d)", name, idx,
                     cfg.get("width", 640), cfg.get("height", 480))
    return caps


def read_cameras(caps: dict) -> dict:
    frames = {}
    for name, cap in caps.items():
        ret, frame = cap.read()
        if ret:
            frames[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frames


def encode_frame(frame: dict, image_keys: list, jpeg_quality: int) -> dict:
    encoded = {}
    for key, value in frame.items():
        if key not in image_keys:
            encoded[key] = value
            continue
        img = np.ascontiguousarray(value)
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            raise RuntimeError(f"Failed to JPEG encode '{key}'.")
        encoded[key] = {"__remote_image_encoding__": "jpg", "data": buf.tobytes()}
    return encoded


def interpolate_action(prev: Optional[dict], target: dict, alpha: float) -> dict:
    if prev is None:
        return target
    out = {}
    for k, v in target.items():
        pv = prev.get(k, v)
        try:
            out[k] = pv + (v - pv) * alpha
        except TypeError:
            out[k] = v
    return out


def busy_wait(seconds: float):
    if seconds <= 0:
        return
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        time.sleep(0)


def main():
    import argparse
    import zmq

    parser = argparse.ArgumentParser("SO-101 Jetson Hardware Sender")
    parser.add_argument("--pc-ip", required=True, help="PC IP address")
    parser.add_argument("--robot-id", default="mobile_follower_arm_dual", help="LeRobot robot calibration ID")
    parser.add_argument("--ports", required=True, help='JSON: {"left": "/dev/...", "right": "/dev/..."}')
    parser.add_argument("--cameras", default="{}", help="JSON camera config dict")
    parser.add_argument("--target-port", type=int, default=5580, help="Port to receive joint targets from PC")
    parser.add_argument("--obs-port", type=int, default=5570, help="Port to send observations to PC")
    parser.add_argument("--command-port", type=int, default=5571, help="Port to receive commands from PC")
    parser.add_argument("--fps", type=int, default=60, help="Sync FPS (observation send rate)")
    parser.add_argument("--control-fps", type=int, default=60, help="Local motor control FPS")
    parser.add_argument("--interpolate-actions", action="store_true", help="Interpolate received targets locally between observation frames")
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--max-relative-target", type=float, default=None)
    parser.add_argument("--calibrate", action="store_true", help="Run calibration on connect")
    args = parser.parse_args()

    ports = json.loads(args.ports)
    camera_configs = json.loads(args.cameras)
    sides = list(ports.keys())

    # ── Setup robot via LeRobot SO101FollowerDual ──
    from lerobot.robots.so101_follower_dual.config_so101_follower_dual import SO101FollowerDualConfig
    from lerobot.robots.so101_follower_dual import SO101FollowerDual

    def _camera_config(cfg):
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        return OpenCVCameraConfig(
            index_or_path=cfg.get("index", 0),
            fps=cfg.get("fps"), width=cfg.get("width"), height=cfg.get("height"),
        )

    robot_cfg = SO101FollowerDualConfig(
        id=args.robot_id,
        calibration_dir=Path("~/.cache/huggingface/lerobot/calibration/robots/so101_follower_dual").expanduser(),
        left_arm_port=ports.get("left", ports.get("right")),
        right_arm_port=ports.get("right", ports.get("left")),
        left_arm_max_relative_target=args.max_relative_target,
        right_arm_max_relative_target=args.max_relative_target,
        cameras={},
    )

    robot = SO101FollowerDual(robot_cfg)
    robot.connect(calibrate=args.calibrate)
    logger.info("Robot connected. Action features: %s", list(robot.action_features))

    caps = open_cameras(camera_configs) if camera_configs else {}
    image_keys = [f"observation.images.{name}" for name in caps]

    # ── Setup ZMQ ──
    ctx = zmq.Context()

    target_socket = ctx.socket(zmq.SUB)
    target_socket.setsockopt(zmq.RCVHWM, 1)
    target_socket.setsockopt(zmq.CONFLATE, 1)
    target_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    target_socket.connect(f"tcp://{args.pc_ip}:{args.target_port}")

    obs_socket = ctx.socket(zmq.PUSH)
    obs_socket.setsockopt(zmq.SNDHWM, 1)
    obs_socket.connect(f"tcp://{args.pc_ip}:{args.obs_port}")

    cmd_socket = ctx.socket(zmq.SUB)
    cmd_socket.setsockopt(zmq.RCVHWM, 1)
    cmd_socket.setsockopt(zmq.CONFLATE, 1)
    cmd_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    cmd_socket.connect(f"tcp://{args.pc_ip}:{args.command_port}")

    # ── Build features for setup message ──
    features = {}
    for side in sides:
        for joint in SO101_JOINT_NAMES:
            features[f"observation.{side}_{joint}.pos"] = {"dtype": "float32", "shape": (1,), "names": None}
    for name in caps:
        features[f"observation.images.{name}"] = {"dtype": "video", "shape": (3,), "names": ["channels", "height", "width"]}

    setup_msg = {
        "type": "setup", "fps": args.fps, "features": features,
        "robot_type": "so101_follower_dual" if len(sides) == 2 else "so101_follower",
        "task": "XR teleop recording",
    }
    try:
        obs_socket.send_pyobj(setup_msg, flags=zmq.NOBLOCK)
    except zmq.Again:
        pass

    logger.info("Ready. Waiting for joint targets from PC...")

    latest_target: Optional[dict] = None
    last_setup_resend = time.perf_counter()
    previous_action: Optional[dict] = None
    period = 1.0 / args.fps
    control_period = 1.0 / args.control_fps

    try:
        while True:
            loop_start = time.perf_counter()

            # Check for stop command
            try:
                cmd = cmd_socket.recv_string(flags=zmq.NOBLOCK)
                if cmd == "stop":
                    logger.info("Received stop command from PC.")
                    break
            except zmq.Again:
                pass

            # Receive latest joint target from PC (non-blocking, take newest)
            try:
                while True:
                    target = target_socket.recv_pyobj(flags=zmq.NOBLOCK)
                    if isinstance(target, dict) and target.get("__hold__"):
                        latest_target = None
                        previous_action = None
                    else:
                        latest_target = target
            except zmq.Again:
                pass

            # Send action to hardware. Interpolation is optional; PC IK defaults to 60Hz.
            if latest_target is not None:
                # Convert PC format {left_shoulder_pan: deg} → LeRobot format {left_shoulder_pan.pos: deg}
                action = {f"{k}.pos": float(v) for k, v in latest_target.items()}

                if not args.interpolate_actions or previous_action is None:
                    try:
                        robot.send_action(action)
                    except Exception as e:
                        logger.warning("Motor write error: %s", e)
                    previous_action = action
                else:
                    remaining = max(0, period - (time.perf_counter() - loop_start))
                    num_steps = max(1, int(round(remaining * args.control_fps)))
                    step_s = remaining / num_steps

                    for step in range(1, num_steps + 1):
                        step_start = time.perf_counter()
                        alpha = step / num_steps
                        interp = interpolate_action(previous_action, action, alpha)
                        try:
                            robot.send_action(interp)
                        except Exception as e:
                            logger.warning("Motor write error: %s", e)
                        previous_action = interp
                        busy_wait(max(0, step_s - (time.perf_counter() - step_start)))

            # Build and send observation frame at fps rate
            obs = robot.get_observation()
            frame = {}
            for k, v in obs.items():
                frame[f"observation.{k}"] = v
            # Add camera frames
            images = read_cameras(caps)
            for name, img in images.items():
                frame[f"observation.images.{name}"] = img

            encoded = encode_frame(frame, image_keys, args.jpeg_quality)
            try:
                obs_socket.send_pyobj({"type": "frame", "frame": encoded}, flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

            # Re-send setup every 2s until PC acknowledges (receives targets)
            now = time.perf_counter()
            if latest_target is None and now - last_setup_resend > 2.0:
                try:
                    obs_socket.send_pyobj(setup_msg, flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
                last_setup_resend = now

            busy_wait(max(0, period - (time.perf_counter() - loop_start)))

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        obs_socket.send_pyobj({"type": "done"})
        robot.disconnect()
        for cap in caps.values():
            cap.release()
        obs_socket.close(0)
        target_socket.close(0)
        cmd_socket.close(0)
        ctx.term()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
