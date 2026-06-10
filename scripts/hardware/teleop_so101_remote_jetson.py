#!/usr/bin/env python3
"""
Jetson-side SO-101 remote hardware controller.

Receives joint targets from PC (XR + Placo IK), controls SO-101 hardware
via LeRobot SO101FollowerDual (with calibration), captures cameras, and
sends observations back to PC in LeRobot-compatible format.

Usage (on Jetson):
  python teleop_so101_remote_jetson.py \
    --pc-ip 192.168.50.75 \
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
        if idx >= 8:  # arm cameras: use MJPEG to avoid USB bandwidth saturation
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
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
    parser.add_argument("--state-port", type=int, default=0, help="Port for high-frequency joint state to PC")
    parser.add_argument("--control-log", type=Path, default=None, help="JSONL path for control loop logging")
    parser.add_argument("--control-log-every-n", type=int, default=1, help="Log every Nth control tick")
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

    # ── Control log ──
    control_log = None
    if args.control_log is not None:
        import queue, threading as _th
        class _AsyncJsonlLogger:
            def __init__(self, path, max_queue=10000):
                self._q = queue.Queue(maxsize=max_queue); self._stop = _th.Event(); self._drop = 0
                self._t = _th.Thread(target=self._run, daemon=True)
            def start(self): self._t.start()
            def log(self, r):
                r["wall_time"] = time.time(); r["monotonic_time"] = time.perf_counter()
                try: self._q.put_nowait(r)
                except queue.Full: self._drop += 1
            def stop(self):
                self._stop.set(); self._t.join(timeout=2)
                if self._drop: logger.warning("Control log dropped %d", self._drop)
            def _run(self):
                with open(args.control_log, "a", buffering=1) as f:
                    while not self._stop.is_set() or not self._q.empty():
                        try: f.write(json.dumps(self._q.get(timeout=0.1), ensure_ascii=False, sort_keys=True) + "\n")
                        except queue.Empty: pass
        control_log = _AsyncJsonlLogger(args.control_log)
        control_log.start()
        logger.info("Control JSONL log: %s", args.control_log)

    # ── Setup ZMQ ──
    ctx = zmq.Context()

    target_socket = ctx.socket(zmq.SUB)
    target_socket.setsockopt(zmq.RCVHWM, 100)
    target_socket.setsockopt(zmq.CONFLATE, 1)
    try: target_socket.setsockopt(50, 1)  # ZMQ_TCP_NODELAY (pyzmq compat)
    except Exception: pass
    target_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    target_socket.connect(f"tcp://{args.pc_ip}:{args.target_port}")

    obs_socket = ctx.socket(zmq.PUSH)
    obs_socket.setsockopt(zmq.SNDHWM, 1)
    obs_socket.connect(f"tcp://{args.pc_ip}:{args.obs_port}")

    state_socket = None
    if args.state_port > 0:
        state_socket = ctx.socket(zmq.PUSH)
        state_socket.setsockopt(zmq.SNDHWM, 1)
        state_socket.connect(f"tcp://{args.pc_ip}:{args.state_port}")

    cmd_socket = ctx.socket(zmq.SUB)
    cmd_socket.setsockopt(zmq.RCVHWM, 1)
    cmd_socket.setsockopt(zmq.CONFLATE, 1)
    cmd_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    cmd_socket.connect(f"tcp://{args.pc_ip}:{args.command_port}")

    # ── Control thread (60Hz): send_action + read obs + state_tx + log ──
    import threading as _th2
    _control_state = {"action": None, "observation": None, "obs_t": 0.0, "seq": 0}
    _control_lock = _th2.Lock()
    _control_stop = _th2.Event()

    def _control_loop():
        period = 1.0 / args.control_fps
        next_tick = time.perf_counter()
        seq, tx_seq = 0, 0
        prev_keys = None; tx_updates = 0; tx_msgs = 0
        last_stats_t = time.perf_counter()
        while not _control_stop.is_set():
            t0 = time.perf_counter()
            # Drain ZMQ targets
            try:
                while True:
                    t = target_socket.recv_pyobj(flags=zmq.NOBLOCK)
                    if isinstance(t, dict) and not t.get("__hold__"):
                        action = {f"{k}.pos": float(v) for k, v in t.items()}
                        with _control_lock: _control_state["action"] = action
                        cur = tuple(sorted(t.items()))
                        if cur != prev_keys: tx_updates += 1; prev_keys = cur
                        tx_msgs += 1; tx_seq += 1
                        if control_log is not None:
                            control_log.log({"event":"target_rx","seq":tx_seq,"target_raw":t,"action_target":action,"recv_ms":(time.perf_counter()-t0)*1000})
                    elif isinstance(t, dict) and t.get("__hold__"):
                        with _control_lock: _control_state["action"] = None
                        prev_keys = None; tx_msgs += 1; tx_seq += 1
            except zmq.Again: pass

            action = None
            with _control_lock: action = _control_state.get("action")
            seq += 1
            try:
                sm, om = 0.0, 0.0
                if action is not None:
                    t0 = time.perf_counter(); robot.send_action(action); sm = (time.perf_counter()-t0)*1000
                t0 = time.perf_counter(); obs = robot.get_observation(); om = (time.perf_counter()-t0)*1000
                obs_t = time.perf_counter()
                with _control_lock:
                    _control_state["observation"] = obs; _control_state["obs_t"] = obs_t; _control_state["seq"] = seq
                if state_socket is not None:
                    sf = {f"observation.{n}": v for n, v in obs.items()}
                    if action is not None:
                        for n, v in action.items(): sf[f"action.{n}"] = v
                    try: state_socket.send_pyobj({"type":"state","frame":sf,"t":obs_t}, flags=zmq.NOBLOCK)
                    except Exception: pass
                if control_log is not None and args.control_log_every_n > 0 and seq % args.control_log_every_n == 0:
                    control_log.log({"event":"control_tick","seq":seq,"tick_start":t0,"action_sent":action,"state":obs,"send_ms":sm,"obs_ms":om})
            except Exception as e:
                logger.warning("Control loop I/O error: %s", e, exc_info=True)

            # Stats
            now = time.perf_counter()
            if now - last_stats_t >= 2.0:
                dt = now - last_stats_t
                logger.info("Control stats: motor=%.1f/s targets=%d/s updates=%d/s",
                            seq/dt, tx_msgs/dt, tx_updates/dt)
                seq = 0; tx_msgs = 0; tx_updates = 0; last_stats_t = now

            next_tick += period
            sleep_s = next_tick - time.perf_counter()
            if sleep_s <= 0: next_tick = time.perf_counter()
            else: _control_stop.wait(sleep_s)

    _control_thread = _th2.Thread(target=_control_loop, daemon=True, name="control")
    _control_thread.start()

    # ── Build features for setup message ──
    # Use aggregated "action" / "observation.state" format aligned with lerobot's
    # hw_to_dataset_features, so downstream training pipelines see the standard schema.
    joint_names = []
    for side in sides:
        for joint in SO101_JOINT_NAMES:
            joint_names.append(f"{side}_{joint}.pos")

    features = {}
    features["action"] = {"dtype": "float32", "shape": (len(joint_names),), "names": joint_names}
    features["observation.state"] = {"dtype": "float32", "shape": (len(joint_names),), "names": joint_names}
    for name, cfg in camera_configs.items():
        h, w = cfg.get("height", 480), cfg.get("width", 640)
        features[f"observation.images.{name}"] = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}

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
    period = 1.0 / args.fps

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

            # Get latest observation from control thread
            with _control_lock:
                obs = dict(_control_state["observation"]) if _control_state["observation"] is not None else None
                action = _control_state.get("action")
            if obs is None:
                busy_wait(max(0, period - (time.perf_counter() - loop_start)))
                continue
            # Update latest_target for setup resend logic
            if action is not None:
                latest_target = {k.replace(".pos", ""): v for k, v in action.items()}
            else:
                latest_target = None

            frame = {}
            # Build aggregated observation.state (aligned with lerobot format)
            obs_values = [obs.get(name, 0.0) for name in joint_names]
            frame["observation.state"] = np.array(obs_values, dtype=np.float32)
            # Build aggregated action (latest PC IK target), fall back to observation
            action_values = [
                action[name] if (action is not None and name in action) else obs.get(name, 0.0)
                for name in joint_names
            ]
            frame["action"] = np.array(action_values, dtype=np.float32)
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
        _control_stop.set()
        _control_thread.join(timeout=2)
        robot.disconnect()
        for cap in caps.values(): cap.release()
        if state_socket is not None: state_socket.close(0)
        obs_socket.close(0); target_socket.close(0); cmd_socket.close(0)
        ctx.term()
        if control_log is not None: control_log.stop()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
