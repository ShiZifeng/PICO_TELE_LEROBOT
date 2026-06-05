#!/usr/bin/env python

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.datasets.utils import (
    EPISODES_PATH,
    EPISODES_STATS_PATH,
    TASKS_PATH,
    load_episodes,
    load_episodes_stats,
    load_info,
    load_tasks,
)
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data


def decode_frame_images(frame: dict[str, Any]) -> dict[str, Any]:
    decoded = {}
    for key, value in frame.items():
        if not isinstance(value, dict) or value.get("__remote_image_encoding__") != "jpg":
            decoded[key] = value
            continue
        image = cv2.imdecode(np.frombuffer(value["data"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to JPEG decode frame key '{key}'.")
        decoded[key] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return decoded


def get_image_keys(features: dict[str, dict]) -> list[str]:
    return [key for key, ft in features.items() if ft["dtype"] in ["image", "video"]]


def make_preview(frame: dict[str, Any], image_keys: list[str], status: str, target_h: int = 240) -> np.ndarray | None:
    images = []
    for key in image_keys:
        image = frame.get(key)
        if image is None:
            continue

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w = image_bgr.shape[:2]
        scale = target_h / h
        image_bgr = cv2.resize(image_bgr, (max(1, int(w * scale)), target_h))
        cv2.putText(image_bgr, key, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(image_bgr, key, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        images.append(image_bgr)

    if not images:
        return None

    preview = cv2.hconcat(images)
    status_bar = np.zeros((36, preview.shape[1], 3), dtype=np.uint8)
    cv2.putText(status_bar, status, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 2)
    return cv2.vconcat([preview, status_bar])


def frame_to_rerun_dicts(frame: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    observation = {}
    action = {}
    for key, value in frame.items():
        if key.startswith("observation.images."):
            observation[key.removeprefix("observation.images.")] = value
        elif key == "observation.state":
            observation["state"] = value
        elif key == "action":
            action["action"] = value
    return observation, action


def init_remote_keyboard_listener() -> tuple[Any, dict[str, bool]]:
    events = {"toggle_recording": False, "discard_episode": False, "stop": False}
    try:
        from pynput import keyboard
    except Exception:
        logging.warning("pynput is not available; keyboard control is disabled.")
        return None, events

    def on_press(key):
        try:
            if key == keyboard.Key.space:
                events["toggle_recording"] = True
            elif getattr(key, "char", None) == "r":
                events["discard_episode"] = True
            elif key == keyboard.Key.esc or getattr(key, "char", None) == "q":
                events["stop"] = True
        except Exception as exc:
            logging.warning("Error handling key press: %s", exc)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def strip_feature_info(features: dict[str, dict]) -> dict[str, dict]:
    return {key: {k: v for k, v in value.items() if k != "info"} for key, value in features.items()}


def check_dataset_compatible(dataset: LeRobotDataset, setup_msg: dict[str, Any]) -> None:
    expected_features = {**setup_msg["features"], **DEFAULT_FEATURES}
    mismatches = []

    if dataset.meta.robot_type != setup_msg["robot_type"]:
        mismatches.append(f"robot_type: expected {setup_msg['robot_type']}, got {dataset.meta.robot_type}")
    if dataset.fps != setup_msg["fps"]:
        mismatches.append(f"fps: expected {setup_msg['fps']}, got {dataset.fps}")
    if strip_feature_info(dataset.features) != strip_feature_info(expected_features):
        mismatches.append("features do not match the remote sender setup")

    if mismatches:
        raise ValueError("Existing dataset is not compatible with this recording:\n" + "\n".join(mismatches))


def dataset_root(repo_id: str, root: Path | None) -> Path:
    return Path(root) if root is not None else HF_LEROBOT_HOME / repo_id


def is_empty_dir(path: Path) -> bool:
    return path.is_dir() and not any(path.iterdir())


def load_local_metadata_for_recording(repo_id: str, root: Path) -> LeRobotDatasetMetadata:
    meta = LeRobotDatasetMetadata.__new__(LeRobotDatasetMetadata)
    meta.repo_id = repo_id
    meta.root = root
    meta.revision = None
    meta.info = load_info(root)
    meta.tasks, meta.task_to_task_index = load_tasks(root) if (root / TASKS_PATH).is_file() else ({}, {})
    meta.episodes = load_episodes(root) if (root / EPISODES_PATH).is_file() else {}
    meta.episodes_stats = (
        load_episodes_stats(root) if (root / EPISODES_STATS_PATH).is_file() else {}
    )
    meta.stats = aggregate_stats(list(meta.episodes_stats.values())) if meta.episodes_stats else {}
    return meta


def load_local_dataset_for_recording(args: argparse.Namespace) -> LeRobotDataset:
    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.meta = load_local_metadata_for_recording(args.repo_id, args.root)
    dataset.repo_id = dataset.meta.repo_id
    dataset.root = dataset.meta.root
    dataset.revision = dataset.meta.revision
    dataset.tolerance_s = 1e-4
    dataset.image_writer = None
    dataset.batch_encoding_size = args.batch_encoding_size
    dataset.episodes_since_last_encoding = 0
    dataset.episodes = None
    dataset.hf_dataset = dataset.create_hf_dataset()
    dataset.image_transforms = None
    dataset.delta_timestamps = None
    dataset.delta_indices = None
    dataset.episode_data_index = None
    dataset.video_backend = None

    if args.image_writer_processes or args.image_writer_threads:
        dataset.start_image_writer(args.image_writer_processes, args.image_writer_threads)
    dataset.episode_buffer = dataset.create_episode_buffer()
    return dataset


def create_or_resume_dataset(args: argparse.Namespace, setup_msg: dict[str, Any]) -> LeRobotDataset:
    root = dataset_root(args.repo_id, args.root)

    if root.exists():
        if not (root / "meta/info.json").is_file():
            if is_empty_dir(root):
                root.rmdir()
            else:
                raise FileExistsError(
                    f"Dataset root exists but is not a complete LeRobotDataset: {root}. "
                    "Remove it, choose a different --root, or point --root to an existing dataset root."
                )
        else:
            if not args.resume:
                raise FileExistsError(f"Dataset already exists: {root}. Use --resume or a new --repo-id.")

            dataset = load_local_dataset_for_recording(args)
            check_dataset_compatible(dataset, setup_msg)
            logging.info("Dataset resumed at %s with %s existing episodes.", dataset.root, dataset.num_episodes)
            return dataset

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=setup_msg["fps"],
        root=args.root,
        robot_type=setup_msg["robot_type"],
        features=setup_msg["features"],
        use_videos=args.video,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        batch_encoding_size=args.batch_encoding_size,
    )
    logging.info("Dataset created at %s", dataset.root)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive remote robot frames and save a LeRobotDataset.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--listen-ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5570)
    parser.add_argument("--command-port", type=int, default=5571)
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-encoding-size", type=int, default=1)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keyboard-control", action="store_true")
    parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--viewer", choices=["rerun", "opencv", "none"], default="rerun")
    parser.add_argument("--preview-height", type=int, default=240)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    import zmq

    init_logging()
    logging.info("Starting remote record receiver on %s:%s", args.listen_ip, args.port)
    if args.display and args.viewer == "rerun":
        _init_rerun(session_name="remote_recording")

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://{args.listen_ip}:{args.port}")

    command_socket = context.socket(zmq.PUB)
    command_socket.bind(f"tcp://{args.listen_ip}:{args.command_port}")

    dataset = None
    task = None
    image_keys = []
    recording = not args.keyboard_control
    episode_frame_count = 0
    saved_episodes = 0
    last_fps_t = time.perf_counter()
    recv_frames = 0
    fps = 0.0
    listener = None
    events = {"toggle_recording": False, "discard_episode": False, "stop": False}
    try:
        if args.keyboard_control:
            logging.info("Keyboard controls: space=start/stop episode, r=discard current episode, q/Esc=quit.")
            listener, events = init_remote_keyboard_listener()
        while True:
            msg = socket.recv_pyobj()
            msg_type = msg.get("type")

            if msg_type == "setup":
                task = msg["task"]
                dataset = create_or_resume_dataset(args, msg)
                image_keys = get_image_keys(msg["features"])

            elif msg_type == "frame":
                if dataset is None:
                    raise RuntimeError("Received frame before setup message.")
                frame = decode_frame_images(msg["frame"])
                recv_frames += 1
                now = time.perf_counter()
                if now - last_fps_t >= 1.0:
                    fps = recv_frames / (now - last_fps_t)
                    recv_frames = 0
                    last_fps_t = now

                if recording:
                    dataset.add_frame(frame, task=task)
                    episode_frame_count += 1

                if args.display and args.viewer == "rerun":
                    observation, action = frame_to_rerun_dicts(frame)
                    log_rerun_data(observation, action)

                if args.display and args.viewer == "opencv":
                    state = "REC" if recording else "PREVIEW"
                    status = (
                        f"{state} | frames:{episode_frame_count} | saved:{saved_episodes} | "
                        f"fps:{fps:.1f} | space start/stop, r discard, q/Esc quit"
                    )
                    preview = make_preview(frame, image_keys, status, args.preview_height)
                    if preview is not None:
                        cv2.imshow("remote_record_receiver", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if args.keyboard_control and key == ord(" "):
                        events["toggle_recording"] = True
                    elif args.keyboard_control and key == ord("r"):
                        events["discard_episode"] = True
                    elif key in (ord("q"), 27):
                        events["stop"] = True

                if args.keyboard_control and events["toggle_recording"]:
                    events["toggle_recording"] = False
                    if recording:
                        if args.keyboard_control and recording and episode_frame_count > 0:
                            dataset.save_episode()
                            saved_episodes += 1
                            logging.info("Saved episode %s", dataset.num_episodes - 1)
                        recording = False
                        episode_frame_count = 0
                        if args.max_episodes is not None and saved_episodes >= args.max_episodes:
                            command_socket.send_string("stop")
                            break
                    else:
                        dataset.clear_episode_buffer()
                        episode_frame_count = 0
                        recording = True
                        logging.info("Started episode %s", dataset.num_episodes)

                if args.keyboard_control and events["discard_episode"]:
                    events["discard_episode"] = False
                    dataset.clear_episode_buffer()
                    episode_frame_count = 0
                    recording = False
                    logging.info("Discarded current episode buffer.")

                if events["stop"]:
                    if args.keyboard_control and recording and episode_frame_count > 0:
                        dataset.save_episode()
                        saved_episodes += 1
                        logging.info("Saved episode %s", dataset.num_episodes - 1)
                        episode_frame_count = 0
                    command_socket.send_string("stop")
                    break

            elif msg_type == "episode_end":
                if dataset is None:
                    raise RuntimeError("Received episode_end before setup message.")
                dataset.save_episode()
                saved_episodes += 1
                episode_frame_count = 0
                logging.info("Saved episode %s", dataset.num_episodes - 1)

            elif msg_type == "done":
                if args.keyboard_control and recording and episode_frame_count > 0:
                    dataset.save_episode()
                    saved_episodes += 1
                    logging.info("Saved episode %s", dataset.num_episodes - 1)
                    episode_frame_count = 0
                break

            else:
                raise ValueError(f"Unknown remote record message type: {msg_type}")

        if dataset is not None and args.push_to_hub:
            dataset.push_to_hub()
    finally:
        if dataset is not None:
            dataset.stop_image_writer()
        if listener is not None:
            listener.stop()
        if args.display and args.viewer == "opencv":
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        command_socket.close(0)
        socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
