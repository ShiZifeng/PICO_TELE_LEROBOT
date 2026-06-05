#!/usr/bin/env python3
"""
Convert XRoboToolkit .pkl recording to LeRobot Dataset format.

Usage:
  # Convert single-arm recording
  python scripts/misc/convert_pkl_to_lerobot.py \\
    --pkl-path logs/so101/teleop_log_20240604_133700_1.pkl \\
    --repo-id local/so101_xr_teleop \\
    --output-dir datasets/so101_xr_teleop

  # Convert dual-arm recording
  python scripts/misc/convert_pkl_to_lerobot.py \\
    --pkl-path logs/so101/teleop_log_20240604_133700_1.pkl \\
    --repo-id local/so101_dual_xr_teleop \\
    --output-dir datasets/so101_dual_xr_teleop \\
    --mode dual
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def _find_lerobot_path() -> Optional[str]:
    env_path = os.environ.get("LEROBOT_PATH")
    project_root = Path(__file__).resolve().parents[2]
    bundled_path = project_root / "third_party" / "lerobot-v0.3.3"
    candidates = [
        os.path.join(env_path, "src") if env_path else None,
        env_path,
        str(bundled_path / "src"),
        "/media/shizifeng/projects21/lerobot-v0.3.3/src",
        os.path.expanduser("~/szf_lerobot/lerobot-v0.3.3/src"),
    ]
    for p in candidates:
        if p and os.path.isdir(os.path.join(p, "lerobot")):
            return p
    return None


def _ensure_lerobot():
    lerobot_path = _find_lerobot_path()
    if lerobot_path and lerobot_path not in sys.path:
        sys.path.insert(0, lerobot_path)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.datasets.utils import hw_to_dataset_features
        return True
    except ImportError as e:
        print(f"ERROR: LeRobot not found. Install it first: pip install -e /path/to/lerobot-v0.3.3")
        print(f"  Original error: {e}")
        return False


def build_features(mode: str, has_camera: bool, camera_names: List[str]) -> Dict[str, Any]:
    """Build LeRobot feature dict from recording structure."""
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    features = {}
    sides = ["left", "right"] if mode == "dual" else ["right"]

    for side in sides:
        prefix = f"{side}_" if mode == "dual" else ""
        for joint in joint_names:
            features[f"{prefix}{joint}.pos"] = {"dtype": "float32", "shape": (1,), "names": None}

    if has_camera:
        for cam_name in camera_names:
            features[cam_name] = {"dtype": "video", "shape": (3,), "names": ["channels", "height", "width"]}

    return features


def convert(
    pkl_path: str,
    repo_id: str,
    output_dir: str = "datasets",
    mode: str = "single",
    fps: int = 30,
    video_backend: str = "pyav",
) -> str:
    """
    Convert a .pkl recording to LeRobot Dataset format.

    Args:
        pkl_path: Path to the .pkl file.
        repo_id: LeRobot dataset repo ID (e.g., "local/so101_xr_teleop").
        output_dir: Output directory for the LeRobot dataset.
        mode: "single" or "dual".
        fps: Recording FPS for the dataset.
        video_backend: Video encoding backend ("pyav" or "opencv").

    Returns:
        Path to the created LeRobot dataset.
    """
    if not _ensure_lerobot():
        raise RuntimeError("LeRobot not available.")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Load .pkl data
    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print(f"  Loaded {len(data)} frames")

    # Detect structure
    first_frame = data[0]
    has_camera = "image" in first_frame
    camera_names = []
    if has_camera:
        camera_names = list(first_frame["image"].keys())
        print(f"  Cameras: {camera_names}")

    # Build features
    features = build_features(mode, has_camera, camera_names)

    # Create dataset
    robot_type = "so101_follower_dual" if mode == "dual" else "so101_follower"
    root = os.path.join(output_dir, repo_id.replace("/", "_"))
    os.makedirs(root, exist_ok=True)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        robot_type=robot_type,
        use_videos=has_camera,
        video_backend=video_backend,
    )

    print(f"Created dataset at: {root}")

    # Write frames
    episode_data = {"observation.state": [], "action": [], "timestamp": []}
    if has_camera:
        for cam_name in camera_names:
            episode_data[cam_name] = []

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    sides = ["left", "right"] if mode == "dual" else ["right"]

    for frame_idx, frame in enumerate(data):
        # Build observation.state
        obs_list = []
        for side in sides:
            prefix = f"{side}_" if mode == "dual" else ""
            qpos = frame.get("qpos", {}).get(f"{side}_arm", np.zeros(6))
            if isinstance(qpos, np.ndarray):
                qpos = qpos.tolist()
            elif not isinstance(qpos, list):
                qpos = [0.0] * 6
            obs_list.extend(qpos)
        episode_data["observation.state"].append(np.array(obs_list, dtype=np.float32))

        # Build action (desired joint positions from IK)
        act_list = []
        for side in sides:
            prefix = f"{side}_" if mode == "dual" else ""
            qpos_des = frame.get("qpos_des", {}).get(f"{side}_arm", np.zeros(6))
            if isinstance(qpos_des, np.ndarray):
                qpos_des = qpos_des.tolist()
            elif not isinstance(qpos_des, list):
                qpos_des = [0.0] * 6
            act_list.extend(qpos_des)
        episode_data["action"].append(np.array(act_list, dtype=np.float32))

        # Timestamp
        episode_data["timestamp"].append(frame.get("timestamp", frame_idx / fps))

        # Camera frames
        if has_camera:
            images = frame.get("image", {})
            for cam_name in camera_names:
                img_data = images.get(cam_name, {})
                if "color" in img_data:
                    color_data = img_data["color"]
                    if isinstance(color_data, bytes):
                        # Decode JPG bytes to numpy array
                        arr = cv2.imdecode(np.frombuffer(color_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if arr is not None:
                            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                    elif isinstance(color_data, np.ndarray):
                        arr = color_data
                    else:
                        arr = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    arr = np.zeros((480, 640, 3), dtype=np.uint8)
                episode_data[cam_name].append(arr)

        if (frame_idx + 1) % 100 == 0:
            print(f"  Processed {frame_idx + 1}/{len(data)} frames...")

    # Save episode
    dataset.save_episode(episode_data)
    print(f"\n✅ LeRobot dataset created: {root}")
    print(f"   Episodes: {len(dataset.episodes) if dataset.episodes else 1}")
    print(f"   Frames: {len(data)}")

    return root


def main():
    parser = argparse.ArgumentParser(description="Convert .pkl recording to LeRobot Dataset")
    parser.add_argument("--pkl-path", required=True, help="Path to .pkl file")
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repo ID")
    parser.add_argument("--output-dir", default="datasets", help="Output directory")
    parser.add_argument("--mode", default="single", choices=["single", "dual"])
    parser.add_argument("--fps", type=int, default=30, help="Recording FPS")
    parser.add_argument("--video-backend", default="pyav", choices=["pyav", "opencv"])
    args = parser.parse_args()

    if not os.path.exists(args.pkl_path):
        print(f"ERROR: File not found: {args.pkl_path}")
        sys.exit(1)

    try:
        convert(
            pkl_path=args.pkl_path,
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            mode=args.mode,
            fps=args.fps,
            video_backend=args.video_backend,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
