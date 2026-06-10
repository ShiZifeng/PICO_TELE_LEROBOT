#!/usr/bin/env python3
"""Stress test the SO-101 LeRobot recording hot path without hardware."""

import argparse
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def make_features(camera_shapes: dict[str, tuple[int, int, int]]) -> dict:
    features = {}
    for side in ("left", "right"):
        for joint in JOINT_NAMES:
            features[f"observation.{side}_{joint}.pos"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            }
            features[f"action.{side}_{joint}.pos"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            }
    for name, shape in camera_shapes.items():
        h, w, c = shape
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (c, h, w),
            "names": ["channel", "height", "width"],
        }
    return features


def make_frame(camera_images: dict[str, np.ndarray], step: int) -> dict:
    frame = {}
    for side_i, side in enumerate(("left", "right")):
        for joint_i, joint in enumerate(JOINT_NAMES):
            value = np.array([step * 0.01 + side_i + joint_i], dtype=np.float32)
            frame[f"observation.{side}_{joint}.pos"] = value
            frame[f"action.{side}_{joint}.pos"] = value
    for name, image in camera_images.items():
        frame[f"observation.images.{name}"] = image
    return frame


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    index = min(len(values_sorted) - 1, max(0, int(round((pct / 100.0) * (len(values_sorted) - 1)))))
    return values_sorted[index]


def run_case(args, writer_threads: int, writer_processes: int) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    camera_shapes = {
        "left_arm": (args.arm_height, args.arm_width, 3),
        "right_arm": (args.arm_height, args.arm_width, 3),
        "head": (args.head_height, args.head_width, 3),
    }
    rng = np.random.default_rng(args.seed)
    camera_images = {
        name: rng.integers(0, 256, size=shape, dtype=np.uint8)
        for name, shape in camera_shapes.items()
    }

    tmp_parent = Path(tempfile.mkdtemp(prefix="so101_record_bench_"))
    root = tmp_parent / f"dataset_t{writer_threads}_p{writer_processes}"
    try:
        dataset = LeRobotDataset.create(
            repo_id=f"local/so101_record_bench_t{writer_threads}_p{writer_processes}",
            fps=args.fps,
            root=root,
            robot_type="so101_follower_dual",
            features=make_features(camera_shapes),
            use_videos=True,
            image_writer_threads=writer_threads,
            image_writer_processes=writer_processes,
            batch_encoding_size=args.batch_encoding_size,
        )

        times_ms = []
        start = time.perf_counter()
        for step in range(args.frames):
            frame = make_frame(camera_images, step)
            t0 = time.perf_counter()
            dataset.add_frame(frame, task="XR teleop benchmark")
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        add_total = time.perf_counter() - start

        wait_t0 = time.perf_counter()
        dataset._wait_image_writer()
        wait_total = time.perf_counter() - wait_t0

        save_total = None
        if args.save_episode:
            save_t0 = time.perf_counter()
            dataset.save_episode()
            save_total = time.perf_counter() - save_t0

        try:
            dataset.stop_image_writer()
        except Exception:
            pass

        return {
            "threads": writer_threads,
            "processes": writer_processes,
            "frames": args.frames,
            "add_total_s": add_total,
            "add_fps": args.frames / add_total if add_total > 0 else 0.0,
            "wait_writer_s": wait_total,
            "save_episode_s": save_total,
            "mean_ms": statistics.fmean(times_ms),
            "median_ms": statistics.median(times_ms),
            "p95_ms": percentile(times_ms, 95),
            "p99_ms": percentile(times_ms, 99),
            "max_ms": max(times_ms),
            "slow_over_20ms": sum(t > 20.0 for t in times_ms),
        }
    finally:
        if not args.keep:
            shutil.rmtree(tmp_parent, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser("Benchmark SO-101 recording add_frame throughput")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--arm-width", type=int, default=640)
    parser.add_argument("--arm-height", type=int, default=400)
    parser.add_argument("--head-width", type=int, default=640)
    parser.add_argument("--head-height", type=int, default=480)
    parser.add_argument("--cases", default="0:0,4:0,8:0", help="Comma list of threads:processes cases")
    parser.add_argument("--batch-encoding-size", type=int, default=64, help="Avoid video encoding in save_episode during hot-path tests")
    parser.add_argument("--save-episode", action="store_true", help="Also measure save_episode time")
    parser.add_argument("--keep", action="store_true", help="Keep temporary dataset roots")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    print(
        f"Recording hot-path stress: frames={args.frames}, cameras="
        f"left/right {args.arm_width}x{args.arm_height}, head {args.head_width}x{args.head_height}"
    )
    print("case threads processes add_fps mean_ms p95_ms p99_ms max_ms >20ms wait_writer_s save_episode_s")
    for case in args.cases.split(","):
        threads_s, processes_s = case.split(":")
        result = run_case(args, int(threads_s), int(processes_s))
        save_s = "n/a" if result["save_episode_s"] is None else f"{result['save_episode_s']:.3f}"
        print(
            f"{result['threads']:>4} {result['threads']:>7} {result['processes']:>9} "
            f"{result['add_fps']:>7.1f} {result['mean_ms']:>7.2f} {result['p95_ms']:>7.2f} "
            f"{result['p99_ms']:>7.2f} {result['max_ms']:>7.2f} {result['slow_over_20ms']:>5} "
            f"{result['wait_writer_s']:>12.3f} {save_s:>14}"
        )


if __name__ == "__main__":
    main()
