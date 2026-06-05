#!/usr/bin/env python3
"""Debug XR controller axis mapping without moving the robot."""

import argparse
import time

import numpy as np

from xrobotoolkit_teleop.common.xr_client import XrClient


XR_FRAME_ROTATIONS = {
    "simulation": np.array(
        [
            [0, 0, -1],
            [-1, 0, 0],
            [0, 1, 0],
        ],
        dtype=float,
    ),
    "unity": np.array(
        [
            [0, 0, 1],
            [-1, 0, 0],
            [0, 1, 0],
        ],
        dtype=float,
    ),
    "pico_y_flip": np.array(
        [
            [0, 0, -1],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=float,
    ),
}


def fmt(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:+.3f}" for x in v) + "]"


def dominant_axis(v: np.ndarray) -> str:
    labels = ["+X forward", "+Y left", "+Z up"]
    if np.linalg.norm(v) < 1e-4:
        return "near zero"
    i = int(np.argmax(np.abs(v)))
    sign = "+" if v[i] >= 0 else "-"
    axis = labels[i].replace("+", sign, 1)
    return axis


def main():
    parser = argparse.ArgumentParser("Debug XR controller position axes")
    parser.add_argument("--controller", choices=["left", "right"], default="right")
    parser.add_argument("--rate", type=float, default=5.0)
    args = parser.parse_args()

    xr = XrClient()
    pose_name = f"{args.controller}_controller"

    print("Hold the controller at a neutral pose, then press Enter.")
    input()
    ref = np.asarray(xr.get_pose_by_name(pose_name)[:3], dtype=float)

    print()
    print("Move the controller in one physical direction at a time.")
    print("Expected robot axes: +X=forward, +Y=left, +Z=up.")
    print("Press Ctrl+C to stop. Press Enter again to reset the baseline.")
    print()

    last_print = 0.0
    try:
        while True:
            now = time.perf_counter()
            pose = np.asarray(xr.get_pose_by_name(pose_name)[:3], dtype=float)
            raw_delta = pose - ref

            if now - last_print >= 1.0 / args.rate:
                last_print = now
                print(f"raw delta {fmt(raw_delta)}")
                for name, R in XR_FRAME_ROTATIONS.items():
                    robot_delta = R @ raw_delta
                    print(f"  {name:12s} -> {fmt(robot_delta)}  {dominant_axis(robot_delta)}")
                print("-" * 72)

            time.sleep(0.01)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
