#!/usr/bin/env python3
"""Record a guided XR axis calibration sequence without moving the robot."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from xrobotoolkit_teleop.common.xr_client import XrClient


XR_FRAME_ROTATIONS = {
    "simulation": np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float),
    "unity": np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]], dtype=float),
    "pico_y_flip": np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]], dtype=float),
}

SEQUENCE = [
    ("right", "forward", "+X forward"),
    ("right", "back", "-X back"),
    ("right", "left", "+Y left"),
    ("right", "right", "-Y right"),
    ("left", "forward", "+X forward"),
    ("left", "back", "-X back"),
    ("left", "left", "+Y left"),
    ("left", "right", "-Y right"),
]


def pose_xyz(xr: XrClient, side: str) -> np.ndarray:
    return np.asarray(xr.get_pose_by_name(f"{side}_controller")[:3], dtype=float)


def mean_pose(xr: XrClient, side: str, duration_s: float, rate_hz: float) -> np.ndarray:
    samples = []
    end = time.perf_counter() + duration_s
    period = 1.0 / rate_hz
    while time.perf_counter() < end:
        p = pose_xyz(xr, side)
        if np.all(np.isfinite(p)):
            samples.append(p)
        time.sleep(period)
    if not samples:
        raise RuntimeError(f"No valid XR samples for {side}")
    return np.mean(np.asarray(samples), axis=0)


def dominant_axis(v: np.ndarray) -> str:
    labels = ["X", "Y", "Z"]
    if np.linalg.norm(v) < 1e-4:
        return "near_zero"
    i = int(np.argmax(np.abs(v)))
    sign = "+" if v[i] >= 0 else "-"
    return f"{sign}{labels[i]}"


def countdown(message: str, seconds: int):
    for i in range(seconds, 0, -1):
        print(f"{message} {i}...", flush=True)
        time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser("Record guided XR axis calibration sequence")
    parser.add_argument("--prepare-s", type=int, default=3)
    parser.add_argument("--baseline-s", type=float, default=1.5)
    parser.add_argument("--move-s", type=float, default=2.5)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--out-dir", type=Path, default=Path("logs"))
    args = parser.parse_args()

    xr = XrClient()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Keep both controllers at neutral pose.", flush=True)
    countdown("Baseline starts in", args.prepare_s)
    refs = {
        "right": mean_pose(xr, "right", args.baseline_s, args.rate_hz),
        "left": mean_pose(xr, "left", args.baseline_s, args.rate_hz),
    }
    print("Baseline captured.", flush=True)

    results = []
    for side, motion, expected in SEQUENCE:
        countdown(f"Prepare: move {side} controller {motion}. Sampling starts in", args.prepare_s)
        samples = []
        end = time.perf_counter() + args.move_s
        while time.perf_counter() < end:
            p = pose_xyz(xr, side)
            if np.all(np.isfinite(p)):
                raw_delta = p - refs[side]
                samples.append(raw_delta)
            time.sleep(1.0 / args.rate_hz)

        arr = np.asarray(samples)
        raw_mean = np.mean(arr, axis=0)
        raw_peak = arr[int(np.argmax(np.linalg.norm(arr, axis=1)))]
        mapped = {}
        for name, R in XR_FRAME_ROTATIONS.items():
            mean_delta = R @ raw_mean
            peak_delta = R @ raw_peak
            mapped[name] = {
                "mean_delta": mean_delta.tolist(),
                "mean_axis": dominant_axis(mean_delta),
                "peak_delta": peak_delta.tolist(),
                "peak_axis": dominant_axis(peak_delta),
            }
        entry = {
            "side": side,
            "motion": motion,
            "expected": expected,
            "raw_mean": raw_mean.tolist(),
            "raw_peak": raw_peak.tolist(),
            "mapped": mapped,
        }
        results.append(entry)
        print(f"{side:5s} {motion:7s} expected {expected}: raw_mean={np.round(raw_mean, 3).tolist()}", flush=True)
        for name, info in mapped.items():
            print(f"  {name:12s} mean_axis={info['mean_axis']:>3s} peak_axis={info['peak_axis']:>3s}", flush=True)

    out = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "refs": {k: v.tolist() for k, v in refs.items()},
        "sequence": results,
    }
    out_path = args.out_dir / f"xr_axis_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
