#!/usr/bin/env python3
"""Benchmark SO-101 PC-side Placo IK without XR service or robot hardware."""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
import types
from pathlib import Path


class FakeXrClient:
    def get_key_value_by_name(self, name: str) -> float:
        if "grip" in name:
            return 1.0
        if "trigger" in name:
            return 0.0
        return 0.0

    def get_button_state_by_name(self, name: str) -> bool:
        return False

    def get_pose_by_name(self, name: str) -> list[float]:
        # XR pose layout: [x, y, z, qx, qy, qz, qw]
        if name == "left_controller":
            return [0.10, 0.20, -0.30, 0.0, 0.0, 0.0, 1.0]
        if name == "right_controller":
            return [0.10, -0.20, -0.30, 0.0, 0.0, 0.0, 1.0]
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def load_remote_pc_module():
    # Avoid requiring the XRoboToolkit runtime SDK for an offline IK benchmark.
    sys.modules.setdefault("xrobotoolkit_sdk", types.SimpleNamespace(init=lambda: None, close=lambda: None))

    path = Path(__file__).resolve().parent / "teleop_so101_remote_pc.py"
    spec = importlib.util.spec_from_file_location("teleop_so101_remote_pc_bench", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.XrClient = FakeXrClient
    return mod


def make_zero_observation(mod) -> dict[str, float]:
    obs = {}
    for side in ["left", "right"]:
        for joint in mod.SO101_JOINT_NAMES:
            obs[f"observation.{side}_{joint}.pos"] = 0.0
    return obs


def benchmark(mode: str, steps: int, warmup: int) -> None:
    mod = load_remote_pc_module()
    ctrl = mod.XRIKController(
        mode="dual",
        scale_factor=1.0,
        control_rate_hz=50,
        max_action_delta_deg=0.0,
        xr_frame="simulation",
        ik_control_mode=mode,
        require_xr_confirm=False,
    )
    ctrl.update_robot_state(make_zero_observation(mod))
    ctrl.step()  # sync task targets to robot state
    ctrl.step()  # initialize XR references

    for _ in range(warmup):
        ctrl.step()

    dts = []
    t0 = time.perf_counter()
    for _ in range(steps):
        start = time.perf_counter()
        ctrl.step()
        dts.append(time.perf_counter() - start)
    total = time.perf_counter() - t0

    dts_ms = [dt * 1000 for dt in dts]
    dts_ms_sorted = sorted(dts_ms)
    print(f"control_mode={mode}")
    print(f"  total: {total:.3f}s for {steps} steps")
    print(f"  throughput: {steps / total:.1f} Hz")
    print(f"  mean step: {statistics.mean(dts_ms):.3f} ms")
    print(f"  median step: {statistics.median(dts_ms):.3f} ms")
    print(f"  p95 step: {dts_ms_sorted[int(0.95 * steps)]:.3f} ms")
    print(f"  p99 step: {dts_ms_sorted[int(0.99 * steps)]:.3f} ms")
    print(f"  max step: {max(dts_ms):.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pose", "position", "both"], default="both")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=200)
    args = parser.parse_args()

    modes = ["pose", "position"] if args.mode == "both" else [args.mode]
    for mode in modes:
        benchmark(mode, args.steps, args.warmup)


if __name__ == "__main__":
    main()
