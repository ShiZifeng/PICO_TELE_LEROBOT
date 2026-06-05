#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def decode_remote_images(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    images = {}
    for key, value in observation.items():
        if isinstance(value, dict) and value.get("__remote_image_encoding__") == "jpg":
            frame = cv2.imdecode(np.frombuffer(value["data"], dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                images[key] = frame
        elif isinstance(value, np.ndarray) and value.ndim == 3:
            images[key] = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview camera observations streamed by remote_host.")
    parser.add_argument("--remote-ip", required=True, help="Jetson IP address.")
    parser.add_argument("--port-zmq-observations", type=int, default=5561)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--log-interval-s", type=float, default=2.0)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("outputs/remote_camera_preview"),
        help="Directory used to save preview frames when OpenCV GUI is unavailable.",
    )
    parser.add_argument("--no-display", action="store_true", help="Do not open OpenCV windows.")
    args = parser.parse_args()

    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    socket.connect(f"tcp://{args.remote_ip}:{args.port_zmq_observations}")

    last_log_time = 0.0
    last_save_time = 0.0
    display_enabled = not args.no_display
    args.save_dir.mkdir(parents=True, exist_ok=True)
    try:
        stream_count = 0
        camera_counts = defaultdict(int)
        last_stats_time = time.perf_counter()
        while True:
            observation = socket.recv_pyobj()
            images = decode_remote_images(observation)
            stream_count += 1
            for name in images:
                camera_counts[name] += 1

            now = time.perf_counter()
            if now - last_stats_time >= args.log_interval_s:
                dt = now - last_stats_time
                stream_fps = stream_count / dt
                camera_parts = []
                for name, image in images.items():
                    h, w = image.shape[:2]
                    camera_parts.append(f"{name}: {camera_counts[name] / dt:.1f} fps {w}x{h}")
                print(
                    f"stream: {stream_fps:.1f} fps | "
                    + (" | ".join(camera_parts) if camera_parts else "no cameras")
                )
                stream_count = 0
                camera_counts.clear()
                last_stats_time = now
            elif now - last_log_time > 2.0:
                print("Receiving cameras:", ", ".join(images) if images else "none")
                last_log_time = now

            if display_enabled:
                try:
                    for name, image in images.items():
                        cv2.imshow(name, image)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                except cv2.error as exc:
                    display_enabled = False
                    print(f"OpenCV GUI is unavailable; saving preview frames to {args.save_dir}. Error: {exc}")

            if not display_enabled and now - last_save_time > 1.0:
                for name, image in images.items():
                    cv2.imwrite(str(args.save_dir / f"{name}.jpg"), image)
                last_save_time = now
    except KeyboardInterrupt:
        pass
    finally:
        socket.close(0)
        context.term()
        if display_enabled:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    main()
