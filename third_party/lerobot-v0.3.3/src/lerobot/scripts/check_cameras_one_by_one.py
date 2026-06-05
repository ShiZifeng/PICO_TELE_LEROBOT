#!/usr/bin/env python

import argparse
import time
from pathlib import Path

import cv2
from PIL import Image

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.orbbec import OrbbecCamera, OrbbecCameraConfig
from lerobot.utils.utils import init_logging


def save_rgb(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def save_debug_variants(output_dir: Path, name: str, image) -> None:
    save_rgb(output_dir / f"{name}_rgb_pil.png", image)
    cv2.imwrite(str(output_dir / f"{name}_as_bgr_cv2.png"), image)
    if image.ndim == 3 and image.shape[2] == 3:
        cv2.imwrite(str(output_dir / f"{name}_rgb_to_bgr_cv2.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def probe_camera(name: str, camera, output_dir: Path, seconds: float) -> None:
    print(f"\n=== {name} ===")
    try:
        camera.connect(warmup=False)
        start = time.perf_counter()
        count = 0
        last_image = None
        while time.perf_counter() - start < seconds:
            last_image = camera.read()
            count += 1
        elapsed = time.perf_counter() - start
        fps = count / elapsed if elapsed else 0
        print(f"OK: {count} frames in {elapsed:.2f}s -> {fps:.1f} fps")
        if last_image is not None:
            print(f"shape={last_image.shape}, dtype={last_image.dtype}")
            save_debug_variants(output_dir, name, last_image)
            print(f"saved: {output_dir}/{name}_*.png")
    except Exception as exc:
        print(f"FAILED: {exc}")
    finally:
        try:
            if camera.is_connected:
                camera.disconnect()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe OpenCV and Orbbec cameras one by one.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/camera_check"))
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--include-orbbec", action="store_true")
    args = parser.parse_args()

    init_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nDetected OpenCV cameras:")
    opencv_infos = OpenCVCamera.find_cameras()
    for info in opencv_infos:
        print(info)

    for info in opencv_infos:
        profile = info.get("default_stream_profile", {})
        camera = OpenCVCamera(
            OpenCVCameraConfig(
                index_or_path=info["id"],
                width=int(profile["width"]),
                height=int(profile["height"]),
                fps=int(round(profile["fps"])),
                color_mode=ColorMode.RGB,
            )
        )
        safe_name = f"opencv_{str(info['id']).replace('/', '_')}"
        probe_camera(safe_name, camera, args.output_dir, args.seconds)

    if not args.include_orbbec:
        print("\nSkipping Orbbec probe. Add --include-orbbec to test it.")
        return

    print("\nDetected Orbbec cameras:")
    orbbec_infos = OrbbecCamera.find_cameras()
    for info in orbbec_infos:
        print(info)

    for info in orbbec_infos:
        profile = info.get("default_stream_profile") or {}
        camera = OrbbecCamera(
            OrbbecCameraConfig(
                serial_number=info["id"],
                width=int(profile["width"]) if profile else None,
                height=int(profile["height"]) if profile else None,
                fps=int(round(profile["fps"])) if profile else None,
                color_mode=ColorMode.RGB,
            )
        )
        safe_name = f"orbbec_{info['id']}"
        probe_camera(safe_name, camera, args.output_dir, args.seconds)


if __name__ == "__main__":
    main()
