#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import argparse
import logging
import shutil
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import DEFAULT_IMAGE_PATH, write_info
from lerobot.datasets.video_utils import encode_video_frames
from lerobot.utils.utils import init_logging


def get_image_dir(root: Path, image_key: str, episode_index: int) -> Path:
    return (
        root
        / DEFAULT_IMAGE_PATH.format(
            image_key=image_key,
            episode_index=episode_index,
            frame_index=0,
        )
    ).parent


def encode_recorded_dataset(
    repo_id: str,
    root: str | Path | None = None,
    overwrite: bool = False,
    keep_images: bool = False,
    push_to_hub: bool = False,
    private: bool = False,
    tags: list[str] | None = None,
) -> None:
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    dataset_root = meta.root

    if len(meta.video_keys) == 0:
        logging.info("Dataset has no video keys; nothing to encode.")
        return

    for episode_index in range(meta.total_episodes):
        for video_key in meta.video_keys:
            video_path = dataset_root / meta.get_video_file_path(episode_index, video_key)
            img_dir = get_image_dir(dataset_root, video_key, episode_index)

            if video_path.is_file() and not overwrite:
                logging.info("Skipping existing video: %s", video_path)
                continue

            if not img_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing raw image directory for episode {episode_index}, camera {video_key}: {img_dir}"
                )

            logging.info("Encoding episode %s camera %s -> %s", episode_index, video_key, video_path)
            encode_video_frames(img_dir, video_path, meta.fps, overwrite=True)

            if not keep_images:
                shutil.rmtree(img_dir)

    meta.update_video_info()
    write_info(meta.info, meta.root)

    if not keep_images:
        images_root = dataset_root / "images"
        if images_root.exists() and not any(images_root.rglob("*.png")):
            shutil.rmtree(images_root)

    if push_to_hub:
        dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
        dataset.push_to_hub(tags=tags, private=private)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode deferred LeRobot dataset PNG frames into mp4 videos.")
    parser.add_argument("--repo-id", required=True, help="Dataset repo id, for example user/my_dataset.")
    parser.add_argument("--root", default=None, help="Local dataset root copied from the recording machine.")
    parser.add_argument("--overwrite", action="store_true", help="Re-encode videos that already exist.")
    parser.add_argument("--keep-images", action="store_true", help="Keep raw PNG frames after encoding.")
    parser.add_argument("--push-to-hub", action="store_true", help="Upload the encoded dataset after processing.")
    parser.add_argument("--private", action="store_true", help="Create or update a private Hub dataset.")
    parser.add_argument("--tags", nargs="*", default=None, help="Optional dataset tags when pushing to Hub.")
    args = parser.parse_args()

    init_logging()
    encode_recorded_dataset(
        repo_id=args.repo_id,
        root=args.root,
        overwrite=args.overwrite,
        keep_images=args.keep_images,
        push_to_hub=args.push_to_hub,
        private=args.private,
        tags=args.tags,
    )


if __name__ == "__main__":
    main()
