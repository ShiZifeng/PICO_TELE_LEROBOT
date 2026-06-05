#!/usr/bin/env python

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def copy_filtered_dataset(src: Path, dst: Path, remove_episodes: set[int]) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source does not exist: {src}")
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    (dst / "data/chunk-000").mkdir(parents=True)
    (dst / "meta").mkdir(parents=True)
    (dst / "videos/chunk-000").mkdir(parents=True)
    if (src / "images").exists():
        (dst / "images").mkdir(parents=True)

    info = json.loads((src / "meta/info.json").read_text())
    episodes = [
        json.loads(line) for line in (src / "meta/episodes.jsonl").read_text().splitlines() if line.strip()
    ]
    stats = [
        json.loads(line)
        for line in (src / "meta/episodes_stats.jsonl").read_text().splitlines()
        if line.strip()
    ]

    old_episode_indices = [ep["episode_index"] for ep in episodes]
    missing = sorted(remove_episodes - set(old_episode_indices))
    if missing:
        raise ValueError(f"Episodes not found in source: {missing}")

    keep_old = [ep for ep in old_episode_indices if ep not in remove_episodes]
    old_to_new = {old: new for new, old in enumerate(keep_old)}

    shutil.copy2(src / "meta/tasks.jsonl", dst / "meta/tasks.jsonl")

    episodes_by_old = {item["episode_index"]: item for item in episodes}
    stats_by_old = {item["episode_index"]: item for item in stats}
    video_dirs = [d for d in (src / "videos/chunk-000").iterdir() if d.is_dir()]

    new_episodes = []
    new_stats = []
    global_index = 0

    for old_ep in keep_old:
        new_ep = old_to_new[old_ep]
        old_parquet = src / f"data/chunk-000/episode_{old_ep:06d}.parquet"
        new_parquet = dst / f"data/chunk-000/episode_{new_ep:06d}.parquet"

        table = pq.read_table(old_parquet)
        episode_len = table.num_rows
        replacements = {
            "episode_index": pa.array([new_ep] * episode_len, type=pa.int64()),
            "index": pa.array(range(global_index, global_index + episode_len), type=pa.int64()),
        }
        for name, array in replacements.items():
            col_idx = table.schema.get_field_index(name)
            table = table.set_column(col_idx, name, array)
        pq.write_table(table, new_parquet)

        episode = dict(episodes_by_old[old_ep])
        episode["episode_index"] = new_ep
        new_episodes.append(episode)

        episode_stats = json.loads(json.dumps(stats_by_old[old_ep]))
        episode_stats["episode_index"] = new_ep
        if "episode_index" in episode_stats["stats"]:
            episode_stats["stats"]["episode_index"].update(
                {"min": [new_ep], "max": [new_ep], "mean": [float(new_ep)], "std": [0.0], "count": [episode_len]}
            )
        if "index" in episode_stats["stats"]:
            episode_stats["stats"]["index"].update(
                {
                    "min": [global_index],
                    "max": [global_index + episode_len - 1],
                    "mean": [(global_index + global_index + episode_len - 1) / 2],
                    "std": episode_stats["stats"]["frame_index"]["std"],
                    "count": [episode_len],
                }
            )
        new_stats.append(episode_stats)

        for video_key_dir in video_dirs:
            rel_dir = video_key_dir.relative_to(src / "videos/chunk-000")
            out_dir = dst / "videos/chunk-000" / rel_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video_key_dir / f"episode_{old_ep:06d}.mp4", out_dir / f"episode_{new_ep:06d}.mp4")

        global_index += episode_len

    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = global_index
    info["total_videos"] = len(new_episodes) * len(video_dirs)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{len(new_episodes)}"}

    (dst / "meta/info.json").write_text(json.dumps(info, indent=4) + "\n")
    (dst / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(episode, ensure_ascii=False) + "\n" for episode in new_episodes)
    )
    (dst / "meta/episodes_stats.jsonl").write_text(
        "".join(json.dumps(episode_stats, ensure_ascii=False) + "\n" for episode_stats in new_stats)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a LeRobotDataset copy with selected episodes removed.")
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--remove-episodes", nargs="+", type=int, required=True)
    args = parser.parse_args()

    copy_filtered_dataset(args.src, args.dst, set(args.remove_episodes))

    info = json.loads((args.dst / "meta/info.json").read_text())
    print(f"created {args.dst}")
    print(f"removed episodes: {sorted(args.remove_episodes)}")
    print(
        f"new episodes: {info['total_episodes']} frames: {info['total_frames']} videos: {info['total_videos']}"
    )


if __name__ == "__main__":
    main()
