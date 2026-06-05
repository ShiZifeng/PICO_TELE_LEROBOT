#!/usr/bin/env python

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def numeric_stats(values: np.ndarray) -> dict:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]

    count = values.shape[0]
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [count],
    }


def rewrite_episode_table(table: pa.Table, episode_index: int, start_global_index: int, fps: int) -> pa.Table:
    num_rows = table.num_rows
    replacements = {
        "episode_index": pa.array([episode_index] * num_rows, type=pa.int64()),
        "frame_index": pa.array(range(num_rows), type=pa.int64()),
        "timestamp": pa.array([i / fps for i in range(num_rows)], type=pa.float32()),
        "index": pa.array(range(start_global_index, start_global_index + num_rows), type=pa.int64()),
    }

    for name, array in replacements.items():
        col_idx = table.schema.get_field_index(name)
        if col_idx >= 0:
            table = table.set_column(col_idx, name, array)
    return table


def recompute_episode_stats(table: pa.Table, old_stats: dict) -> dict:
    stats = json.loads(json.dumps(old_stats))
    stats["episode_index"] = int(table["episode_index"][0].as_py())

    for key in ["action", "observation.state"]:
        if key in table.column_names and key in stats["stats"]:
            stats["stats"][key] = numeric_stats(np.asarray(table[key].to_pylist(), dtype=np.float32))

    for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        if key in table.column_names and key in stats["stats"]:
            stats["stats"][key] = numeric_stats(np.asarray(table[key].to_pylist()))

    # Image stats are kept from the source episode. The actual image/video data is trimmed
    # below; these image stats are only coarse metadata and are not needed for alignment.
    return stats


def trim_video(src: Path, dst: Path, trim_frames: int, fps: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    trim_filter = f"select='gte(n\\,{trim_frames})',setpts=N/({fps}*TB)"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        trim_filter,
        "-r",
        str(fps),
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def trim_episode_prefix(src: Path, dst: Path, episode_index: int, trim_seconds: float) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source does not exist: {src}")
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    info = json.loads((src / "meta/info.json").read_text())
    fps = int(info["fps"])
    trim_frames = int(round(trim_seconds * fps))
    if trim_frames <= 0:
        raise ValueError("trim_seconds must remove at least one frame")

    episodes = [
        json.loads(line) for line in (src / "meta/episodes.jsonl").read_text().splitlines() if line.strip()
    ]
    stats = [
        json.loads(line)
        for line in (src / "meta/episodes_stats.jsonl").read_text().splitlines()
        if line.strip()
    ]
    stats_by_ep = {item["episode_index"]: item for item in stats}

    if episode_index not in {ep["episode_index"] for ep in episodes}:
        raise ValueError(f"Episode {episode_index} not found")

    shutil.copytree(src, dst)

    new_episodes = []
    new_stats = []
    global_index = 0
    removed_frames = 0

    for episode in episodes:
        ep_idx = episode["episode_index"]
        parquet_path = dst / f"data/chunk-000/episode_{ep_idx:06d}.parquet"
        table = pq.read_table(parquet_path)

        if ep_idx == episode_index:
            if trim_frames >= table.num_rows:
                raise ValueError(
                    f"Cannot trim {trim_frames} frames from episode {episode_index}; it only has {table.num_rows} frames"
                )
            table = table.slice(trim_frames)
            removed_frames = trim_frames

            for video_dir in (dst / "videos/chunk-000").iterdir():
                if not video_dir.is_dir():
                    continue
                video_path = video_dir / f"episode_{ep_idx:06d}.mp4"
                tmp_path = video_path.with_suffix(".trim.mp4")
                trim_video(video_path, tmp_path, trim_frames, fps)
                tmp_path.replace(video_path)

        table = rewrite_episode_table(table, ep_idx, global_index, fps)
        pq.write_table(table, parquet_path)

        new_episode = dict(episode)
        new_episode["length"] = table.num_rows
        new_episodes.append(new_episode)
        new_stats.append(recompute_episode_stats(table, stats_by_ep[ep_idx]))

        global_index += table.num_rows

    info["total_frames"] = global_index
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4) + "\n")
    (dst / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(episode, ensure_ascii=False) + "\n" for episode in new_episodes)
    )
    (dst / "meta/episodes_stats.jsonl").write_text(
        "".join(json.dumps(episode_stats, ensure_ascii=False) + "\n" for episode_stats in new_stats)
    )

    print(f"created {dst}")
    print(f"trimmed episode: {episode_index}")
    print(f"removed prefix: {trim_seconds:g}s = {removed_frames} frames at {fps} fps")
    print(f"new total frames: {global_index}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim the beginning of one LeRobotDataset episode.")
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--trim-seconds", type=float, required=True)
    args = parser.parse_args()

    trim_episode_prefix(args.src, args.dst, args.episode, args.trim_seconds)


if __name__ == "__main__":
    main()
