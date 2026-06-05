#!/usr/bin/env python

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def strip_feature_info(features: dict) -> dict:
    return {key: {k: v for k, v in value.items() if k != "info"} for key, value in features.items()}


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items))


def validate_compatible(infos: list[dict], roots: list[Path]) -> None:
    base = infos[0]
    for root, info in zip(roots[1:], infos[1:]):
        mismatches = []
        for key in ["fps", "robot_type"]:
            if info.get(key) != base.get(key):
                mismatches.append(f"{key}: {info.get(key)!r} != {base.get(key)!r}")
        if strip_feature_info(info["features"]) != strip_feature_info(base["features"]):
            mismatches.append("features differ")
        if mismatches:
            raise ValueError(f"Dataset {root} is not compatible:\n" + "\n".join(mismatches))


def replace_column(table: pa.Table, name: str, values: list | range, pa_type: pa.DataType) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx < 0:
        return table
    return table.set_column(idx, name, pa.array(values, type=pa_type))


def merge_datasets(srcs: list[Path], dst: Path) -> None:
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    for src in srcs:
        if not (src / "meta/info.json").is_file():
            raise FileNotFoundError(f"Missing dataset metadata: {src / 'meta/info.json'}")

    infos = [json.loads((src / "meta/info.json").read_text()) for src in srcs]
    validate_compatible(infos, srcs)

    (dst / "data/chunk-000").mkdir(parents=True)
    (dst / "meta").mkdir(parents=True)
    (dst / "videos/chunk-000").mkdir(parents=True)

    task_to_new_index: dict[str, int] = {}
    tasks_out: list[dict] = []
    episodes_out: list[dict] = []
    stats_out: list[dict] = []

    global_frame_index = 0
    global_episode_index = 0
    video_keys = [p.name for p in (srcs[0] / "videos/chunk-000").iterdir() if p.is_dir()]
    video_keys.sort()

    for src in srcs:
        source_tasks = {item["task_index"]: item["task"] for item in read_jsonl(src / "meta/tasks.jsonl")}
        source_episodes = read_jsonl(src / "meta/episodes.jsonl")
        source_stats = {item["episode_index"]: item for item in read_jsonl(src / "meta/episodes_stats.jsonl")}

        for source_episode in source_episodes:
            old_ep = source_episode["episode_index"]
            old_parquet = src / f"data/chunk-000/episode_{old_ep:06d}.parquet"
            new_parquet = dst / f"data/chunk-000/episode_{global_episode_index:06d}.parquet"

            table = pq.read_table(old_parquet)
            episode_len = table.num_rows

            old_task_indices = table["task_index"].to_pylist() if "task_index" in table.column_names else []
            new_task_indices = []
            for old_task_idx in old_task_indices:
                task = source_tasks[old_task_idx]
                if task not in task_to_new_index:
                    task_to_new_index[task] = len(tasks_out)
                    tasks_out.append({"task_index": task_to_new_index[task], "task": task})
                new_task_indices.append(task_to_new_index[task])

            table = replace_column(table, "episode_index", [global_episode_index] * episode_len, pa.int64())
            table = replace_column(
                table, "index", range(global_frame_index, global_frame_index + episode_len), pa.int64()
            )
            if new_task_indices:
                table = replace_column(table, "task_index", new_task_indices, pa.int64())
            pq.write_table(table, new_parquet)

            episode_tasks = []
            for task in source_episode.get("tasks", []):
                if task not in task_to_new_index:
                    task_to_new_index[task] = len(tasks_out)
                    tasks_out.append({"task_index": task_to_new_index[task], "task": task})
                episode_tasks.append(task)
            episodes_out.append(
                {
                    "episode_index": global_episode_index,
                    "tasks": episode_tasks,
                    "length": source_episode["length"],
                }
            )

            episode_stats = json.loads(json.dumps(source_stats[old_ep]))
            episode_stats["episode_index"] = global_episode_index
            if "episode_index" in episode_stats["stats"]:
                episode_stats["stats"]["episode_index"].update(
                    {
                        "min": [global_episode_index],
                        "max": [global_episode_index],
                        "mean": [float(global_episode_index)],
                        "std": [0.0],
                        "count": [episode_len],
                    }
                )
            if "index" in episode_stats["stats"]:
                episode_stats["stats"]["index"].update(
                    {
                        "min": [global_frame_index],
                        "max": [global_frame_index + episode_len - 1],
                        "mean": [(global_frame_index + global_frame_index + episode_len - 1) / 2],
                        "std": episode_stats["stats"]["frame_index"]["std"],
                        "count": [episode_len],
                    }
                )
            if "task_index" in episode_stats["stats"] and new_task_indices:
                task_min = min(new_task_indices)
                task_max = max(new_task_indices)
                episode_stats["stats"]["task_index"].update(
                    {
                        "min": [task_min],
                        "max": [task_max],
                        "mean": [sum(new_task_indices) / len(new_task_indices)],
                        "std": [0.0] if task_min == task_max else episode_stats["stats"]["task_index"]["std"],
                        "count": [episode_len],
                    }
                )
            stats_out.append(episode_stats)

            for video_key in video_keys:
                src_video = src / "videos/chunk-000" / video_key / f"episode_{old_ep:06d}.mp4"
                dst_video = dst / "videos/chunk-000" / video_key / f"episode_{global_episode_index:06d}.mp4"
                dst_video.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_video, dst_video)

            global_frame_index += episode_len
            global_episode_index += 1

    info = json.loads(json.dumps(infos[0]))
    info["total_episodes"] = global_episode_index
    info["total_frames"] = global_frame_index
    info["total_tasks"] = len(tasks_out)
    info["total_videos"] = global_episode_index * len(video_keys)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{global_episode_index}"}

    (dst / "meta/info.json").write_text(json.dumps(info, indent=4) + "\n")
    write_jsonl(dst / "meta/tasks.jsonl", tasks_out)
    write_jsonl(dst / "meta/episodes.jsonl", episodes_out)
    write_jsonl(dst / "meta/episodes_stats.jsonl", stats_out)

    print(f"created {dst}")
    print(f"sources: {', '.join(str(src) for src in srcs)}")
    print(
        f"episodes: {info['total_episodes']} frames: {info['total_frames']} "
        f"tasks: {info['total_tasks']} videos: {info['total_videos']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge compatible LeRobotDataset directories.")
    parser.add_argument("--src", nargs="+", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    args = parser.parse_args()

    merge_datasets(args.src, args.dst)


if __name__ == "__main__":
    main()
