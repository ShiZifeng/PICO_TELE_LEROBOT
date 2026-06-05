#!/usr/bin/env python

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def resolve_dataset_root(dataset_root: str | Path, repo_id: str | None) -> Path:
    if repo_id is None:
        return Path(dataset_root)

    repo_path = Path(repo_id)
    if repo_path.parts and repo_path.parts[0] == "local":
        return Path.home() / ".cache/huggingface/lerobot" / repo_path

    return Path(dataset_root) / repo_path


def load_joint_names(dataset_root: Path) -> list[str]:
    info = json.loads((dataset_root / "meta/info.json").read_text())
    return info["features"]["action"]["names"]


def select_joint_indices(joint_names: list[str], joints: list[str]) -> list[int]:
    if joints == ["all"]:
        return list(range(len(joint_names)))

    selected = []
    missing = []
    for joint in joints:
        exact = [i for i, name in enumerate(joint_names) if name == joint]
        if exact:
            selected.extend(exact)
            continue

        partial = [i for i, name in enumerate(joint_names) if joint in name]
        if partial:
            selected.extend(partial)
        else:
            missing.append(joint)

    if missing:
        raise ValueError(f"Could not find joints: {missing}. Available joints: {joint_names}")

    return sorted(set(selected))


def load_episode_arrays(parquet_path: Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(parquet_path, columns=["action", "observation.state", "timestamp", "episode_index"])
    data = table.to_pydict()
    action = np.asarray(data["action"], dtype=np.float32)
    state = np.asarray(data["observation.state"], dtype=np.float32)
    timestamp = np.asarray(data["timestamp"], dtype=np.float32)
    episode_index = int(data["episode_index"][0])
    return episode_index, timestamp, action, state


def color_for_index(index: int) -> tuple[int, int, int]:
    palette = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]
    return palette[index % len(palette)]


def draw_text(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.55) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (35, 35, 35), 1, cv2.LINE_AA)


def normalize_points(x: np.ndarray, y: np.ndarray, rect: tuple[int, int, int, int], y_min: float, y_max: float):
    left, top, width, height = rect
    if len(x) == 0:
        return []
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    xs = left + (x - x_min) / (x_max - x_min) * width
    ys = top + height - (y - y_min) / (y_max - y_min) * height
    return np.column_stack([xs, ys]).astype(np.int32)


def save_cv2_trajectory_plot(
    joint_name: str,
    joint_idx: int,
    episodes: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path,
    max_overlay_episodes: int,
) -> None:
    canvas = np.full((920, 1500, 3), 255, dtype=np.uint8)
    draw_text(canvas, f"{joint_name} trajectories", (30, 38), 0.9)

    plot_rects = [(90, 80, 1340, 330), (90, 520, 1340, 330)]
    labels = [("action", 0), ("observation.state", 1)]

    selected = episodes[:max_overlay_episodes]
    all_values = []
    for _ep_idx, _timestamp, action, state in selected:
        all_values.extend([action[:, joint_idx], state[:, joint_idx]])
    y_min = min(float(np.nanmin(v)) for v in all_values)
    y_max = max(float(np.nanmax(v)) for v in all_values)
    margin = max(1e-6, (y_max - y_min) * 0.08)
    y_min -= margin
    y_max += margin

    for rect, (label, source_idx) in zip(plot_rects, labels, strict=True):
        left, top, width, height = rect
        cv2.rectangle(canvas, (left, top), (left + width, top + height), (220, 220, 220), 1)
        draw_text(canvas, label, (left, top - 16), 0.65)
        draw_text(canvas, f"{y_max:.1f}", (18, top + 8), 0.45)
        draw_text(canvas, f"{y_min:.1f}", (18, top + height), 0.45)

        for i, (ep_idx, timestamp, action, state) in enumerate(selected):
            values = action[:, joint_idx] if source_idx == 0 else state[:, joint_idx]
            points = normalize_points(timestamp, values, rect, y_min, y_max)
            cv2.polylines(canvas, [points], False, color_for_index(i), 1, cv2.LINE_AA)
            if i < 10:
                draw_text(canvas, f"ep{ep_idx:03d}", (left + 10 + i * 90, top + height + 24), 0.42)

    cv2.imwrite(str(output_dir / f"{sanitize_filename(joint_name)}_trajectories.png"), canvas)


def save_cv2_bar_plot(joint_name: str, rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    joint_rows = [row for row in rows if row["joint"] == joint_name and row["source"] == "state"]
    episode_indices = [int(row["episode_index"]) for row in joint_rows]
    metrics = [
        ("range", "Range"),
        ("mean_abs_delta", "Mean |delta|"),
        ("max_abs_delta", "Max |delta|"),
        ("mean_abs_action_state_error", "Mean |action-state|"),
    ]

    canvas = np.full((1180, 1500, 3), 255, dtype=np.uint8)
    draw_text(canvas, f"{joint_name} per-episode variation", (30, 38), 0.9)

    for m_idx, (metric_key, title) in enumerate(metrics):
        values = np.asarray([float(row[metric_key]) for row in joint_rows], dtype=np.float32)
        left, top, width, height = 90, 80 + m_idx * 265, 1340, 190
        cv2.rectangle(canvas, (left, top), (left + width, top + height), (220, 220, 220), 1)
        draw_text(canvas, title, (left, top - 15), 0.62)
        vmax = float(np.nanmax(values)) if len(values) else 1.0
        if vmax <= 0:
            vmax = 1.0
        bar_w = max(2, int(width / max(1, len(values)) * 0.75))
        for i, value in enumerate(values):
            x = int(left + i / max(1, len(values)) * width)
            bar_h = int((float(value) / vmax) * height)
            cv2.rectangle(canvas, (x, top + height - bar_h), (x + bar_w, top + height), (70, 130, 200), -1)
        draw_text(canvas, f"max {vmax:.2f}", (left + width - 110, top + 18), 0.45)
        for i, ep_idx in enumerate(episode_indices):
            if i % max(1, len(episode_indices) // 12) == 0:
                x = int(left + i / max(1, len(episode_indices)) * width)
                draw_text(canvas, str(ep_idx), (x, top + height + 18), 0.38)

    cv2.imwrite(str(output_dir / f"{sanitize_filename(joint_name)}_variation_by_episode.png"), canvas)


def compute_metrics(values: np.ndarray, action: np.ndarray, state: np.ndarray, joint_idx: int) -> dict[str, float]:
    action_j = action[:, joint_idx]
    state_j = state[:, joint_idx]
    if len(values) > 1:
        delta = np.abs(np.diff(values))
    else:
        delta = np.array([0.0], dtype=np.float32)

    return {
        "range": float(np.nanmax(values) - np.nanmin(values)),
        "mean_abs_delta": float(np.nanmean(delta)),
        "max_abs_delta": float(np.nanmax(delta)),
        "mean_abs_action_state_error": float(np.nanmean(np.abs(action_j - state_j))),
        "max_abs_action_state_error": float(np.nanmax(np.abs(action_j - state_j))),
    }


def plot_joint_trajectories(
    joint_name: str,
    joint_idx: int,
    episodes: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path,
    max_overlay_episodes: int,
) -> None:
    if plt is None:
        save_cv2_trajectory_plot(joint_name, joint_idx, episodes, output_dir, max_overlay_episodes)
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    fig.suptitle(f"{joint_name} trajectories")

    for ep_idx, timestamp, action, state in episodes[:max_overlay_episodes]:
        axes[0].plot(timestamp, action[:, joint_idx], linewidth=0.9, alpha=0.45, label=f"ep{ep_idx:03d}")
        axes[1].plot(timestamp, state[:, joint_idx], linewidth=0.9, alpha=0.45, label=f"ep{ep_idx:03d}")

    axes[0].set_title("action")
    axes[1].set_title("observation.state")
    for ax in axes:
        ax.set_xlabel("time (s)")
        ax.set_ylabel("joint value")
        ax.grid(True, alpha=0.25)

    if len(episodes) <= 20:
        axes[0].legend(ncol=4, fontsize=7)
        axes[1].legend(ncol=4, fontsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / f"{sanitize_filename(joint_name)}_trajectories.png", dpi=160)
    plt.close(fig)


def plot_joint_variation_bars(joint_name: str, rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    if plt is None:
        save_cv2_bar_plot(joint_name, rows, output_dir)
        return

    joint_rows = [row for row in rows if row["joint"] == joint_name and row["source"] == "state"]
    episode_indices = [int(row["episode_index"]) for row in joint_rows]

    metrics = [
        ("range", "Range"),
        ("mean_abs_delta", "Mean |delta|"),
        ("max_abs_delta", "Max |delta|"),
        ("mean_abs_action_state_error", "Mean |action-state|"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{joint_name} per-episode variation")
    for ax, (metric_key, title) in zip(axes, metrics, strict=True):
        values = [float(row[metric_key]) for row in joint_rows]
        ax.bar(episode_indices, values, width=0.8)
        ax.set_ylabel(title)
        ax.grid(True, axis="y", alpha=0.25)
    axes[-1].set_xlabel("episode_index")
    fig.tight_layout()
    fig.savefig(output_dir / f"{sanitize_filename(joint_name)}_variation_by_episode.png", dpi=160)
    plt.close(fig)


def write_summary_csv(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    fieldnames = [
        "episode_index",
        "joint",
        "source",
        "length",
        "range",
        "mean_abs_delta",
        "max_abs_delta",
        "mean_abs_action_state_error",
        "max_abs_action_state_error",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize joint trajectories and per-episode variation.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/mobile_dual_remote_test_clean"),
        help="Path to a local LeRobotDataset root.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Optional repo id, e.g. local/mobile_dual_remote_test_clean. If set, overrides --dataset-root.",
    )
    parser.add_argument(
        "--joints",
        nargs="+",
        default=["all"],
        help="Joint names or substrings. Use 'all' for every joint.",
    )
    parser.add_argument(
        "--episodes",
        nargs="*",
        type=int,
        default=None,
        help="Optional episode indices to visualize. Defaults to all episodes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/joint_variation"),
        help="Directory where plots and CSV summaries are written.",
    )
    parser.add_argument(
        "--max-overlay-episodes",
        type=int,
        default=80,
        help="Maximum episodes overlaid in trajectory plots.",
    )
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.dataset_root, args.repo_id)
    data_dir = dataset_root / "data/chunk-000"
    if not data_dir.exists():
        raise FileNotFoundError(f"Could not find parquet directory: {data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joint_names = load_joint_names(dataset_root)
    joint_indices = select_joint_indices(joint_names, args.joints)

    parquet_files = sorted(data_dir.glob("episode_*.parquet"))
    if args.episodes is not None:
        episode_set = set(args.episodes)
        parquet_files = [p for p in parquet_files if int(p.stem.split("_")[-1]) in episode_set]

    episodes = [load_episode_arrays(path) for path in parquet_files]

    rows = []
    for ep_idx, _timestamp, action, state in episodes:
        for joint_idx in joint_indices:
            joint_name = joint_names[joint_idx]
            for source, values in [("action", action[:, joint_idx]), ("state", state[:, joint_idx])]:
                row = {
                    "episode_index": ep_idx,
                    "joint": joint_name,
                    "source": source,
                    "length": len(values),
                }
                row.update(compute_metrics(values, action, state, joint_idx))
                rows.append(row)

    write_summary_csv(rows, args.output_dir / "joint_variation_summary.csv")

    for joint_idx in joint_indices:
        joint_name = joint_names[joint_idx]
        plot_joint_trajectories(
            joint_name,
            joint_idx,
            episodes,
            args.output_dir,
            max_overlay_episodes=args.max_overlay_episodes,
        )
        plot_joint_variation_bars(joint_name, rows, args.output_dir)

    print(f"Dataset: {dataset_root}")
    print(f"Episodes: {len(episodes)}")
    print(f"Joints: {[joint_names[i] for i in joint_indices]}")
    print(f"Wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
