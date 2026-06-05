#!/usr/bin/env python

import argparse
import copy
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

EE_FEATURE_NAMES = [
    "left_ee.x",
    "left_ee.y",
    "left_ee.z",
    "left_ee.roll",
    "left_ee.pitch",
    "left_ee.yaw",
    "left_gripper.pos",
    "right_ee.x",
    "right_ee.y",
    "right_ee.z",
    "right_ee.roll",
    "right_ee.pitch",
    "right_ee.yaw",
    "right_gripper.pos",
]


def parse_vec(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(value) for value in text.split()], dtype=np.float64)


def rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    c1 = 1 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=np.float64,
    )


def transform_from_origin(origin: ET.Element | None) -> np.ndarray:
    xyz = parse_vec(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
    rpy = parse_vec(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_from_rpy(rpy)
    transform[:3, 3] = xyz
    return transform


def transform_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_from_axis_angle(axis, angle)
    return transform


def rpy_from_rotation(rotation: np.ndarray) -> np.ndarray:
    sy = -rotation[2, 0]
    pitch = math.asin(max(-1.0, min(1.0, sy)))
    cp = math.cos(pitch)
    if abs(cp) > 1e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


class UrdfForwardKinematics:
    def __init__(
        self,
        urdf_path: Path,
        target_link: str,
        joint_names: list[str],
        base_link: str = "base_link",
    ) -> None:
        self.joint_names = joint_names
        self.chain = self._load_chain(urdf_path, base_link, target_link)

    def _load_chain(self, urdf_path: Path, base_link: str, target_link: str) -> list[dict]:
        root = ET.parse(urdf_path).getroot()
        joints_by_child = {}
        for joint in root.findall("joint"):
            child = joint.find("child")
            parent = joint.find("parent")
            if child is None or parent is None:
                continue
            joints_by_child[child.get("link")] = {
                "name": joint.get("name"),
                "type": joint.get("type", "fixed"),
                "parent": parent.get("link"),
                "child": child.get("link"),
                "origin": transform_from_origin(joint.find("origin")),
                "axis": parse_vec(joint.find("axis").get("xyz") if joint.find("axis") is not None else None, (0, 0, 1)),
            }

        chain = []
        link = target_link
        while link != base_link:
            if link not in joints_by_child:
                raise ValueError(f"Cannot find a joint chain from {base_link!r} to {target_link!r}")
            joint = joints_by_child[link]
            chain.append(joint)
            link = joint["parent"]
        chain.reverse()
        return chain

    def pose(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        joint_values = {
            name: math.radians(float(value)) for name, value in zip(self.joint_names, joint_pos_deg, strict=True)
        }
        transform = np.eye(4, dtype=np.float64)
        for joint in self.chain:
            transform = transform @ joint["origin"]
            if joint["type"] in {"revolute", "continuous"}:
                transform = transform @ transform_from_axis_angle(joint["axis"], joint_values[joint["name"]])
        xyz = transform[:3, 3]
        rpy = rpy_from_rotation(transform[:3, :3])
        return np.concatenate([xyz, rpy])


def joints_to_ee(values: list[list[float]], fk: UrdfForwardKinematics) -> np.ndarray:
    out = np.empty((len(values), 14), dtype=np.float32)
    for i, row in enumerate(values):
        joints = np.asarray(row, dtype=np.float64)
        left_pose = fk.pose(joints[0:5])
        right_pose = fk.pose(joints[6:11])
        out[i, 0:6] = left_pose
        out[i, 6] = joints[5]
        out[i, 7:13] = right_pose
        out[i, 13] = joints[11]
    return out


def fixed_size_list_array(values: np.ndarray) -> pa.FixedSizeListArray:
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), values.shape[1])


def numeric_stats(values: np.ndarray) -> dict:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items))


def replace_vector_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"Missing required parquet column: {name}")
    return table.set_column(index, name, fixed_size_list_array(values))


def convert_info(info: dict, urdf_path: Path, target_link: str, joint_names: list[str]) -> dict:
    info = copy.deepcopy(info)
    for key in ["action", "observation.state"]:
        info["features"][key]["shape"] = [14]
        info["features"][key]["names"] = EE_FEATURE_NAMES
        info["features"][key]["info"] = {
            "space": "end_effector",
            "position_unit": "m",
            "orientation": "rpy",
            "orientation_unit": "rad",
            "gripper_unit": "source_dataset_units",
            "source_joint_unit": "deg",
            "urdf_path": str(urdf_path),
            "target_link": target_link,
            "joint_names": joint_names,
        }
    return info


def convert_dataset(
    src: Path,
    dst: Path,
    urdf: Path,
    target_link: str,
    joint_names: list[str],
) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source does not exist: {src}")
    if not urdf.is_file():
        raise FileNotFoundError(f"URDF does not exist: {urdf}")
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    fk = UrdfForwardKinematics(urdf, target_link, joint_names)
    shutil.copytree(src, dst)

    episodes = read_jsonl(src / "meta/episodes.jsonl")
    old_stats = {item["episode_index"]: item for item in read_jsonl(src / "meta/episodes_stats.jsonl")}
    new_stats = []

    for episode in episodes:
        episode_index = episode["episode_index"]
        parquet_path = dst / f"data/chunk-000/episode_{episode_index:06d}.parquet"
        table = pq.read_table(parquet_path)
        action_ee = joints_to_ee(table["action"].to_pylist(), fk)
        state_ee = joints_to_ee(table["observation.state"].to_pylist(), fk)

        table = replace_vector_column(table, "action", action_ee)
        table = replace_vector_column(table, "observation.state", state_ee)
        pq.write_table(table.replace_schema_metadata(None), parquet_path)

        episode_stats = copy.deepcopy(old_stats[episode_index])
        episode_stats["stats"]["action"] = numeric_stats(action_ee)
        episode_stats["stats"]["observation.state"] = numeric_stats(state_ee)
        new_stats.append(episode_stats)

    info = convert_info(json.loads((src / "meta/info.json").read_text()), urdf, target_link, joint_names)
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n")
    write_jsonl(dst / "meta/episodes_stats.jsonl", new_stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SO101 joint-space LeRobotDataset columns to EE space.")
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--target-link", default="gripper_frame_link")
    parser.add_argument("--joint-names", nargs=5, default=DEFAULT_JOINT_NAMES)
    args = parser.parse_args()

    convert_dataset(args.src, args.dst, args.urdf, args.target_link, args.joint_names)
    info = json.loads((args.dst / "meta/info.json").read_text())
    print(f"created {args.dst}")
    print(f"episodes: {info['total_episodes']} frames: {info['total_frames']}")
    print("converted action and observation.state to 14-D EE vectors")


if __name__ == "__main__":
    main()
