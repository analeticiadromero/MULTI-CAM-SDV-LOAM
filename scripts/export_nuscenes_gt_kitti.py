#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def quat_wxyz_to_rot(q):
    w, x, y, z = q
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("Zero quaternion")
    w, x, y, z = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def transform_from_record(record):
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_rot(np.asarray(record["rotation"], dtype=float))
    T[:3, 3] = np.asarray(record["translation"], dtype=float)
    return T


def load_json(root, name):
    with (root / name).open() as f:
        return json.load(f)


def scene_sample_tokens(samples_by_token, scene):
    tokens = []
    token = scene["first_sample_token"]
    while token:
        tokens.append(token)
        token = samples_by_token[token]["next"]
    return set(tokens)


def to_relative(poses):
    if not poses:
        return []
    T0_inv = np.linalg.inv(poses[0])
    return [T0_inv @ T for T in poses]


def write_kitti(path, poses):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for T in poses:
            values = T[:3, :4].reshape(-1)
            f.write(" ".join(f"{v:.9f}" for v in values) + "\n")


def write_timestamps(path, timestamps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for timestamp in timestamps:
            f.write(f"{timestamp / 1e6:.9f}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Export nuScenes ego/sensor ground truth to KITTI 3x4 pose format."
    )
    parser.add_argument("--dataroot",
                        default="/media/analeticiaromero/ExtremeSSD/ANA/nuScenes/v1.0-mini",
                        help="nuScenes root containing samples/, sweeps/ and metadata folder.")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene-name", default="scene-0061")
    parser.add_argument("--sensor", default="LIDAR_TOP",
                        help="Sensor pose to export, e.g. LIDAR_TOP or CAM_FRONT.")
    parser.add_argument("--keyframes-only", action="store_true",
                        help="Use only keyframes. Default exports all sample_data for the sensor.")
    parser.add_argument("--absolute", action="store_true",
                        help="Write global poses instead of poses relative to the first selected pose.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0,
                        help="Maximum number of poses to write after start-index. 0 writes all.")
    parser.add_argument("--output",
                        default="src/sdv_loam/output/nuscenes_0061_lidar_gt_rel.txt")
    parser.add_argument("--timestamps-output",
                        default="src/sdv_loam/output/nuscenes_0061_lidar_gt_timestamps.txt")
    args = parser.parse_args()

    metadata_root = Path(args.dataroot) / args.version
    scenes = load_json(metadata_root, "scene.json")
    samples = load_json(metadata_root, "sample.json")
    sample_data = load_json(metadata_root, "sample_data.json")
    sensors = load_json(metadata_root, "sensor.json")
    calibrated_sensors = load_json(metadata_root, "calibrated_sensor.json")
    ego_poses = load_json(metadata_root, "ego_pose.json")

    scene = next((s for s in scenes if s["name"] == args.scene_name), None)
    if scene is None:
        raise ValueError(f"Scene {args.scene_name} not found in {metadata_root}")

    samples_by_token = {s["token"]: s for s in samples}
    scene_tokens = scene_sample_tokens(samples_by_token, scene)
    sensors_by_token = {s["token"]: s for s in sensors}
    calib_by_token = {c["token"]: c for c in calibrated_sensors}
    ego_by_token = {e["token"]: e for e in ego_poses}

    selected = []
    for record in sample_data:
        if record["sample_token"] not in scene_tokens:
            continue
        calib = calib_by_token[record["calibrated_sensor_token"]]
        sensor_name = sensors_by_token[calib["sensor_token"]]["channel"]
        if sensor_name != args.sensor:
            continue
        if args.keyframes_only and not record["is_key_frame"]:
            continue
        selected.append(record)

    selected.sort(key=lambda r: r["timestamp"])
    if args.start_index < 0 or args.start_index >= len(selected):
        raise ValueError(f"start-index {args.start_index} outside selected pose count {len(selected)}")

    selected = selected[args.start_index:]
    if args.limit > 0:
        selected = selected[:args.limit]

    poses = []
    timestamps = []
    for record in selected:
        T_global_ego = transform_from_record(ego_by_token[record["ego_pose_token"]])
        T_ego_sensor = transform_from_record(calib_by_token[record["calibrated_sensor_token"]])
        poses.append(T_global_ego @ T_ego_sensor)
        timestamps.append(record["timestamp"])

    output_poses = poses if args.absolute else to_relative(poses)
    write_kitti(args.output, output_poses)
    if args.timestamps_output:
        write_timestamps(args.timestamps_output, timestamps)

    mode = "absolute" if args.absolute else "relative"
    print(f"scene: {args.scene_name}")
    print(f"sensor: {args.sensor}")
    print(f"poses written: {len(output_poses)}")
    print(f"mode: {mode}")
    print(f"output: {args.output}")
    if args.timestamps_output:
        print(f"timestamps: {args.timestamps_output}")


if __name__ == "__main__":
    main()
