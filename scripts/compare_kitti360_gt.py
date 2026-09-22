#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def load_kitti_poses(path, has_frame_id):
    poses = []
    frame_ids = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        values = [float(v) for v in line.split()]
        if has_frame_id:
            frame_ids.append(int(values[0]))
            values = values[1:]
        else:
            frame_ids.append(len(frame_ids))
        if len(values) != 12:
            raise ValueError(f"Expected 12 pose values in {path}, got {len(values)}")
        T = np.eye(4)
        T[:3, :4] = np.asarray(values, dtype=float).reshape(3, 4)
        poses.append(T)
    return frame_ids, poses


def load_cam_to_pose(path):
    transforms = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        name, values = line.split(":", 1)
        T = np.eye(4)
        T[:3, :4] = np.asarray([float(v) for v in values.split()]).reshape(3, 4)
        transforms[name.strip()] = T
    return transforms


def load_3x4(path):
    values = [float(v) for v in Path(path).read_text().split()]
    if len(values) != 12:
        raise ValueError(f"Expected 12 values in {path}, got {len(values)}")
    T = np.eye(4)
    T[:3, :4] = np.asarray(values, dtype=float).reshape(3, 4)
    return T


def to_relative(poses):
    if not poses:
        return []
    T0_inv = np.linalg.inv(poses[0])
    return [T0_inv @ T for T in poses]


def write_kitti(path, poses):
    with Path(path).open("w") as f:
        for T in poses:
            vals = T[:3, :4].reshape(-1)
            f.write(" ".join(f"{v:.9f}" for v in vals) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare SDV-LOAM KITTI-format predictions against KITTI-360 poses."
    )
    parser.add_argument("--pred", default="src/sdv_loam/output/pred.txt")
    parser.add_argument("--gt", required=True, help="KITTI-360 data_poses/.../poses.txt")
    parser.add_argument("--calib-cam-to-pose", required=True)
    parser.add_argument("--calib-cam-to-velo", required=True)
    parser.add_argument("--gt-start-index", type=int, default=0,
                        help="Index in the GT pose list matched to the first prediction pose.")
    parser.add_argument("--camera", default="image_00",
                        help="Camera key in calib_cam_to_pose used as GT reference.")
    parser.add_argument("--write-prefix", default="",
                        help="Optional prefix for writing relative pred/gt KITTI files.")
    args = parser.parse_args()

    _, pred = load_kitti_poses(args.pred, has_frame_id=False)
    gt_frame_ids, gt_pose = load_kitti_poses(args.gt, has_frame_id=True)
    cam_to_pose = load_cam_to_pose(args.calib_cam_to_pose)
    if args.camera not in cam_to_pose:
        raise ValueError(f"{args.camera} not found in {args.calib_cam_to_pose}")

    # KITTI-360 calib_cam_to_velo is cam0 -> velodyne. SDV-LOAM pred is velodyne pose.
    T_velo_cam0 = load_3x4(args.calib_cam_to_velo)
    T_cam0_velo = np.linalg.inv(T_velo_cam0)
    T_pose_cam = cam_to_pose[args.camera]
    T_pose_velo = T_pose_cam @ T_cam0_velo

    end = args.gt_start_index + len(pred)
    if end > len(gt_pose):
        raise ValueError(
            f"Prediction has {len(pred)} poses, but GT slice "
            f"[{args.gt_start_index}:{end}] exceeds {len(gt_pose)} poses."
        )

    gt_slice = gt_pose[args.gt_start_index:end]
    gt_velo = [T_map_pose @ T_pose_velo for T_map_pose in gt_slice]

    pred_rel = to_relative(pred)
    gt_rel = to_relative(gt_velo)

    errors = np.asarray([
        np.linalg.norm(pred_rel[i][:3, 3] - gt_rel[i][:3, 3])
        for i in range(len(pred_rel))
    ])

    print(f"pred poses: {len(pred)}")
    print(f"gt poses: {len(gt_pose)}")
    print(f"gt matched frames: {gt_frame_ids[args.gt_start_index]} -> {gt_frame_ids[end - 1]}")
    print(f"ATE RMSE: {np.sqrt(np.mean(errors ** 2)):.6f} m")
    print(f"ATE mean: {np.mean(errors):.6f} m")
    print(f"ATE median: {np.median(errors):.6f} m")
    print(f"ATE max: {np.max(errors):.6f} m")

    if args.write_prefix:
        write_kitti(f"{args.write_prefix}_pred_rel.txt", pred_rel)
        write_kitti(f"{args.write_prefix}_gt_velo_rel.txt", gt_rel)


if __name__ == "__main__":
    main()
