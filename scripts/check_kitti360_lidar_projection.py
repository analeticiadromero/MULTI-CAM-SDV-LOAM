#!/usr/bin/env python3
"""Project KITTI-360 Velodyne points onto rectified camera images.

This script uses the same projection convention used by SDV-LOAM:

    p_cam = R * p_lidar + t
    u = fx * X / Z + cx
    v = fy * Y / Z + cy

The sensor file is expected to follow the SDV-LOAM format used in
src/sdv_loam/sensor/kitti_360*.txt.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_SEQUENCE = "2013_05_28_drive_0004_sync"
DEFAULT_IMAGE_SIZE = (1408, 376)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create LiDAR projection overlays for KITTI-360 rectified images."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="KITTI-360 root containing data_2d_raw and data_3d_raw.",
    )
    parser.add_argument(
        "--sequence",
        default=DEFAULT_SEQUENCE,
        help=f"KITTI-360 sequence directory. Default: {DEFAULT_SEQUENCE}",
    )
    parser.add_argument(
        "--frame",
        nargs="+",
        required=True,
        help="Frame ids, with or without .png/.bin extension, e.g. 0000001061.",
    )
    parser.add_argument(
        "--camera",
        nargs="+",
        choices=("image_00", "image_01"),
        default=("image_00", "image_01"),
        help="Camera(s) to project. Default: both image_00 and image_01.",
    )
    parser.add_argument(
        "--sensor",
        type=Path,
        help=(
            "Sensor file for single-camera mode. If omitted, defaults to "
            "src/sdv_loam/sensor/kitti_360_retif.txt for image_00 and "
            "src/sdv_loam/sensor/kitti_360_cam2_retif.txt for image_01."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("projection_overlays"),
        help="Directory where overlay PNGs will be saved.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=80.0,
        help="Depth in camera frame used to normalize colors. Default: 80 m.",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.2,
        help="Minimum positive camera-frame Z depth. Default: 0.2 m.",
    )
    parser.add_argument(
        "--point-radius",
        type=int,
        default=1,
        help="Overlay point radius in pixels. Default: 1.",
    )
    return parser.parse_args()


def normalize_frame_id(frame):
    return Path(frame).stem


def repo_root_from_script():
    return Path(__file__).resolve().parents[1]


def default_sensor_path(camera):
    sensor_dir = repo_root_from_script() / "sensor"
    if camera == "image_00":
        return sensor_dir / "kitti_360_retif.txt"
    if camera == "image_01":
        return sensor_dir / "kitti_360_cam2_retif.txt"
    raise ValueError(f"Unsupported camera: {camera}")


def load_sensor(path):
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 4:
        raise ValueError(f"Invalid sensor file: {path}")

    fx, fy, cx, cy = map(float, lines[0].split()[:4])
    transform_rows = [list(map(float, lines[i].split()[:4])) for i in range(1, 4)]
    transform = np.asarray(transform_rows, dtype=np.float64)
    return fx, fy, cx, cy, transform[:, :3], transform[:, 3]


def load_velodyne(path):
    points = np.fromfile(path, dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(f"Velodyne file does not contain Nx4 float32 points: {path}")
    return points.reshape((-1, 4))[:, :3].astype(np.float64)


def depth_color(depth, max_depth):
    ratio = np.clip(depth / max_depth, 0.0, 1.0)
    red = (255 * (1.0 - ratio)).astype(np.uint8)
    green = (255 * (1.0 - np.abs(ratio - 0.5) * 2.0)).astype(np.uint8)
    blue = (255 * ratio).astype(np.uint8)
    return np.stack([red, green, blue], axis=1)


def project_points(points_lidar, fx, fy, cx, cy, rotation, translation, min_depth):
    points_cam = points_lidar @ rotation.T + translation
    z = points_cam[:, 2]
    valid = z > min_depth
    points_cam = points_cam[valid]
    z = z[valid]

    u = fx * points_cam[:, 0] / z + cx
    v = fy * points_cam[:, 1] / z + cy
    return u, v, z


def draw_overlay(image_path, lidar_path, sensor_path, output_path, max_depth, min_depth, radius):
    fx, fy, cx, cy, rotation, translation = load_sensor(sensor_path)
    lidar_points = load_velodyne(lidar_path)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    u, v, z = project_points(
        lidar_points, fx, fy, cx, cy, rotation, translation, min_depth
    )
    in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[in_image]
    v = v[in_image]
    z = z[in_image]
    colors = depth_color(z, max_depth)

    draw = ImageDraw.Draw(image)
    for px, py, color in zip(u.astype(int), v.astype(int), colors):
        rgb = tuple(int(c) for c in color)
        if radius <= 0:
            draw.point((px, py), fill=rgb)
        else:
            draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius),
                fill=rgb,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return len(lidar_points), len(u)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.sensor and len(args.camera) != 1:
        raise SystemExit("--sensor can only be used with a single --camera.")

    for frame_arg in args.frame:
        frame = normalize_frame_id(frame_arg)
        lidar_path = (
            args.dataset
            / "data_3d_raw"
            / args.sequence
            / "velodyne_points"
            / "data"
            / f"{frame}.bin"
        )
        if not lidar_path.exists():
            raise FileNotFoundError(f"Missing Velodyne frame: {lidar_path}")

        for camera in args.camera:
            image_path = (
                args.dataset
                / "data_2d_raw"
                / args.sequence
                / camera
                / "data_rect"
                / f"{frame}.png"
            )
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image frame: {image_path}")

            sensor_path = args.sensor if args.sensor else default_sensor_path(camera)
            if not sensor_path.exists():
                raise FileNotFoundError(f"Missing sensor file: {sensor_path}")

            output_path = args.output_dir / f"{camera}_{frame}_lidar_overlay.png"
            total, projected = draw_overlay(
                image_path=image_path,
                lidar_path=lidar_path,
                sensor_path=sensor_path,
                output_path=output_path,
                max_depth=args.max_depth,
                min_depth=args.min_depth,
                radius=args.point_radius,
            )
            print(
                f"{camera} frame={frame}: projected {projected}/{total} points -> {output_path}"
            )


if __name__ == "__main__":
    main()
