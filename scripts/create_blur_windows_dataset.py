#!/usr/bin/env python3
import argparse
import csv
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

try:
    import numpy as np
except ImportError:
    np = None


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_interval(value):
    try:
        start_text, end_text = value.split(":", 1)
        start = int(start_text)
        end = int(end_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid interval '{value}'. Use START:END, e.g. 300:500."
        ) from exc

    if start < 0 or end < start:
        raise argparse.ArgumentTypeError(
            f"Invalid interval '{value}'. Expected 0 <= START <= END."
        )
    return start, end


def list_images(source):
    images = sorted(
        p for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"No images found in {source}")
    return images


def fixed_window(index, intervals):
    return any(start <= index <= end for start, end in intervals)


def periodic_window(index, period, length, offset):
    if period <= 0 or length <= 0:
        return False
    return ((index - offset) % period) < length


def make_random_intervals(frame_count, count, min_length, max_length, rng):
    if count <= 0:
        return []

    intervals = []
    max_length = max(min_length, max_length)
    for _ in range(count):
        length = rng.randint(min_length, max_length)
        if length >= frame_count:
            intervals.append((0, frame_count - 1))
            continue
        start = rng.randint(0, frame_count - length)
        intervals.append((start, start + length - 1))
    return intervals


def should_degrade(index, intervals, period, length, offset):
    return fixed_window(index, intervals) or periodic_window(index, period, length, offset)


def clamp_int(value):
    return max(0, min(255, int(round(value))))


def apply_blur_dark(img, radius, brightness, contrast):
    degraded = img.filter(ImageFilter.GaussianBlur(radius=radius))
    degraded = ImageEnhance.Brightness(degraded).enhance(brightness)
    degraded = ImageEnhance.Contrast(degraded).enhance(contrast)
    return degraded


def apply_noise(img, sigma, rng):
    if np is not None:
        array = np.asarray(img, dtype=np.float32)
        noise_rng = np.random.default_rng(rng.randrange(0, 2**32))
        noise = noise_rng.normal(0.0, sigma, array.shape)
        noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy, mode="RGB")

    # Pillow effect_noise is deterministic only through the global RNG state, so
    # roll our own lightweight RGB perturbation for reproducible experiments.
    pixels = img.load()
    width, height = img.size
    noisy = Image.new("RGB", img.size)
    noisy_pixels = noisy.load()
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            noisy_pixels[x, y] = (
                clamp_int(r + rng.gauss(0.0, sigma)),
                clamp_int(g + rng.gauss(0.0, sigma)),
                clamp_int(b + rng.gauss(0.0, sigma)),
            )
    return noisy


def apply_occlusion(img, coverage, rng, rectangles, color):
    occluded = img.copy()
    draw = ImageDraw.Draw(occluded)
    width, height = occluded.size
    target_area = max(0.0, min(0.95, coverage)) * width * height
    drawn_area = 0.0
    rect_count = max(1, rectangles)

    for _ in range(rect_count):
        remaining = max(1.0, target_area - drawn_area)
        rect_area = remaining / float(rect_count)
        rect_w = int(max(8, min(width, (rect_area * rng.uniform(1.0, 2.5)) ** 0.5)))
        rect_h = int(max(8, min(height, rect_area / max(1, rect_w))))
        x0 = rng.randint(0, max(0, width - rect_w))
        y0 = rng.randint(0, max(0, height - rect_h))
        draw.rectangle([x0, y0, x0 + rect_w, y0 + rect_h], fill=color)
        drawn_area += rect_w * rect_h
        if drawn_area >= target_area:
            break

    return occluded


def apply_degradation(img, effect, args, rng):
    rgb = img.convert("RGB")
    if effect == "blur":
        return rgb.filter(ImageFilter.GaussianBlur(radius=args.radius))
    if effect == "blur_dark":
        return apply_blur_dark(rgb, args.radius, args.brightness, args.contrast)
    if effect == "noise":
        return apply_noise(rgb, args.noise_sigma, rng)
    if effect == "occlusion":
        return apply_occlusion(rgb, args.occlusion, rng, args.occlusion_rectangles, args.occlusion_color)
    if effect == "mixed":
        degraded = apply_blur_dark(rgb, args.radius, args.brightness, args.contrast)
        degraded = apply_noise(degraded, args.noise_sigma, rng)
        degraded = apply_occlusion(degraded, args.occlusion, rng, args.occlusion_rectangles, args.occlusion_color)
        return degraded
    raise ValueError(f"Unknown effect: {effect}")


def copy_or_degrade_folder(source, output, args, rng, camera_name=""):
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output}. Use --overwrite if this is intentional."
            )
        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=True)
    images = list_images(source)
    intervals = list(args.interval)
    intervals.extend(
        make_random_intervals(
            frame_count=len(images),
            count=args.random_windows,
            min_length=args.random_min_length,
            max_length=args.random_max_length,
            rng=rng,
        )
    )

    manifest_name = "manifest_degradation.csv" if not camera_name else f"manifest_degradation_{camera_name}.csv"
    manifest_path = output / manifest_name
    degraded_count = 0
    copied_count = 0

    with manifest_path.open("w", newline="") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow([
            "camera",
            "index",
            "filename",
            "degraded",
            "effect",
            "intervals",
            "period",
            "length",
            "offset",
            "radius",
            "brightness",
            "contrast",
            "noise_sigma",
            "occlusion",
            "occlusion_rectangles",
        ])

        for index, image_path in enumerate(images):
            target = output / image_path.name
            degrade = should_degrade(index, intervals, args.period, args.length, args.offset)

            if degrade:
                with Image.open(image_path) as img:
                    apply_degradation(img, args.effect, args, rng).save(target)
                degraded_count += 1
            else:
                shutil.copy2(image_path, target)
                copied_count += 1

            writer.writerow([
                camera_name,
                index,
                image_path.name,
                1 if degrade else 0,
                args.effect,
                ";".join(f"{start}:{end}" for start, end in intervals),
                args.period,
                args.length,
                args.offset,
                args.radius,
                args.brightness,
                args.contrast,
                args.noise_sigma,
                args.occlusion,
                args.occlusion_rectangles,
            ])

    readme = output / "README_degradation.txt"
    readme.write_text(
        "Image degradation dataset\n"
        f"source: {source}\n"
        f"output: {output}\n"
        f"camera: {camera_name}\n"
        f"frames: {len(images)}\n"
        f"degraded_frames: {degraded_count}\n"
        f"copied_frames: {copied_count}\n"
        f"effect: {args.effect}\n"
        f"intervals: {intervals}\n"
        f"period: {args.period}\n"
        f"length: {args.length}\n"
        f"offset: {args.offset}\n"
        f"seed: {args.seed}\n"
    )

    return len(images), degraded_count, copied_count, manifest_path


def build_jobs(args):
    if args.source_root or args.output_root or args.camera:
        if not args.source_root or not args.output_root or not args.camera:
            raise ValueError("--source-root, --output-root and at least one --camera must be used together.")
        jobs = []
        for camera in args.camera:
            source = Path(args.source_root) / camera / args.camera_subdir
            output = Path(args.output_root) / camera / args.camera_subdir
            jobs.append((camera, source, output))
        return jobs

    if not args.source or not args.output:
        raise ValueError("Use --source/--output for one folder or --source-root/--output-root/--camera for KITTI-style roots.")
    return [("", Path(args.source), Path(args.output))]


def main():
    parser = argparse.ArgumentParser(
        description="Create independent image folders with controlled degradation windows."
    )
    parser.add_argument("--source", help="Source image directory for single-camera mode.")
    parser.add_argument("--output", help="Output image directory for single-camera mode.")
    parser.add_argument("--source-root", help="Dataset root containing camera folders, e.g. image_00.")
    parser.add_argument("--output-root", help="Output dataset root for multi-camera mode.")
    parser.add_argument("--camera", action="append", default=[],
                        help="Camera folder to process in multi-camera mode. Repeat for multiple cameras.")
    parser.add_argument("--camera-subdir", default="data_rect",
                        help="Image subdirectory inside each camera folder.")
    parser.add_argument("--effect", choices=["blur", "blur_dark", "noise", "occlusion", "mixed"],
                        default="blur_dark", help="Degradation effect.")
    parser.add_argument("--interval", type=parse_interval, action="append", default=[],
                        help="Fixed degraded interval START:END. Repeat for multiple windows.")
    parser.add_argument("--period", type=int, default=0,
                        help="Frame period for periodic degradation windows. 0 disables periodic mode.")
    parser.add_argument("--length", type=int, default=0,
                        help="Number of degraded frames in each period.")
    parser.add_argument("--offset", type=int, default=0,
                        help="Frame index offset where the first periodic window starts.")
    parser.add_argument("--random-windows", type=int, default=0,
                        help="Number of random degradation windows.")
    parser.add_argument("--random-min-length", type=int, default=50,
                        help="Minimum random window length.")
    parser.add_argument("--random-max-length", type=int, default=150,
                        help="Maximum random window length.")
    parser.add_argument("--radius", type=float, default=7.0,
                        help="Gaussian blur radius.")
    parser.add_argument("--brightness", type=float, default=0.55,
                        help="Brightness multiplier for blur_dark/mixed.")
    parser.add_argument("--contrast", type=float, default=0.65,
                        help="Contrast multiplier for blur_dark/mixed.")
    parser.add_argument("--noise-sigma", type=float, default=25.0,
                        help="Gaussian noise sigma in intensity units.")
    parser.add_argument("--occlusion", type=float, default=0.45,
                        help="Approximate image area covered by occlusion rectangles, 0..0.95.")
    parser.add_argument("--occlusion-rectangles", type=int, default=2,
                        help="Number of occlusion rectangles per degraded image.")
    parser.add_argument("--occlusion-color", type=int, nargs=3, default=(0, 0, 0),
                        metavar=("R", "G", "B"), help="RGB color for occlusion.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed for reproducible degradation.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow writing into an existing output directory.")
    args = parser.parse_args()

    if not args.interval and args.period <= 0 and args.random_windows <= 0:
        raise ValueError("No degradation window selected. Use --interval, --period/--length, or --random-windows.")

    rng = random.Random(args.seed)
    jobs = build_jobs(args)
    totals = {"frames": 0, "degraded": 0, "copied": 0}

    for camera, source, output in jobs:
        frames, degraded, copied, manifest_path = copy_or_degrade_folder(source, output, args, rng, camera)
        totals["frames"] += frames
        totals["degraded"] += degraded
        totals["copied"] += copied
        print(f"camera: {camera or '(single)'}")
        print(f"  source: {source}")
        print(f"  output: {output}")
        print(f"  frames: {frames}")
        print(f"  degraded_frames: {degraded}")
        print(f"  copied_frames: {copied}")
        print(f"  manifest: {manifest_path}")

    print("summary:")
    print(f"  frames: {totals['frames']}")
    print(f"  degraded_frames: {totals['degraded']}")
    print(f"  copied_frames: {totals['copied']}")


if __name__ == "__main__":
    main()
