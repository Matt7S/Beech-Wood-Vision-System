"""
Data Cleaning Script – Step 4 of the Beech-Wood Vision System pipeline.

Scans an input directory of raw images, discards blurry or too-dark/too-bright
frames, and copies the accepted images to an output directory resized to the
target resolution.

Sharpness is measured with the Laplacian variance method: a high variance means
a sharp image; a low variance means a blurry one.

Usage
-----
    python src/preprocessing/data_cleaning.py \
        --input  data/raw \
        --output data/clean \
        --size   640
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter blurry/poorly-lit images and resize to target resolution."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join("data", "raw"),
        help="Root directory that is searched recursively for images (default: data/raw).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("data", "clean"),
        help="Directory where accepted images are written (default: data/clean).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=640,
        help="Target size in pixels; images are resized to SIZE×SIZE (default: 640).",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100.0,
        help=(
            "Laplacian variance threshold. Images with variance below this value are "
            "considered blurry and discarded (default: 100)."
        ),
    )
    parser.add_argument(
        "--min-brightness",
        type=float,
        default=20.0,
        help="Minimum mean brightness (0–255); darker images are discarded (default: 20).",
    )
    parser.add_argument(
        "--max-brightness",
        type=float,
        default=235.0,
        help="Maximum mean brightness (0–255); brighter images are discarded (default: 235).",
    )
    return parser


def laplacian_variance(gray: "cv2.Mat") -> float:
    """Return the variance of the Laplacian – a proxy for image sharpness."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_brightness(gray: "cv2.Mat") -> float:
    """Return the mean pixel value of a grayscale image."""
    return float(np.mean(gray))


def is_acceptable(
    image_path: Path,
    blur_threshold: float,
    min_brightness: float,
    max_brightness: float,
) -> Tuple[bool, str]:
    """Return (True, '') if the image passes all quality checks, otherwise (False, reason)."""
    img = cv2.imread(str(image_path))
    if img is None:
        return False, "unreadable"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lap_var = laplacian_variance(gray)
    if lap_var < blur_threshold:
        return False, f"blurry (Laplacian var={lap_var:.1f} < {blur_threshold})"

    brightness = mean_brightness(gray)
    if brightness < min_brightness:
        return False, f"too dark (mean={brightness:.1f} < {min_brightness})"
    if brightness > max_brightness:
        return False, f"too bright (mean={brightness:.1f} > {max_brightness})"

    return True, ""


def clean_dataset(
    input_dir: str,
    output_dir: str,
    size: int,
    blur_threshold: float,
    min_brightness: float,
    max_brightness: float,
) -> None:
    """Walk *input_dir* recursively, filter images and write resized copies to *output_dir*."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"ERROR: Input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    image_files = [
        p for p in input_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print(f"No images found in '{input_dir}'.")
        return

    accepted = 0
    rejected = 0

    for img_path in sorted(image_files):
        ok, reason = is_acceptable(img_path, blur_threshold, min_brightness, max_brightness)
        if not ok:
            print(f"  REJECTED  {img_path.name}  – {reason}")
            rejected += 1
            continue

        # Preserve subdirectory structure relative to input root
        relative = img_path.relative_to(input_path)
        dest = output_path / relative
        dest.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_path))
        resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dest), resized)
        print(f"  accepted  {img_path.name}  → {dest}")
        accepted += 1

    total = accepted + rejected
    print(
        f"\nDone. {accepted}/{total} images accepted "
        f"({rejected} rejected). Output: {output_dir}"
    )


def main() -> None:
    args = build_parser().parse_args()
    clean_dataset(
        input_dir=args.input,
        output_dir=args.output,
        size=args.size,
        blur_threshold=args.blur_threshold,
        min_brightness=args.min_brightness,
        max_brightness=args.max_brightness,
    )


if __name__ == "__main__":
    main()
