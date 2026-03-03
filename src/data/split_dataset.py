"""
Dataset Split Script – Step 6 of the Beech-Wood Vision System pipeline.

Randomly splits a directory of clean (optionally labeled) images into
training, validation and test sub-sets and writes them to the standard
YOLO-style folder layout:

    data/dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

Label files (*.txt) with the same stem as the image are copied alongside the
image when they exist; the script works fine without labels too.

Usage
-----
    python src/data/split_dataset.py \
        --input  data/clean \
        --output data/dataset \
        --train  0.75 \
        --val    0.15 \
        --test   0.10 \
        --seed   42
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Optional


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split a clean image dataset into train / val / test subsets."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join("data", "clean"),
        help="Directory containing the cleaned images (default: data/clean).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("data", "dataset"),
        help="Root output directory for the split dataset (default: data/dataset).",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.75,
        help="Fraction of images for training (default: 0.75).",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.15,
        help="Fraction of images for validation (default: 0.15).",
    )
    parser.add_argument(
        "--test",
        type=float,
        default=0.10,
        help="Fraction of images for testing (default: 0.10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help=(
            "Directory containing YOLO label (.txt) files.  When omitted, the script "
            "looks for label files next to each image with the same stem."
        ),
    )
    return parser


def _copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dest))


def split_dataset(
    input_dir: str,
    output_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    labels_dir: Optional[str] = None,
) -> None:
    """Perform the train/val/test split and copy files to *output_dir*."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        print(
            f"ERROR: train + val + test must sum to 1.0, got {total:.4f}.",
            file=sys.stderr,
        )
        sys.exit(1)

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"ERROR: Input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    images: List[Path] = sorted(
        p for p in input_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not images:
        print(f"No images found in '{input_dir}'.")
        return

    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    output_path = Path(output_dir)
    labels_path = Path(labels_dir) if labels_dir else None

    for split_name, split_images in splits.items():
        for img_path in split_images:
            dest_img = output_path / split_name / "images" / img_path.name
            _copy(img_path, dest_img)

            # Try to find a matching label file
            if labels_path:
                label_src = labels_path / img_path.with_suffix(".txt").name
            else:
                label_src = img_path.with_suffix(".txt")

            if label_src.exists():
                dest_lbl = output_path / split_name / "labels" / label_src.name
                _copy(label_src, dest_lbl)

    counts = {k: len(v) for k, v in splits.items()}
    print(
        f"Split complete: {n} images → "
        f"train={counts['train']}, val={counts['val']}, test={counts['test']}\n"
        f"Output: {output_dir}"
    )


def main() -> None:
    args = build_parser().parse_args()
    split_dataset(
        input_dir=args.input,
        output_dir=args.output,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
        labels_dir=args.labels,
    )


if __name__ == "__main__":
    main()
