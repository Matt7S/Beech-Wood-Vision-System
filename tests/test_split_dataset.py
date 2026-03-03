"""
Tests for src/data/split_dataset.py
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.split_dataset import split_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _write_label(path: Path, content: str = "0 0.5 0.5 1.0 1.0\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSplitDataset:
    def _create_images(self, directory: Path, n: int) -> None:
        for i in range(n):
            _write_image(directory / f"img_{i:04d}.jpg")

    def test_correct_counts_20_images(self, tmp_path):
        input_dir = tmp_path / "clean"
        output_dir = tmp_path / "dataset"
        self._create_images(input_dir, 20)

        split_dataset(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            train_ratio=0.75,
            val_ratio=0.15,
            test_ratio=0.10,
            seed=42,
        )

        train_imgs = list((output_dir / "train" / "images").glob("*.jpg"))
        val_imgs = list((output_dir / "val" / "images").glob("*.jpg"))
        test_imgs = list((output_dir / "test" / "images").glob("*.jpg"))

        assert len(train_imgs) == 15
        assert len(val_imgs) == 3
        assert len(test_imgs) == 2

    def test_no_overlap_between_splits(self, tmp_path):
        input_dir = tmp_path / "clean"
        output_dir = tmp_path / "dataset"
        self._create_images(input_dir, 30)

        split_dataset(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            seed=0,
        )

        names = {}
        for split in ("train", "val", "test"):
            names[split] = {p.name for p in (output_dir / split / "images").glob("*.jpg")}

        assert names["train"].isdisjoint(names["val"])
        assert names["train"].isdisjoint(names["test"])
        assert names["val"].isdisjoint(names["test"])

    def test_labels_copied_alongside_images(self, tmp_path):
        input_dir = tmp_path / "clean"
        output_dir = tmp_path / "dataset"

        for i in range(10):
            _write_image(input_dir / f"img_{i:04d}.jpg")
            _write_label(input_dir / f"img_{i:04d}.txt")

        split_dataset(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=1,
        )

        for split in ("train", "val", "test"):
            imgs = list((output_dir / split / "images").glob("*.jpg"))
            lbls = list((output_dir / split / "labels").glob("*.txt"))
            assert len(imgs) == len(lbls), f"{split}: images={len(imgs)}, labels={len(lbls)}"

    def test_invalid_ratio_sum_exits(self, tmp_path):
        input_dir = tmp_path / "clean"
        output_dir = tmp_path / "dataset"
        self._create_images(input_dir, 5)

        with pytest.raises(SystemExit):
            split_dataset(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                train_ratio=0.5,
                val_ratio=0.5,
                test_ratio=0.5,
                seed=42,
            )

    def test_missing_input_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            split_dataset(
                input_dir=str(tmp_path / "nonexistent"),
                output_dir=str(tmp_path / "out"),
                train_ratio=0.75,
                val_ratio=0.15,
                test_ratio=0.10,
                seed=42,
            )

    def test_reproducibility_with_same_seed(self, tmp_path):
        input_dir = tmp_path / "clean"
        self._create_images(input_dir, 20)

        out1 = tmp_path / "ds1"
        out2 = tmp_path / "ds2"

        for out in (out1, out2):
            split_dataset(
                input_dir=str(input_dir),
                output_dir=str(out),
                train_ratio=0.75,
                val_ratio=0.15,
                test_ratio=0.10,
                seed=99,
            )

        names1 = {p.name for p in (out1 / "train" / "images").glob("*.jpg")}
        names2 = {p.name for p in (out2 / "train" / "images").glob("*.jpg")}
        assert names1 == names2
