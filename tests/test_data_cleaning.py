"""
Tests for src/preprocessing/data_cleaning.py
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure repo root is on sys.path so imports work without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.data_cleaning import (
    is_acceptable,
    laplacian_variance,
    mean_brightness,
    clean_dataset,
)


# ---------------------------------------------------------------------------
# Helper: create a synthetic image on disk
# ---------------------------------------------------------------------------

def _write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _sharp_bright_image() -> np.ndarray:
    """Return a high-contrast checkerboard (sharp, mid-brightness)."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    for i in range(64):
        for j in range(64):
            img[i, j] = 255 if (i // 8 + j // 8) % 2 == 0 else 0
    return img


def _blurry_image() -> np.ndarray:
    """Return a heavily blurred uniform-grey image."""
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    img = cv2.GaussianBlur(img, (21, 21), 0)
    return img


def _dark_image() -> np.ndarray:
    return np.full((64, 64, 3), 5, dtype=np.uint8)


def _bright_image() -> np.ndarray:
    return np.full((64, 64, 3), 250, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestLaplacianVariance:
    def test_sharp_image_has_high_variance(self):
        gray = cv2.cvtColor(_sharp_bright_image(), cv2.COLOR_BGR2GRAY)
        assert laplacian_variance(gray) > 100

    def test_blurry_image_has_low_variance(self):
        gray = cv2.cvtColor(_blurry_image(), cv2.COLOR_BGR2GRAY)
        assert laplacian_variance(gray) < 10


class TestMeanBrightness:
    def test_dark_image(self):
        gray = cv2.cvtColor(_dark_image(), cv2.COLOR_BGR2GRAY)
        assert mean_brightness(gray) < 20

    def test_bright_image(self):
        gray = cv2.cvtColor(_bright_image(), cv2.COLOR_BGR2GRAY)
        assert mean_brightness(gray) > 235


# ---------------------------------------------------------------------------
# Unit tests for is_acceptable
# ---------------------------------------------------------------------------

class TestIsAcceptable:
    def test_sharp_mid_brightness_accepted(self, tmp_path):
        p = tmp_path / "sharp.jpg"
        _write_image(p, _sharp_bright_image())
        ok, reason = is_acceptable(p, blur_threshold=100, min_brightness=20, max_brightness=235)
        assert ok
        assert reason == ""

    def test_blurry_rejected(self, tmp_path):
        p = tmp_path / "blurry.jpg"
        _write_image(p, _blurry_image())
        ok, reason = is_acceptable(p, blur_threshold=100, min_brightness=20, max_brightness=235)
        assert not ok
        assert "blurry" in reason

    def test_dark_rejected(self, tmp_path):
        p = tmp_path / "dark.jpg"
        _write_image(p, _dark_image())
        # blur_threshold=0 so only the brightness check fires
        ok, reason = is_acceptable(p, blur_threshold=0, min_brightness=20, max_brightness=235)
        assert not ok
        assert "dark" in reason

    def test_bright_rejected(self, tmp_path):
        p = tmp_path / "bright.jpg"
        _write_image(p, _bright_image())
        # blur_threshold=0 so only the brightness check fires
        ok, reason = is_acceptable(p, blur_threshold=0, min_brightness=20, max_brightness=235)
        assert not ok
        assert "bright" in reason

    def test_unreadable_file_rejected(self, tmp_path):
        p = tmp_path / "bad.jpg"
        p.write_bytes(b"not an image")
        ok, reason = is_acceptable(p, blur_threshold=100, min_brightness=20, max_brightness=235)
        assert not ok
        assert "unreadable" in reason


# ---------------------------------------------------------------------------
# Integration test for clean_dataset
# ---------------------------------------------------------------------------

class TestCleanDataset:
    def test_only_acceptable_images_copied_and_resized(self, tmp_path):
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "clean"

        _write_image(input_dir / "good.jpg", _sharp_bright_image())
        _write_image(input_dir / "blurry.jpg", _blurry_image())

        clean_dataset(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            size=32,
            blur_threshold=100,
            min_brightness=20,
            max_brightness=235,
        )

        output_files = list(output_dir.rglob("*.jpg"))
        assert len(output_files) == 1
        assert output_files[0].name == "good.jpg"

        saved = cv2.imread(str(output_files[0]))
        assert saved.shape[:2] == (32, 32)

    def test_missing_input_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            clean_dataset(
                input_dir=str(tmp_path / "nonexistent"),
                output_dir=str(tmp_path / "out"),
                size=640,
                blur_threshold=100,
                min_brightness=20,
                max_brightness=235,
            )
