"""
Data Capture Script – Step 1 of the Beech-Wood Vision System pipeline.

Connects to a camera (industrial or webcam), shows a live preview and saves
high-resolution frames to a configurable output folder.

Trigger modes
-------------
space   – press the SPACE key in the preview window
motion  – automatic frame capture whenever significant motion is detected

Usage
-----
    python src/capture/data_capture.py --output data/raw/healthy --camera 0
    python src/capture/data_capture.py --output data/raw/defective --trigger motion
"""

import argparse
import datetime
import os
import sys

import cv2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live camera preview with frame capture for beech-wood inspection."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("data", "raw"),
        help="Directory where captured frames are saved (default: data/raw).",
    )
    parser.add_argument(
        "--trigger",
        choices=["space", "motion"],
        default="space",
        help="Capture trigger: 'space' key or 'motion' detection (default: space).",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.5,
        help=(
            "Percentage of changed pixels (0-100) required to trigger a motion capture "
            "(default: 0.5)."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Requested capture width in pixels (default: 1920).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Requested capture height in pixels (default: 1080).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame",
        help="Filename prefix for saved images (default: frame).",
    )
    return parser


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _save_frame(frame: "cv2.Mat", output_dir: str, prefix: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{prefix}_{_timestamp()}.jpg"
    path = os.path.join(output_dir, filename)
    cv2.imwrite(path, frame)
    return path


def _motion_detected(prev_gray: "cv2.Mat", curr_gray: "cv2.Mat", threshold: float) -> bool:
    """Return True when the fraction of changed pixels exceeds *threshold* percent."""
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    changed = cv2.countNonZero(mask)
    total = mask.shape[0] * mask.shape[1]
    return (changed / total) * 100.0 >= threshold


def run_capture(
    camera: int = 0,
    output: str = os.path.join("data", "raw"),
    trigger: str = "space",
    motion_threshold: float = 0.5,
    width: int = 1920,
    height: int = 1080,
    prefix: str = "frame",
) -> None:
    """Main capture loop.

    Parameters
    ----------
    camera:           Camera device index.
    output:           Directory where captured frames are saved.
    trigger:          ``'space'`` or ``'motion'``.
    motion_threshold: Percentage of changed pixels that triggers a motion capture.
    width/height:     Requested resolution.
    prefix:           Filename prefix.
    """
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera}.", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print(
        f"Camera opened  : device {camera}\n"
        f"Resolution     : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}\n"
        f"Output folder  : {output}\n"
        f"Trigger        : {trigger}\n"
    )
    if trigger == "space":
        print("Press SPACE to capture a frame. Press Q or ESC to quit.")
    else:
        print(f"Motion threshold: {motion_threshold}% changed pixels. Press Q or ESC to quit.")

    os.makedirs(output, exist_ok=True)
    prev_gray = None
    captured_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame from camera.", file=sys.stderr)
            break

        display = frame.copy()
        cv2.putText(
            display,
            f"Captured: {captured_count}  |  [{trigger.upper()}]  |  Q=quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Beech-Wood Vision – Live Preview", display)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):  # Q or ESC
            break

        if trigger == "space":
            if key == ord(" "):
                path = _save_frame(frame, output, prefix)
                captured_count += 1
                print(f"Saved [{captured_count}]: {path}")
        else:  # motion
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.GaussianBlur(curr_gray, (21, 21), 0)
            if prev_gray is not None and _motion_detected(prev_gray, curr_gray, motion_threshold):
                path = _save_frame(frame, output, prefix)
                captured_count += 1
                print(f"Motion captured [{captured_count}]: {path}")
            prev_gray = curr_gray

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended. Total frames saved: {captured_count}")


def main() -> None:
    args = build_parser().parse_args()
    run_capture(
        camera=args.camera,
        output=args.output,
        trigger=args.trigger,
        motion_threshold=args.motion_threshold,
        width=args.width,
        height=args.height,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
