from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class VideoLoadError(Exception):
    pass


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float
    codec: Optional[str] = None


def _fourcc_to_str(fourcc: int) -> str:
    return "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip()


def validate_video_path(video_path: str | Path) -> Path:
    path = Path(video_path)

    if not str(path).strip():
        raise ValueError("Video path must not be empty.")

    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file path, got: {path}")

    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension '{path.suffix}'. "
            f"Supported formats: {sorted(SUPPORTED_VIDEO_EXTENSIONS)}"
        )

    return path


def load_video(video_path: str | Path) -> cv2.VideoCapture:
    path = validate_video_path(video_path)

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoLoadError(f"Failed to open video: {path}")

    return capture


def get_video_metadata(capture: cv2.VideoCapture, video_path: str | Path) -> VideoMetadata:
    if capture is None or not capture.isOpened():
        raise VideoLoadError("Video capture is not opened.")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec = _fourcc_to_str(fourcc)
    duration_seconds = frame_count / fps if fps > 0 else 0.0

    return VideoMetadata(
        path=str(video_path),
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_seconds=duration_seconds,
        codec=codec,
    )


def release_video(capture: cv2.VideoCapture) -> None:
    if capture is not None and capture.isOpened():
        capture.release()