from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class FramePacket:
    frame_index: int
    timestamp_seconds: float
    original_frame: np.ndarray


def seek_to_frame(capture: cv2.VideoCapture, frame_number: int) -> None:
    if capture is None or not capture.isOpened():
        raise ValueError("Video capture is not opened.")

    if frame_number < 0:
        raise ValueError("frame_number must be >= 0")

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)


def extract_frames(
    capture: cv2.VideoCapture,
    frame_skip: int = 1,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    max_output_frames: Optional[int] = None,
) -> Generator[FramePacket, None, None]:
    if capture is None or not capture.isOpened():
        raise ValueError("Video capture is not opened.")

    if frame_skip < 1:
        raise ValueError("frame_skip must be >= 1")

    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")

    seek_to_frame(capture, start_frame)

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    current_frame = start_frame
    yielded = 0

    while True:
        if end_frame is not None and current_frame > end_frame:
            break

        ok, frame = capture.read()
        if not ok:
            break

        if (current_frame - start_frame) % frame_skip == 0:
            timestamp_seconds = current_frame / fps if fps > 0 else 0.0

            yield FramePacket(
                frame_index=current_frame,
                timestamp_seconds=timestamp_seconds,
                original_frame=frame.copy(),
            )

            yielded += 1
            if max_output_frames is not None and yielded >= max_output_frames:
                break

        current_frame += 1