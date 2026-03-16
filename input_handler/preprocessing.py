from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessingConfig:
    target_size: Optional[Tuple[int, int]] = (960, 540)  # (width, height)
    denoise: bool = True
    sharpen: bool = True
    clahe: bool = True
    gaussian_blur: bool = False
    gamma_correction: bool = False
    gamma: float = 1.0
    convert_bgr_to_rgb: bool = False
    normalize: bool = False


def resize_frame(frame: np.ndarray, target_size: Optional[Tuple[int, int]]) -> np.ndarray:
    if frame is None:
        raise ValueError("Frame is None.")
    if target_size is None:
        return frame

    width, height = target_size
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def apply_denoise(frame: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)


def apply_gaussian_blur(frame: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(frame, (3, 3), 0)


def apply_sharpen(frame: np.ndarray) -> np.ndarray:
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    return cv2.filter2D(frame, -1, kernel)


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_gamma_correction(frame: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("Gamma must be greater than 0.")

    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
    ]).astype("uint8")
    return cv2.LUT(frame, table)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    return frame.astype(np.float32) / 255.0


def preprocess_frame(frame: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    if frame is None:
        raise ValueError("Frame is None.")

    output = resize_frame(frame, config.target_size)

    if config.denoise:
        output = apply_denoise(output)

    if config.gaussian_blur:
        output = apply_gaussian_blur(output)

    if config.clahe:
        output = apply_clahe(output)

    if config.sharpen:
        output = apply_sharpen(output)

    if config.gamma_correction:
        output = apply_gamma_correction(output, config.gamma)

    if config.convert_bgr_to_rgb:
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    if config.normalize:
        output = normalize_frame(output)

    return output