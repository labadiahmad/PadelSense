from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2

from input_handler.frame_extractor import FramePacket, extract_frames
from input_handler.preprocessing import PreprocessingConfig, preprocess_frame
from input_handler.video_loader import (
    VideoMetadata,
    get_video_metadata,
    load_video,
    release_video,
)


@dataclass(frozen=True)
class DisplayConfig:
    window_name: str = "PadelSense Input Handler Preview"
    backend_window_name: str = "PadelSense Backend Preview"
    display_width: int = 1280
    display_height: int = 720
    backend_width: int = 960
    backend_height: int = 540
    overlay_color: tuple = (0, 255, 0)
    overlay_font_scale: float = 0.7
    overlay_thickness: int = 2
    show_backend_preview: bool = True


class InputHandlerRuntime:
    def __init__(
        self,
        video_path: str,
        model_preprocess_config: PreprocessingConfig,
        display_config: DisplayConfig = DisplayConfig(),
        model_callback: Optional[Callable[[FramePacket, object], None]] = None,
    ):
        self.video_path = video_path
        self.model_preprocess_config = model_preprocess_config
        self.display_config = display_config
        self.model_callback = model_callback

        self.capture = load_video(video_path)
        self.metadata: VideoMetadata = get_video_metadata(self.capture, video_path)

        self.stop_event = threading.Event()

        self.latest_packet_lock = threading.Lock()
        self.latest_packet: Optional[FramePacket] = None

        self.latest_processed_lock = threading.Lock()
        self.latest_processed_frame = None
        self.latest_processed_index: Optional[int] = None

        self.paused = False
        self.displayed_frames = 0
        self.start_wall_time = time.perf_counter()
        self.playback_anchor: Optional[float] = None

    def _draw_overlay(self, frame, packet, playback_fps):
        output = frame.copy()

        progress = 0.0
        if self.metadata.frame_count > 0:
            progress = (packet.frame_index / self.metadata.frame_count) * 100

        lines = [
            f"Frame: {packet.frame_index}/{self.metadata.frame_count}",
            f"Time: {packet.timestamp_seconds:.2f}s / {self.metadata.duration_seconds:.2f}s",
            f"Video FPS: {self.metadata.fps:.2f}",
            f"Playback FPS: {playback_fps:.2f}",
            f"Progress: {progress:.1f}%",
            "ESC = quit | SPACE = pause",
        ]

        h, w = output.shape[:2]
        y = 30

        for line in lines:
            text_size = cv2.getTextSize(
                line,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.display_config.overlay_font_scale,
                self.display_config.overlay_thickness,
            )[0]
            x = w - text_size[0] - 20

            cv2.putText(
                output,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.display_config.overlay_font_scale,
                self.display_config.overlay_color,
                self.display_config.overlay_thickness,
            )
            y += 30

        return output

    def _draw_backend_overlay(self, frame, frame_index: Optional[int]):
        output = frame.copy()

        lines = [
            "MODEL INPUT (DEBUG)",
            f"Processed Frame: {frame_index if frame_index is not None else '-'}",
        ]

        h, w = output.shape[:2]
        y = 30

        for line in lines:
            text_size = cv2.getTextSize(
                line,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.display_config.overlay_font_scale,
                self.display_config.overlay_thickness,
            )[0]
            x = w - text_size[0] - 20

            cv2.putText(
                output,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.display_config.overlay_font_scale,
                (0, 255, 0),
                self.display_config.overlay_thickness,
            )
            y += 30

        return output

    def _backend_worker(self):
        while not self.stop_event.is_set():
            packet_to_process = None

            with self.latest_packet_lock:
                if self.latest_packet is not None:
                    packet_to_process = self.latest_packet
                    self.latest_packet = None

            if packet_to_process is not None:
                processed_frame = preprocess_frame(
                    packet_to_process.original_frame,
                    self.model_preprocess_config,
                )

                with self.latest_processed_lock:
                    self.latest_processed_frame = processed_frame.copy()
                    self.latest_processed_index = packet_to_process.frame_index

                if self.model_callback is not None:
                    self.model_callback(packet_to_process, processed_frame)
            else:
                time.sleep(0.001)

    def run(self):
        print("\nSelected video:")
        print(self.video_path)

        print("\nVideo metadata:")
        print(self.metadata)

        print("\nPreview started.")
        print("Window 1 = original video for user")
        print("Window 2 = backend preprocessed frame")
        print("Press ESC to quit.")
        print("Press SPACE to pause/resume.\n")

        worker = threading.Thread(target=self._backend_worker, daemon=True)
        worker.start()

        cv2.namedWindow(self.display_config.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.display_config.window_name,
            self.display_config.display_width,
            self.display_config.display_height,
        )

        if self.display_config.show_backend_preview:
            cv2.namedWindow(self.display_config.backend_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                self.display_config.backend_window_name,
                self.display_config.backend_width,
                self.display_config.backend_height,
            )

        try:
            for packet in extract_frames(self.capture, frame_skip=1):
                if self.playback_anchor is None:
                    self.playback_anchor = time.perf_counter() - packet.timestamp_seconds

                target_time = self.playback_anchor + packet.timestamp_seconds
                now = time.perf_counter()

                if target_time > now:
                    time.sleep(target_time - now)

                with self.latest_packet_lock:
                    self.latest_packet = packet

                elapsed = max(time.perf_counter() - self.start_wall_time, 1e-6)
                playback_fps = self.displayed_frames / elapsed if self.displayed_frames > 0 else 0.0

                # USER WINDOW
                frame_to_show = self._draw_overlay(packet.original_frame, packet, playback_fps)
                cv2.imshow(self.display_config.window_name, frame_to_show)

                # BACKEND WINDOW
                if self.display_config.show_backend_preview:
                    with self.latest_processed_lock:
                        backend_frame = None if self.latest_processed_frame is None else self.latest_processed_frame.copy()
                        backend_index = self.latest_processed_index

                    if backend_frame is not None:
                        backend_frame = self._draw_backend_overlay(backend_frame, backend_index)
                        cv2.imshow(self.display_config.backend_window_name, backend_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC
                    break

                if key == ord(" "):
                    self.paused = not self.paused

                    while self.paused:
                        key = cv2.waitKey(30) & 0xFF

                        if key == 27:
                            self.paused = False
                            self.stop_event.set()
                            return

                        if key == ord(" "):
                            self.paused = False
                            self.playback_anchor = time.perf_counter() - packet.timestamp_seconds

                self.displayed_frames += 1

        finally:
            self.stop_event.set()
            release_video(self.capture)
            cv2.destroyAllWindows()
            print("\nDone.")