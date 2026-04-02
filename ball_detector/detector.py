import os
import cv2
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from ultralytics import YOLO

@dataclass
class BallTrackerConfig:
    smoothing_alpha: float = 0.5
    max_missed_frames: int = 5
    history_size: int = 10
    max_prediction_step: int = 5
    missed_prediction_boost: float = 1.2
    minimum_change_frames_for_hit: int = 25

class BallTracker:
    def __init__(self, config: BallTrackerConfig, model_path: str):
        self.config = config
        self.model = YOLO(model_path)

    def detect_frames(self, frames: List[np.ndarray], read_from_stub: bool = False, stub_path: Optional[str] = None) -> List[Dict[int, List[float]]]:
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        ball_detections = []
        prev_center = None
        missed_frames = 0

        for frame in frames:
            results = self.model(frame)[0]
            boxes = results.boxes

            best_score = -float('inf')
            best_box = None
            best_center = None

            # Predict max distance for this frame based on missed frames
            max_dist = self.config.max_prediction_step * (self.config.missed_prediction_boost ** missed_frames)

            for box in boxes:
                coords = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                center_x = (coords[0] + coords[2]) / 2.0
                center_y = (coords[1] + coords[3]) / 2.0

                if prev_center is not None:
                    dist = np.sqrt((center_x - prev_center[0])**2 + (center_y - prev_center[1])**2)
                    # Normalize distance score (e.g. max score 1 when dist=0)
                    dist_score = max(0, 1 - (dist / max(1, max_dist)))

                    # Weighted scoring: Distance 0.78 vs Confidence 0.22
                    score = (dist_score * 0.78) + (conf * 0.22)
                else:
                    score = conf

                if score > best_score:
                    best_score = score
                    best_box = coords
                    best_center = (center_x, center_y)

            # Match condition
            if best_box is not None and (prev_center is None or best_score > 0):
                ball_detections.append({1: best_box})
                prev_center = best_center
                missed_frames = 0
            else:
                ball_detections.append({})
                missed_frames += 1
                if missed_frames > self.config.max_missed_frames:
                    prev_center = None

        if stub_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(stub_path)), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def interpolate_ball_positions(self, ball_positions: List[Dict[int, List[float]]]) -> List[Dict[int, List[float]]]:
        data = []
        for frame_idx, frame_data in enumerate(ball_positions):
            if 1 in frame_data:
                bbox = frame_data[1]
                data.append({'frame': frame_idx, 'x1': bbox[0], 'y1': bbox[1], 'x2': bbox[2], 'y2': bbox[3]})
            else:
                data.append({'frame': frame_idx, 'x1': None, 'y1': None, 'x2': None, 'y2': None})

        df = pd.DataFrame(data)

        # Ensure coordinates are available for frames YOLO missed
        df = df.interpolate(method='linear')
        df = df.bfill()

        interpolated_positions = []
        for idx, row in df.iterrows():
            if pd.isna(row['x1']):
                interpolated_positions.append({})
            else:
                interpolated_positions.append({1: [row['x1'], row['y1'], row['x2'], row['y2']]})

        return interpolated_positions

    def get_ball_shot_frames(self, ball_positions: List[Dict[int, List[float]]]) -> List[int]:
        data = []
        for frame_idx, frame_data in enumerate(ball_positions):
            if 1 in frame_data:
                bbox = frame_data[1]
                mid_y = (bbox[1] + bbox[3]) / 2.0
                data.append({'frame': frame_idx, 'mid_y': mid_y})
            else:
                data.append({'frame': frame_idx, 'mid_y': None})

        df = pd.DataFrame(data)

        # Calculate rolling mean window=5
        df['mid_y_rolling_mean'] = df['mid_y'].rolling(window=5, min_periods=1, center=True).mean()

        # Analyze delta_y
        df['delta_y'] = df['mid_y_rolling_mean'].diff()

        hits = []
        direction = 0 # 1 for down, -1 for up
        sustained_frames = 0

        for i in range(1, len(df)):
            delta = df['delta_y'].iloc[i]
            if pd.isna(delta) or delta == 0:
                # If delta is zero or NaN, maintain previous direction but don't count towards sustained if we want strict
                # Let's just continue
                sustained_frames += 1
                continue

            current_direction = 1 if delta > 0 else -1

            if current_direction != direction:
                if sustained_frames >= self.config.minimum_change_frames_for_hit:
                    # Valid hit occurred
                    # Inversion frame is i - sustained_frames
                    hit_frame = df['frame'].iloc[i - sustained_frames]
                    hits.append(int(hit_frame))

                sustained_frames = 1
                direction = current_direction
            else:
                sustained_frames += 1

        # Check last sequence
        if sustained_frames >= self.config.minimum_change_frames_for_hit:
            hit_frame = df['frame'].iloc[len(df) - sustained_frames]
            hits.append(int(hit_frame))

        return hits

    def draw_bboxes(self, video_frames: List[np.ndarray], ball_positions: List[Dict[int, List[float]]]) -> List[np.ndarray]:
        output_frames = []
        for frame, positions in zip(video_frames, ball_positions):
            out_frame = frame.copy()
            for ball_id, bbox in positions.items():
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(out_frame, f"Ball ID: {ball_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            output_frames.append(out_frame)
        return output_frames

# Stub for compatibility if needed
def detect_ball(frame):
    print("Detecting ball...")
    return []
