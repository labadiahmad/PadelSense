from input_handler import (
    DisplayConfig,
    InputHandlerRuntime,
    PreprocessingConfig,
)


import os

def choose_video_file():
    input_dir = "input_videos"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    videos = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not videos:
        print(f"\nNo videos found in '{input_dir}/'. Please place a video there.")
        print("Alternatively, paste the full video path manually.\n")
        path = input("Video path: ").strip().strip('"').strip("'")
        return path

    print("\nAvailable videos in 'input_videos/':")
    for i, vid in enumerate(videos):
        print(f"[{i+1}] {vid}")

    print(f"\nSelect a video [1-{len(videos)}] or paste a full path:")
    choice = input("Choice/Path: ").strip().strip('"').strip("'")

    if choice.isdigit() and 1 <= int(choice) <= len(videos):
        return os.path.join(input_dir, videos[int(choice)-1])
    else:
        return choice


import cv2
import pandas as pd
from tqdm import tqdm
from court_detection.manual_selector import ManualCourtSelector
from ball_detector.detector import BallTracker, BallTrackerConfig

def main():
    video_path = choose_video_file()

    if not video_path:
        print("No video selected. Exiting.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the first frame.")
        return

    selector = ManualCourtSelector(first_frame)
    keypoints = selector.select_keypoints()

    if len(keypoints) != 12:
        print("12 keypoints were not selected. Proceeding without keypoints.")

    # Reload video to read all frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Reading video frames...")
    frames = []
    for _ in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    print("Initializing BallTracker...")
    config = BallTrackerConfig()
    tracker = BallTracker(config, "yolov8n.pt") # Ensure yolov8n.pt or valid model

    print("Running detection...")
    raw_detections = tracker.detect_frames(frames, read_from_stub=False)

    print("Interpolating ball positions...")
    interpolated_positions = tracker.interpolate_ball_positions(raw_detections)

    print("Mapping to mini court...")
    from mini_court.mini_court_mapper import map_to_mini_court
    from mini_court.draw_mini_court import draw_mini_court

    # We map all positions to the mini court
    mini_court_positions = map_to_mini_court(interpolated_positions, keypoints)

    print("Rendering output video...")

    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    output_path = os.path.join("outputs", "output_video.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    csv_data = []

    for i in tqdm(range(len(frames))):
        frame = frames[i].copy()

        # Draw Keypoints
        for i_kp, kp in enumerate(keypoints):
            cv2.circle(frame, kp, 5, (0, 0, 255), -1)

        # Draw BBox
        pos = interpolated_positions[i]
        conf = 1.0
        x_c, y_c = None, None

        if 1 in pos:
            bbox = pos[1]
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, "Ball ID: 1", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            x_c = (x1 + x2) / 2.0
            y_c = (y1 + y2) / 2.0

            # Check if this was an original detection or interpolated by checking raw_detections
            if 1 in raw_detections[i]:
                conf = 1.0
            else:
                conf = 0.5

        csv_data.append({'Frame': i, 'X': x_c, 'Y': y_c, 'Confidence': conf if x_c is not None else None})

        # Draw Mini Court
        if len(keypoints) >= 12 and mini_court_positions: # basic check to ensure mapping exists
            # We assume draw_mini_court takes the frame and mapped positions or trail.
            # Adapting slightly for the interface in draw_mini_court.py
            # The exact interface of draw_mini_court isn't fully robust in this stub,
            # but we follow the signature `draw_mini_court(frame, mapped_positions)` found in the original stub.
            frame = draw_mini_court(frame, mini_court_positions)

        out.write(frame)

    out.release()

    print("Saving ball coordinates to CSV...")
    df = pd.DataFrame(csv_data)
    df.to_csv(os.path.join("outputs", "ball_coordinates.csv"), index=False)

    print("Processing complete!")


if __name__ == "__main__":
    main()