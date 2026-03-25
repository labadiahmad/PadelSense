import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from collections import deque
import json
import os

# ── Constants ──────────────────────────────────────────────────────────────
NUM_KPTS       = 13
IMG_SIZE       = 640
MINI_W         = 200
MINI_H         = 400
MINI_PAD       = 20
DIFF_THRESHOLD = 3
PATCH_MARGIN   = 10
MAX_BOX_SIZE   = 60
TRAIL_LENGTH   = 15

VIDEO_PATH       = "/Users/karam/Documents/padelsense/data/videos/rally.mp4"
OUTPUT_PATH      = "/Users/karam/Documents/padelsense/data/videos/rally_homography.mp4"
BALL_MODEL_PATH  = "/Users/karam/Documents/padelsense/runs/detect/train9/weights/best.pt"
KEYPOINTS_FILE   = "/Users/karam/Documents/padelsense/manual_keypoints.json"

KEYPOINT_NAMES = ["FCL","FSL","NL","BSL","BCL","BCR","BSR","NR","FSR","FCR","NC","FSM","BSM"]


# ── Mini court 2D points ───────────────────────────────────────────────────
def build_mini_court_points():
    w = MINI_W - 2 * MINI_PAD
    p = MINI_PAD
    full_w    = 10.0
    full_h    = 20.0
    net_y     = 10.0
    svc_front = 3.05
    svc_back  = 16.95

    def rx(x): return int(p + (x / full_w) * w)
    def ry(y): return int(p + ((full_h - y) / full_h) * (MINI_H - 2 * MINI_PAD))

    return np.array([
        [rx(0),        ry(0)],
        [rx(0),        ry(svc_front)],
        [rx(0),        ry(net_y)],
        [rx(0),        ry(svc_back)],
        [rx(0),        ry(full_h)],
        [rx(full_w),   ry(full_h)],
        [rx(full_w),   ry(svc_back)],
        [rx(full_w),   ry(net_y)],
        [rx(full_w),   ry(svc_front)],
        [rx(full_w),   ry(0)],
        [rx(full_w/2), ry(net_y)],
        [rx(full_w/2), ry(svc_front)],
        [rx(full_w/2), ry(svc_back)],
    ], dtype=np.float32)


# ── Manual keypoint picker ─────────────────────────────────────────────────
clicked_points = []
current_idx    = [0]

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        idx = current_idx[0]
        if idx < NUM_KPTS:
            clicked_points.append((x, y))
            print(f"  ✓ {idx+1}/{NUM_KPTS} — {KEYPOINT_NAMES[idx]}: ({x}, {y})")
            current_idx[0] += 1

def pick_keypoints(frame):
    global clicked_points
    clicked_points = []
    current_idx[0] = 0

    clone = frame.copy()
    cv2.namedWindow("Pick Keypoints", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Pick Keypoints", mouse_callback)

    print("\nClick the 13 court keypoints IN THIS ORDER:")
    for i, name in enumerate(KEYPOINT_NAMES):
        print(f"  {i+1}. {name}")
    print()

    while True:
        display = clone.copy()
        idx = current_idx[0]

        # Draw already clicked points
        for i, (px, py) in enumerate(clicked_points):
            cv2.circle(display, (px, py), 8, (0, 255, 0), -1)
            cv2.circle(display, (px, py), 8, (0, 0, 0), 2)
            cv2.putText(display, KEYPOINT_NAMES[i], (px+10, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show next point to click
        if idx < NUM_KPTS:
            cv2.putText(display, f"Click: {KEYPOINT_NAMES[idx]} ({idx+1}/{NUM_KPTS})",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        else:
            cv2.putText(display, "All done! Press ENTER to confirm or Z to undo.",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Pick Keypoints", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and idx == NUM_KPTS:  # ENTER
            break
        elif key == ord('z') and len(clicked_points) > 0:  # undo
            clicked_points.pop()
            current_idx[0] -= 1
            print(f"  ↩ Undid last point, re-click {KEYPOINT_NAMES[current_idx[0]]}")

    cv2.destroyAllWindows()
    return clicked_points


# ── Draw mini court ────────────────────────────────────────────────────────
def draw_mini_court(trail):
    panel = np.zeros((MINI_H, MINI_W, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)

    p   = MINI_PAD
    w   = MINI_W - 2 * MINI_PAD
    pts = build_mini_court_points()

    net_y = int(pts[2][1])
    fsl_y = int(pts[1][1])
    bsl_y = int(pts[3][1])
    cx    = MINI_W // 2

    cv2.rectangle(panel, (p, p), (p + w, MINI_H - p), (255, 255, 255), 2)
    cv2.line(panel, (p, net_y), (p + w, net_y), (255, 255, 255), 2)
    cv2.line(panel, (p, fsl_y), (p + w, fsl_y), (180, 180, 180), 1)
    cv2.line(panel, (p, bsl_y), (p + w, bsl_y), (180, 180, 180), 1)
    cv2.line(panel, (cx, fsl_y), (cx, bsl_y), (180, 180, 180), 1)

    # Fading trail
    trail_list = [t for t in trail if t is not None]
    for i, pos in enumerate(trail_list):
        bx, by = int(pos[0]), int(pos[1])
        alpha  = (i + 1) / max(len(trail_list), 1)
        radius = max(2, int(5 * alpha))
        color  = (int(80 * alpha), int(200 * alpha), int(80 * alpha))
        cv2.circle(panel, (bx, by), radius, color, -1)

    # Current ball
    if trail_list:
        bx, by = int(trail_list[-1][0]), int(trail_list[-1][1])
        cv2.circle(panel, (bx, by), 6, (0, 255, 0), -1)
        cv2.circle(panel, (bx, by), 6, (0, 0, 0), 1)

    return panel


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ball_model = YOLO(BALL_MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    ret, first_frame = cap.read()

    # Load or pick keypoints
    if os.path.exists(KEYPOINTS_FILE):
        with open(KEYPOINTS_FILE) as f:
            src_pts = np.array(json.load(f), dtype=np.float32)
        print(f"Loaded saved keypoints from {KEYPOINTS_FILE}")
    else:
        print("No saved keypoints found — opening picker...")
        picked = pick_keypoints(first_frame)
        src_pts = np.array(picked, dtype=np.float32)
        with open(KEYPOINTS_FILE, 'w') as f:
            json.dump(picked, f)
        print(f"Keypoints saved to {KEYPOINTS_FILE}")

    dst_pts = build_mini_court_points()
    H_mat, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print("Homography computed.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_gray = None
    trail     = deque(maxlen=TRAIL_LENGTH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = ball_model.predict(frame, conf=0.3, verbose=False, device="mps")

        ball_found = False

        if prev_gray is not None and results[0].boxes is not None:
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)

                if (x2 - x1) > MAX_BOX_SIZE or (y2 - y1) > MAX_BOX_SIZE:
                    continue

                px1 = max(0, x1 - PATCH_MARGIN)
                py1 = max(0, y1 - PATCH_MARGIN)
                px2 = min(W, x2 + PATCH_MARGIN)
                py2 = min(H, y2 + PATCH_MARGIN)

                curr_patch = gray[py1:py2, px1:px2]
                prev_patch = prev_gray[py1:py2, px1:px2]

                if curr_patch.shape != prev_patch.shape or curr_patch.size == 0:
                    continue

                if np.mean(cv2.absdiff(curr_patch, prev_patch)) > DIFF_THRESHOLD:
                    cx_ball = (x1 + x2) / 2
                    cy_ball = y2 + 80

                    # If ball is too high in the image it's airborne — skip projection
#                    AIRBORNE_THRESHOLD = int(H * 0.30)  # top 35% of frame = airborne
#                    if cy_ball < AIRBORNE_THRESHOLD:
#                        if not ball_found:
#                            trail.append(None)
#                        continue

                    # Project to mini court
                    ball_pt = np.array([[[cx_ball, cy_ball]]], dtype=np.float32)
                    mini_pt = cv2.perspectiveTransform(ball_pt, H_mat)[0][0]

                    # Clamp to mini court bounds so ball never disappears
                    mx = float(np.clip(mini_pt[0], MINI_PAD, MINI_W - MINI_PAD))
                    my = float(np.clip(mini_pt[1], MINI_PAD, MINI_H - MINI_PAD))
                    trail.append((mx, my))
                    ball_found = True

                    # Draw on main frame
                    cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (0,0,0), 4)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    corner = 8
                    for (ccx, ccy, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                        cv2.line(frame, (ccx,ccy), (ccx+dx*corner,ccy), (0,255,0), 3)
                        cv2.line(frame, (ccx,ccy), (ccx,ccy+dy*corner), (0,255,0), 3)
                    lbl = "ball"
                    (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (x1, y1-lh-8), (x1+lw+6, y1), (0,255,0), -1)
                    cv2.putText(frame, lbl, (x1+3, y1-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
                    break

        if not ball_found:
            trail.append(None)

        # Mini court overlay
        mini  = draw_mini_court(trail)
        y_off = H - MINI_H - 20
        x_off = 20
        roi   = frame[y_off:y_off+MINI_H, x_off:x_off+MINI_W]
        frame[y_off:y_off+MINI_H, x_off:x_off+MINI_W] = cv2.addWeighted(roi, 0.3, mini, 0.7, 0)

        prev_gray = gray
        out.write(frame)

    cap.release()
    out.release()
    print(f"Done — saved to {OUTPUT_PATH}")
