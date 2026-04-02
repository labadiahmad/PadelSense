import cv2
import numpy as np

class ManualCourtSelector:
    def __init__(self, frame):
        self.frame = frame.copy()
        self.original_frame = frame.copy()
        self.keypoints = []
        self.window_name = "Select 12 Court Keypoints"

    def select_keypoints(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print("Please click on 12 keypoints of the court.")
        print("Press 'r' to reset, 'c' to confirm/finish early, or 'q' to quit.")

        while True:
            display_frame = self.frame.copy()
            for i, kp in enumerate(self.keypoints):
                cv2.circle(display_frame, kp, 5, (0, 0, 255), -1)
                cv2.putText(display_frame, str(i+1), (kp[0]+10, kp[1]+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                self.keypoints = []
                self.frame = self.original_frame.copy()
            elif key == ord('c') or len(self.keypoints) == 12:
                if len(self.keypoints) == 12:
                    print("12 keypoints selected.")
                break

        cv2.destroyWindow(self.window_name)
        return self.keypoints

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.keypoints) < 12:
                self.keypoints.append((x, y))
