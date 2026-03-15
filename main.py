from input_handler.video_loader import load_video
from ball_detector.detector import detect_ball
from court_detection.keypoint_detector import detect_court_keypoints
from mini_court.mini_court_mapper import map_to_mini_court
from event_detector.event_classifier import classify_events
from analysis.trajectory_analysis import analyze_trajectory
from rule_engine.rule_engine import apply_rules
from output_module.output_writer import write_output


def main():
    print("PadelSense pipeline started")


if __name__ == "__main__":
    main()
