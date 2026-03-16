from input_handler import (
    DisplayConfig,
    InputHandlerRuntime,
    PreprocessingConfig,
)


def choose_video_file():
    print("\nDrag and drop the video file here, then press Enter.")
    print("Or paste the full video path manually.\n")
    return input("Video path: ").strip().strip('"').strip("'")


def backend_model_callback(packet, processed_frame):
    # future AI pipeline goes here
    _ = packet
    _ = processed_frame


def main():
    video_path = choose_video_file()

    if not video_path:
        print("No video selected. Exiting.")
        return

    runtime = InputHandlerRuntime(
        video_path=video_path,
        model_preprocess_config=PreprocessingConfig(
            target_size=(960, 540),
            denoise=True,
            sharpen=True,
            clahe=True,
            gaussian_blur=False,
            gamma_correction=False,
            gamma=1.0,
            convert_bgr_to_rgb=False,
            normalize=False,
        ),
        display_config=DisplayConfig(
            window_name="PadelSense User Preview",
            backend_window_name="PadelSense Backend Preview",
            display_width=1280,
            display_height=720,
            backend_width=960,
            backend_height=540,
            overlay_color=(0, 255, 0),
            overlay_font_scale=0.7,
            overlay_thickness=2,
            show_backend_preview=False,
        ),
        model_callback=backend_model_callback,
    )

    runtime.run()


if __name__ == "__main__":
    main()