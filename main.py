from video_processor import VideoProcessor

def main():
    # Video path
    video_path = 'Input/Calibration_test2.mov'

    # Calibration file path
    calibration_file = 'Files/my_camera_calibration.json'

    # Mask file path
    road_mask_file = 'Files/my_road_mask.npy'

    # Initialize the video processor
    processor = VideoProcessor(video_path, calibration_file, road_mask_file)

    # Process the video
    processor.process_video()

if __name__ == "__main__":
    main()