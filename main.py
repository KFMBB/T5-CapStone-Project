from video_processor import VideoProcessor

def main():
    # Set your video path here
    video_path = 'Input/Calibration_test2.mov'

    # Initialize the video processor
    processor = VideoProcessor(video_path)
    
    # Process the video
    processor.process_video()

if __name__ == "__main__":
    main()
