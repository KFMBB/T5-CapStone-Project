from video_processor import VideoProcessor

def main():
    # Set your video path here
    video_path = 'Output/output_video.mp4'

    # Initialize the video processor
    processor = VideoProcessor(video_path)
    
    # Process the video
    processor.process_video()

if __name__ == "__main__":
    main()
