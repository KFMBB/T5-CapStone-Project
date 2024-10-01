import cv2
import numpy as np
import os
import json
import logging
from camera_calibration import CameraCalibration
from car_detection import CarDetection
from bounding_box_constructor import BoundingBoxConstructor
from vehicle_tracker import VehicleTracker
from speed_calculator import SpeedCalculator
from depth_estimation import DepthEstimationModel
from masker import Masker

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoProcessor:
    def __init__(self, video_path, calibration_file='Files/my_camera_calibration.json', road_mask_file='Files/my_road_mask.npy',
                 detection_confidence=0.4, frame_skip=5):
        self.video_path = video_path
        self.calibration_file = calibration_file
        self.road_mask_file = road_mask_file
        self.detection_confidence = detection_confidence
        self.frame_skip = frame_skip

        self.calibration = CameraCalibration()
        self.car_detection = CarDetection()
        self.depth_model = DepthEstimationModel()
        self.tracker = VehicleTracker(max_frames_to_skip=10, min_hits=3, max_track_length=30)
        self.speed_calculator = SpeedCalculator(smoothing_window=5, speed_confidence_threshold=0.8, max_history=100)

        self.cap = cv2.VideoCapture(video_path)
        self.ret, self.frame = self.cap.read()
        if not self.ret:
            raise ValueError("Failed to read the video file.")
        self.height, self.width = self.frame.shape[:2]

        self.masker = Masker(self.height, self.width)
        self.ipm_matrix = None
        self.bbox_constructor = None

        self.output_path = 'Output/output_video.mp4'
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.out = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))

        self.results = []
        self.json_file_path = 'Output/speed_estimation_results.json'
        self.last_json_update = 0

    def process_video(self):
        # Road mask selection or loading
        if os.path.exists(self.road_mask_file):
            self.masker.load_road_mask(self.road_mask_file)
            logging.info(f"Loaded existing road mask from {self.road_mask_file}")
        else:
            logging.info("Please select the road area...")
            self.masker.manual_road_selection(self.frame)
            self.masker.save_road_mask(self.road_mask_file)
            logging.info(f"Saved road mask to {self.road_mask_file}")

        # Camera calibration
        if os.path.exists(self.calibration_file):
            self.calibration.load_calibration(self.calibration_file)
            logging.info(f"Loaded existing camera calibration from {self.calibration_file}")
        else:
            logging.info("Performing camera calibration...")
            calibration_frames = self._collect_calibration_frames()
            self.calibration.calibrate_camera(calibration_frames)
            self.calibration.save_calibration(self.calibration_file)
            logging.info(f"Saved camera calibration to {self.calibration_file}")

        # Reset video capture to the beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Get necessary matrices
        self.ipm_matrix = self.calibration.ipm_matrix
        camera_matrix = self.calibration.get_camera_matrix()
        vanishing_points = self.calibration.vanishing_points

        # Initialize BoundingBoxConstructor with calibration results
        self.bbox_constructor = BoundingBoxConstructor(vanishing_points, camera_matrix)

        frame_count = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cv2.namedWindow('Processed Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Processed Frame', 1280, 720)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % self.frame_skip != 0:
                continue

            current_time = frame_count / self.fps

            vis_frame = frame.copy()
            vis_ipm_frame = self.calibration.apply_ipm(frame.copy())

            masked_frame = self.masker.apply_mask(frame)
            ipm_frame = self.calibration.apply_ipm(masked_frame)

            depth_map = self.depth_model.estimate_depth(ipm_frame)

            # Debug: Print and visualize depth map statistics
            depth_min, depth_max, depth_mean, depth_std = depth_map.min(), depth_map.max(), depth_map.mean(), depth_map.std()
            logging.debug(f"Frame {frame_count}: Depth Map - min: {depth_min}, max: {depth_max}, mean: {depth_mean}, std: {depth_std}")

            # Visualize depth map
            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            detections = self.car_detection.detect_cars(ipm_frame, self.ipm_matrix, self.detection_confidence)

            # Debug: Print number of detections
            logging.debug(f"Frame {frame_count}: Number of detections: {len(detections)}")

            # Draw bounding boxes on the IPM frame
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                cv2.rectangle(vis_ipm_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Combine all four views into one image
            combined_frame = np.vstack((np.hstack((vis_frame, vis_ipm_frame)),
                                         np.hstack((depth_vis, np.zeros_like(depth_vis)))))

            cv2.imshow('Processed Frame', combined_frame)

            # Write to video output
            self.out.write(combined_frame)

            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

    def _collect_calibration_frames(self):
        # Placeholder for calibration frame collection
        return []

if __name__ == "__main__":
    video_path = 'path_to_your_video.mp4'  # Replace with your video file path
    processor = VideoProcessor(video_path)
    processor.process_video()
