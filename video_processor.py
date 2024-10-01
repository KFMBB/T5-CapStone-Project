import cv2
import numpy as np
import os
from camera_calibration import CameraCalibration
from car_detection import CarDetection
from bounding_box_constructor import BoundingBoxConstructor
from vehicle_tracker import VehicleTracker
from speed_calculator import SpeedCalculator
from depth_estimation import DepthEstimationModel
from masker import Masker

class VideoProcessor:
    def __init__(self, video_path, calibration_file='Output/my_camera_calibration.json', 
                 road_mask_file='Output/my_road_mask.npy', detection_confidence=0.4, frame_skip=5):
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
        self.output_path = 'Output/output_video.mp4'
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.out = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))

        if not os.path.exists('Output'):
            os.makedirs('Output')

    def process_video(self):
        if os.path.exists(self.road_mask_file):
            self.masker.load_road_mask(self.road_mask_file)
        else:
            self.masker.manual_road_selection(self.frame)
            self.masker.save_road_mask(self.road_mask_file)

        if os.path.exists(self.calibration_file):
            self.calibration.load_calibration(self.calibration_file)
        else:
            calibration_frames = self._collect_calibration_frames()
            self.calibration.calibrate_camera(calibration_frames)
            self.calibration.save_calibration(self.calibration_file)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.ipm_matrix = self.calibration.ipm_matrix
        camera_matrix = self.calibration.get_camera_matrix()
        vanishing_points = self.calibration.vanishing_points
        self.bbox_constructor = BoundingBoxConstructor(vanishing_points, camera_matrix)

        frame_count = 0
        cv2.namedWindow('Processed Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Processed Frame', 1280, 720)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % self.frame_skip != 0:
                continue

            vis_frame = frame.copy()
            vis_ipm_frame = self.calibration.apply_ipm(frame.copy())
            masked_frame = self.masker.apply_mask(frame)
            ipm_frame = self.calibration.apply_ipm(masked_frame)
            depth_map = self.depth_model.estimate_depth(ipm_frame)

            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            detections = self.car_detection.detect_cars(ipm_frame, self.ipm_matrix, self.detection_confidence)

            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                cv2.rectangle(vis_ipm_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            combined_frame = np.vstack((np.hstack((vis_frame, vis_ipm_frame)), np.hstack((masked_frame, depth_vis))))
            self.out.write(combined_frame)
            cv2.imshow('Processed Frame', combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

    def _collect_calibration_frames(self):
        # Implement frame collection for camera calibration here
        return []

