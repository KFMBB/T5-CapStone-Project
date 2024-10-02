import cv2
import numpy as np
import os
import json
from camera_calibration import CameraCalibration
from car_detection import CarDetection
from bounding_box_constructor import BoundingBoxConstructor
from vehicle_tracker import VehicleTracker
from speed_calculator import SpeedCalculator
from depth_estimation import DepthEstimationModel
from masker import Masker
from speed_estimator import SpeedEstimator


class VideoProcessor:
    def __init__(self, video_path, calibration_file='camera_calibration.json', road_mask_file='road_mask.npy',
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
        self.speed_calculator = SpeedCalculator(smoothing_window=5, max_history=100)

        self.cap = cv2.VideoCapture(video_path)
        self.ret, self.frame = self.cap.read()
        if not self.ret:
            raise ValueError("Failed to read the video file.")
        self.height, self.width = self.frame.shape[:2]
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.masker = Masker(self.height, self.width)
        self.ipm_matrix = None
        self.bbox_constructor = None

        self.output_path = 'Output/output_video.mp4'
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))

        self.results = []
        self.json_file_path = 'Output/speed_estimation_results.json'
        self.last_json_update = 0

        self.meters_per_pixel = 0.1  # This is an example value, adjust based on your setup
        self.speed_estimator = None  # We'll initialize this after camera calibration

        print("VideoProcessor initialized successfully.")

    def select_speed_lines(self, frame):
        lines = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(lines) < 2:
                    lines.append(y)
                    cv2.line(frame, (0, y), (frame.shape[1], y), (0, 255, 0), 2)
                    cv2.imshow('Select Speed Lines', frame)

                if len(lines) == 2:
                    cv2.setMouseCallback('Select Speed Lines', lambda *args: None)

        clone = frame.copy()
        cv2.namedWindow('Select Speed Lines')
        cv2.setMouseCallback('Select Speed Lines', mouse_callback)

        while len(lines) < 2:
            cv2.imshow('Select Speed Lines', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                frame = clone.copy()
                lines = []
            elif key == 27:  # ESC key
                break

        cv2.destroyWindow('Select Speed Lines')

        if len(lines) == 2:
            return sorted(lines)
        else:
            return None

    def process_video(self):
        print("Starting video processing...")
        print(f"Frame dimensions: {self.height}x{self.width}")
        print(f"FPS: {self.fps}")

        if os.path.exists(self.road_mask_file):
            self.masker.load_road_mask(self.road_mask_file)
            print(f"Loaded existing road mask from {self.road_mask_file}")
        else:
            print("Please select the road area...")
            self.masker.manual_road_selection(self.frame)
            self.masker.save_road_mask(self.road_mask_file)
            print(f"Saved road mask to {self.road_mask_file}")

        if os.path.exists(self.calibration_file):
            self.calibration.load_calibration(self.calibration_file)
            print(f"Loaded existing camera calibration from {self.calibration_file}")
        else:
            print("Performing camera calibration...")
            calibration_frames = self._collect_calibration_frames()
            self.calibration.calibrate_camera(calibration_frames)
            self.calibration.save_calibration(self.calibration_file)
            print(f"Saved camera calibration to {self.calibration_file}")

        dummy_frame = np.zeros((self.height, self.width), dtype=np.uint8)
        ipm_frame = self.calibration.apply_ipm(dummy_frame)
        ipm_height, ipm_width = ipm_frame.shape[:2]

        print("Please select two lines for speed calculation on the IPM view...")
        lines = self.select_speed_lines(ipm_frame)
        if lines is None:
            print("Line selection cancelled. Using default values.")
            line1_y = int(ipm_height * 0.2)
            line2_y = int(ipm_height * 0.8)
        else:
            line1_y, line2_y = lines

        self.speed_estimator = SpeedEstimator(ipm_height, ipm_width, self.fps, self.meters_per_pixel, line1_y, line2_y)
        print(f"SpeedEstimator initialized with IPM dimensions: {ipm_width}x{ipm_height}")
        print(f"Speed calculation lines set at y={line1_y} and y={line2_y}")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print("Video capture reset to beginning.")

        self.ipm_matrix = self.calibration.ipm_matrix
        camera_matrix = self.calibration.get_camera_matrix()
        vanishing_points = self.calibration.vanishing_points

        self.bbox_constructor = BoundingBoxConstructor(vanishing_points, camera_matrix)
        print("BoundingBoxConstructor initialized.")

        frame_count = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cv2.namedWindow('Processed Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Processed Frame', 1280, 720)

        print("Starting main processing loop...")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video reached.")
                break

            frame_count += 1

            if frame_count % self.frame_skip != 0:
                continue

            print(f"Processing frame {frame_count}/{total_frames}")

            vis_frame = frame.copy()
            vis_ipm_frame = self.calibration.apply_ipm(frame.copy())

            masked_frame = self.masker.apply_mask(frame)
            ipm_frame = self.calibration.apply_ipm(masked_frame)

            depth_map = self.depth_model.estimate_depth(ipm_frame)

            detections = self.car_detection.detect_cars(ipm_frame, self.ipm_matrix, self.detection_confidence)
            print(f"Detected {len(detections)} cars in frame {frame_count}")

            bboxes_3d = []
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                center_depth = np.mean(depth_map[int(y1):int(y2), int(x1):int(x2)])
                if np.isnan(center_depth) or np.isinf(center_depth):
                    print(f"Warning: Invalid depth value for detection {det}")
                    continue
                bbox_3d = self.bbox_constructor.construct_3d_box([x1, y1, x2, y2], center_depth, aspect_ratio=1.5)
                if bbox_3d is not None:
                    bboxes_3d.append(bbox_3d)

            tracks = self.tracker.update(bboxes_3d)
            print(f"Tracking {len(tracks)} vehicles in frame {frame_count}")

            for track_id, track in tracks.items():
                position_3d = track['bbox_3d'].mean(axis=0)
                ipm_position = self.calibration.apply_ipm(position_3d[:2].reshape(1, 2)).flatten()
                depth = position_3d[2]

                speed = self.speed_estimator.estimate_speed(track_id, ipm_position, depth, frame_count)
                print(f"Estimated speed for track {track_id}: {speed}")

                if speed is not None:
                    self.speed_calculator.update_history(track_id, speed)
                    smoothed_speed = self.speed_calculator.get_smoothed_speed(track_id)
                    confidence = self.speed_calculator.calculate_speed_confidence(track_id)
                    self._store_results(frame_count, track_id, smoothed_speed, confidence, position_3d.tolist())

                    if smoothed_speed is not None:
                        speed_text = f"ID: {track_id}, Speed: {smoothed_speed:.2f} km/h, Conf: {confidence:.2f}"
                    else:
                        speed_text = f"ID: {track_id}, Speed: Calculating..., Conf: {confidence:.2f}"
                else:
                    speed_text = f"ID: {track_id}, Speed: N/A"
                    self._store_results(frame_count, track_id, None, None, position_3d.tolist())

                corners_2d = self.bbox_constructor.project_3d_to_2d(track['bbox_3d'])
                self.draw_3d_box(vis_frame, corners_2d, color=(0, 255, 0))
                x1, y1 = corners_2d.min(axis=0).astype(int)
                cv2.putText(vis_frame, speed_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                ipm_corners = self.calibration.apply_ipm(corners_2d.reshape(-1, 1, 2)).reshape(-1, 2)
                self.draw_3d_box(vis_ipm_frame, ipm_corners, color=(0, 255, 0))
                ipm_x1, ipm_y1 = ipm_corners.min(axis=0).astype(int)
                cv2.putText(vis_ipm_frame, speed_text, (ipm_x1, ipm_y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            line1_y, line2_y = self.speed_estimator.get_line_positions()
            cv2.line(vis_ipm_frame, (0, line1_y), (ipm_width, line1_y), (0, 255, 0), 2)
            cv2.line(vis_ipm_frame, (0, line2_y), (ipm_width, line2_y), (0, 255, 0), 2)

            mask_vis = cv2.addWeighted(frame, 0.7, cv2.cvtColor(self.masker.get_mask(), cv2.COLOR_GRAY2BGR), 0.3, 0)
            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            top_row = np.hstack((vis_frame, mask_vis))
            bottom_row = np.hstack((depth_vis, vis_ipm_frame))
            combined_vis = np.vstack((top_row, bottom_row))

            combined_vis = cv2.resize(combined_vis, (self.width, self.height))

            cv2.putText(combined_vis, f"Frame: {frame_count}/{total_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            self.out.write(combined_vis)

            cv2.imshow('Processed Frame', combined_vis)

            if frame_count % 100 == 0:
                print(f"Updating JSON file at frame {frame_count}")
                self.write_results_to_json()
                self.last_json_update = frame_count

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Processing stopped by user.")
                break
            elif key == ord('p'):
                print("Processing paused. Press any key to continue.")
                cv2.waitKey(0)

        print("Writing final results to JSON file")
        self.write_results_to_json()
        print(f"Total results stored: {len(self.results)}")

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        print("Video processing completed.")

    def _collect_calibration_frames(self, num_frames=10):
        calibration_frames = []
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if ret:
                masked_frame = self.masker.apply_mask(frame)
                calibration_frames.append(masked_frame)
            else:
                break
        print(f"Collected {len(calibration_frames)} frames for calibration.")
        return calibration_frames

    def _store_results(self, frame_count, track_id, speed, confidence, position):
        result = {
            'frame': int(frame_count),
            'track_id': int(track_id),
            'speed': float(speed) if speed is not None else None,
            'confidence': float(confidence) if confidence is not None else None,
            'position': position
        }
        self.results.append(result)
        print(f"Stored result for frame {frame_count}, track {track_id}. Speed: {speed}, Position: {position}")
        print(f"Total results: {len(self.results)}")

    def write_results_to_json(self):
        try:
            print(f"Attempting to write {len(self.results)} results to JSON file")
            with open(self.json_file_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"Successfully wrote {len(self.results)} results to JSON file")
        except Exception as e:
            print(f"Error writing to JSON file: {e}")

    def draw_3d_box(self, img, corners, color=(0, 255, 0)):
        def draw_line(start, end):
            cv2.line(img, tuple(map(int, start)), tuple(map(int, end)), color, 2)

        for i in range(4):
            draw_line(corners[i], corners[(i + 1) % 4])
            draw_line(corners[i + 4], corners[(i + 1) % 4 + 4])
            draw_line(corners[i], corners[i + 4])

        return img

    def __del__(self):
        cv2.destroyAllWindows()
        print("VideoProcessor cleanup completed.")