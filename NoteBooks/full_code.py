#camera_calibration.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import random

class CameraCalibration:
    def __init__(self, model_path='models/vp_using_seg_model_best.keras'):
        self.focal_length = None
        self.principal_point = None
        self.vanishing_points = None
        self.model_path = model_path
        self.model = None
        self.ipm_matrix = None
        self.width = None
        self.height = None

    def load_model(self):
        try:
            if self.model is None:
                self.model = keras.models.load_model(self.model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def calibrate_camera(self, frames):
        self.load_model()
        self.frames = frames
        self.height, self.width = frames[0].shape[:2]
        self.principal_point = (self.width / 2, self.height / 2)

        vps = []
        for frame in frames:
            vp = self.find_vanishing_point(frame)
            if vp is not None:
                vps.append(vp)

        if not vps:
            raise ValueError("No valid vanishing points detected.")

        best_vp = self.ransac_line_fit(np.array(vps), threshold=10, iterations=100)

        vp2 = self.find_vanishing_point(random.choice(frames))
        vp2 = self.orthogonalize_vanishing_points(best_vp, vp2)

        self.focal_length = self.estimate_focal_length(best_vp, vp2)

        vp3 = np.cross(best_vp, vp2)
        vp3 /= np.linalg.norm(vp3)

        self.vanishing_points = [best_vp, vp2, vp3]

        self.compute_ipm_matrix()

        self.visualize_vanishing_points(frames[0], [best_vp, vp2, vp3])

        return self.ipm_matrix

    def ransac_line_fit(self, points, threshold, iterations):
        best_line = None
        best_inliers = 0
        for _ in range(iterations):
            sample = random.sample(range(len(points)), 2)
            p1, p2 = points[sample]

            line = np.cross(p1, p2)
            inliers = 0

            for p in points:
                dist = abs(np.dot(line, p)) / np.linalg.norm(line)
                if dist < threshold:
                    inliers += 1
            if inliers > best_inliers:
                best_inliers = inliers
                best_line = line

        if best_line is not None:
            return best_line / np.linalg.norm(best_line)
        return None

    def orthogonalize_vanishing_points(self, vp1, vp2):
        vp2_ortho = vp2 - np.dot(vp2, vp1) * vp1
        vp2_ortho /= np.linalg.norm(vp2_ortho)
        return vp2_ortho

    def find_vanishing_point(self, frame):
        self.load_model()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype('float32') / 255.0
        frame = np.expand_dims(frame, axis=0)

        segmentation, vp = self.model.predict(frame)
        vp = vp[0]

        vp = np.array([vp[0] * self.width, vp[1] * self.height, 1])
        return vp

    def estimate_focal_length(self, vp1, vp2):
        return np.sqrt(abs(np.dot(vp1, vp2)))

    def compute_ipm_matrix(self):
        src_points = np.float32([
            [0, self.height],
            [self.width, self.height],
            [self.width, 0],
            [0, 0]
        ])

        dst_points = np.float32([
            [0, self.height],
            [self.width, self.height],
            [self.width * 0.75, 0],
            [self.width * 0.25, 0]
        ])

        self.ipm_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    def visualize_vanishing_points(self, frame, vanishing_points, output_path=None):
        frame_with_vps = frame.copy()

        def normalize_vp(vp):
            if vp[2] != 0:
                return [vp[0] / vp[2], vp[1] / vp[2]]
            return [vp[0], vp[1]]

        normalized_vps = [normalize_vp(vp) for vp in vanishing_points]

        img_h, img_w = frame.shape[:2]

        for idx, vp in enumerate(normalized_vps):
            x, y = int(vp[0]), int(vp[1])

            if 0 <= x < img_w and 0 <= y < img_h:
                cv2.circle(frame_with_vps, (x, y), 10, (0, 0, 255), -1)
                cv2.putText(frame_with_vps, f'VP{idx+1}', (x + 15, y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                cv2.arrowedLine(frame_with_vps, (img_w // 2, img_h), (x, y), (0, 255, 0), 2)

        for vp in normalized_vps:
            x, y = int(vp[0]), int(vp[1])
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))

            cv2.line(frame_with_vps, (img_w // 4, img_h), (x, y), (0, 255, 0), 2)
            cv2.line(frame_with_vps, (img_w // 2, img_h), (x, y), (0, 255, 0), 2)
            cv2.line(frame_with_vps, (3 * img_w // 4, img_h), (x, y), (0, 255, 0), 2)

        if output_path:
            cv2.imwrite(output_path, frame_with_vps)
            print(f"Vanishing points visualization saved to {output_path}")
        else:
            cv2.imshow('Vanishing Points Visualization', frame_with_vps)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def get_camera_matrix(self):
        if self.focal_length is None or self.principal_point is None:
            raise ValueError("Camera not calibrated. Call calibrate_camera first.")
        return np.array([
            [self.focal_length, 0, self.principal_point[0]],
            [0, self.focal_length, self.principal_point[1]],
            [0, 0, 1]
        ])

    def save_calibration(self, filename):
        data = {
            'focal_length': self.focal_length.tolist() if isinstance(self.focal_length, np.ndarray) else self.focal_length,
            'principal_point': self.principal_point.tolist() if isinstance(self.principal_point, np.ndarray) else self.principal_point,
            'vanishing_points': [vp.tolist() for vp in self.vanishing_points],
            'ipm_matrix': self.ipm_matrix.tolist(),
            'width': self.width,
            'height': self.height
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load_calibration(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        self.focal_length = np.array(data['focal_length'])
        self.principal_point = np.array(data['principal_point'])
        self.vanishing_points = [np.array(vp) for vp in data['vanishing_points']]
        self.ipm_matrix = np.array(data['ipm_matrix'])
        self.width = data['width']
        self.height = data['height']

    def apply_ipm(self, frame):
        if self.ipm_matrix is None:
            raise ValueError("IPM matrix is not initialized. Call calibrate_camera first.")
        return cv2.warpPerspective(frame, self.ipm_matrix, (self.width, self.height))


#masker.py:
import cv2
import numpy as np
import json

class Masker:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.road_mask = None
        self.points = []

    def manual_road_selection(self, frame):
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Road Selection', frame)

        clone = frame.copy()
        cv2.namedWindow('Road Selection')
        cv2.setMouseCallback('Road Selection', mouse_callback)

        while True:
            cv2.imshow('Road Selection', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                frame = clone.copy()
                self.points = []
            elif key == ord('c'):
                break

        cv2.destroyWindow('Road Selection')

        if len(self.points) > 2:
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            points = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
            self.road_mask = mask
        else:
            print("Not enough points selected. Using full frame.")
            self.road_mask = np.ones((self.height, self.width), dtype=np.uint8) * 255

    def apply_mask(self, frame):
        if self.road_mask is None:
            raise ValueError("Road mask has not been created. Call manual_road_selection first.")
        return cv2.bitwise_and(frame, frame, mask=self.road_mask)

    def get_mask(self):
        return self.road_mask

    def apply_depth_mask(self, frame, depth_map, threshold=0.5):
        if self.road_mask is None:
            raise ValueError("Road mask has not been created. Call manual_road_selection first.")

        if frame.shape[:2] != (self.height, self.width) or depth_map.shape != (self.height, self.width):
            raise ValueError("Frame or depth map dimensions do not match the initialized dimensions.")

        # Normalize depth map to 0-1 range
        normalized_depth = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX)

        # Create binary mask based on depth threshold
        depth_mask = (normalized_depth > threshold).astype(np.uint8) * 255

        # Combine depth mask with road mask
        combined_mask = cv2.bitwise_and(self.road_mask, depth_mask)

        # Apply combined mask to frame
        return cv2.bitwise_and(frame, frame, mask=combined_mask)

    def save_road_mask(self, filename):
        np.save(filename, self.road_mask)

    def load_road_mask(self, filename):
        self.road_mask = np.load(filename)


#car_detection.py:
from ultralytics import YOLO
import numpy as np
import cv2


class CarDetection:
    def __init__(self):
        self.model = YOLO('models/yolov9m.pt')
        self.vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']

    def detect_cars(self, frame, ipm_matrix, conf_threshold=0.5):
        try:
            # Apply IPM to the frame
            ipm_frame = cv2.warpPerspective(frame, ipm_matrix, (frame.shape[1], frame.shape[0]))

            # Perform object detection on the IPM frame
            results = self.model(ipm_frame)
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    if self.model.names[int(cls)] in self.vehicle_classes and conf > conf_threshold:
                        # Transform bounding box back to original image space
                        original_box = self.transform_bbox_to_original(
                            [x1, y1, x2, y2],
                            cv2.invert(ipm_matrix)[1]
                        )
                        detections.append([*original_box, conf, cls])
            return np.array(detections)
        except Exception as e:
            print(f"Error in car detection: {str(e)}")
            return np.array([])

    def transform_bbox_to_original(self, bbox, inverse_ipm_matrix):
        def transform_point(x, y):
            p = np.dot(inverse_ipm_matrix, [x, y, 1])
            return p[:2] / p[2]

        x1, y1, x2, y2 = bbox
        p1 = transform_point(x1, y1)
        p2 = transform_point(x2, y2)
        return [*p1, *p2]



#depth_estimation.py:
import torch
import cv2
import numpy as np
from transformers import DPTForDepthEstimation, DPTImageProcessor

class DepthEstimationModel:
    def __init__(self, model_type="Intel/dpt-hybrid-midas"):
        # Initialize the DPT depth estimation model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DPTForDepthEstimation.from_pretrained(model_type)
        self.model.to(self.device).eval()
        self.processor = DPTImageProcessor.from_pretrained(model_type)

    def estimate_depth(self, img):
        # Preprocess the image
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        # Perform depth estimation
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # Normalize the depth map
        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX)

        depth_map = prediction.cpu().numpy()

        # Normalize depth map to a fixed range (e.g., 1 to 100 meters)
        depth_min, depth_max = 1, 100
        depth_map = depth_min + (depth_max - depth_min) * (depth_map - depth_map.min()) / (
                    depth_map.max() - depth_map.min())

        # Apply additional smoothing
        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)

        return depth_map


#vehicle_tracker.py:
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


class VehicleTracker:
    def __init__(self, max_frames_to_skip=10, min_hits=3, max_track_length=30, distance_threshold=50):
        self.tracks = {}
        self.frame_count = 0
        self.max_frames_to_skip = max_frames_to_skip
        self.min_hits = min_hits
        self.max_track_length = max_track_length
        self.track_id_count = 0
        self.distance_threshold = distance_threshold

    def update(self, detections):
        self.frame_count += 1
        print(f"Frame {self.frame_count}: Received {len(detections)} detections")

        # Predict new states for all tracks
        for track in self.tracks.values():
            track['kf'].predict()

        # Match detections to tracks
        if len(detections) > 0 and len(self.tracks) > 0:
            # Compute distance matrix
            distance_matrix = np.zeros((len(detections), len(self.tracks)))
            for i, detection in enumerate(detections):
                detection_center = np.mean(detection, axis=0)
                for j, (track_id, track) in enumerate(self.tracks.items()):
                    predicted_pos = track['kf'].x[:3].flatten()
                    distance_matrix[i, j] = np.linalg.norm(detection_center - predicted_pos)

            # Perform assignment
            detection_indices, track_indices = linear_sum_assignment(distance_matrix)

            # Update matched tracks and create new tracks for unmatched detections
            matched_indices = list(zip(detection_indices, track_indices))
            unmatched_detections = set(range(len(detections))) - set(detection_indices)
            unmatched_tracks = set(range(len(self.tracks))) - set(track_indices)

            for d, t in matched_indices:
                if distance_matrix[d, t] < self.distance_threshold:
                    self.update_track(list(self.tracks.keys())[t], detections[d])
                else:
                    self.create_new_track(detections[d])

            for d in unmatched_detections:
                self.create_new_track(detections[d])

            for t in unmatched_tracks:
                track_id = list(self.tracks.keys())[t]
                self.tracks[track_id]['missed_frames'] += 1

        elif len(detections) > 0:
            # If there are no existing tracks, create new tracks for all detections
            for detection in detections:
                self.create_new_track(detection)

        # Remove old tracks
        for track_id in list(self.tracks.keys()):
            if self.frame_count - self.tracks[track_id]['last_seen'] > self.max_frames_to_skip or \
                    self.frame_count - self.tracks[track_id]['first_seen'] > self.max_track_length:
                print(f"Removing track {track_id} due to age or long absence")
                del self.tracks[track_id]

        print(f"Frame {self.frame_count}: Tracking {len(self.tracks)} vehicles")
        for track_id, track in self.tracks.items():
            print(f"Track {track_id}: position = {track['kf'].x[:3].T}, velocity = {track['kf'].x[3:6].T}")

        return self.tracks

    def create_new_track(self, detection):
        self.track_id_count += 1
        kf = KalmanFilter(dim_x=6, dim_z=3)  # 3D position and velocity
        kf.F = np.array([[1, 0, 0, 1, 0, 0],
                         [0, 1, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0, 1],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])  # state transition matrix
        kf.H = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0]])  # measurement function
        kf.R *= 10  # measurement uncertainty
        kf.Q *= 0.1  # process uncertainty

        # Use the center of the 3D bounding box as the initial position
        initial_pos = np.mean(detection, axis=0)
        kf.x[:3] = initial_pos.reshape(3, 1)  # initial state (position)
        kf.x[3:] = 0  # initial velocity

        self.tracks[self.track_id_count] = {
            'kf': kf,
            'bbox_3d': detection,
            'last_seen': self.frame_count,
            'first_seen': self.frame_count,
            'missed_frames': 0,
            'hits': 1
        }
        print(f"Created new track {self.track_id_count} at position {kf.x[:3].T}")

    def update_track(self, track_id, detection):
        # Update Kalman filter with the center of the 3D bounding box
        detection_center = np.mean(detection, axis=0)
        self.tracks[track_id]['kf'].update(detection_center.reshape(3, 1))

        self.tracks[track_id]['bbox_3d'] = detection
        self.tracks[track_id]['last_seen'] = self.frame_count
        self.tracks[track_id]['missed_frames'] = 0
        self.tracks[track_id]['hits'] += 1
        print(f"Updated track {track_id}: new position = {self.tracks[track_id]['kf'].x[:3].T}")


#video_processor.py:
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



#speed_estimator.py
import numpy as np


class SpeedEstimator:
    def __init__(self, ipm_height, ipm_width, fps, meters_per_pixel, line1_y, line2_y):
        self.ipm_height = ipm_height
        self.ipm_width = ipm_width
        self.fps = fps
        self.meters_per_pixel = meters_per_pixel

        self.line1_y = line1_y
        self.line2_y = line2_y

        self.real_world_distance = self.calculate_real_world_distance()
        self.vehicle_positions = {}

    def calculate_real_world_distance(self):
        return abs(self.line2_y - self.line1_y) * self.meters_per_pixel

    def estimate_speed(self, track_id, ipm_position, depth, frame_number):
        current_time = frame_number / self.fps

        if track_id not in self.vehicle_positions:
            self.vehicle_positions[track_id] = {'last_position': ipm_position, 'last_time': current_time,
                                                'last_depth': depth}
            return None

        last_position = self.vehicle_positions[track_id]['last_position']
        last_time = self.vehicle_positions[track_id]['last_time']
        last_depth = self.vehicle_positions[track_id]['last_depth']

        # Calculate distance traveled in IPM space
        distance_pixels = np.linalg.norm(ipm_position - last_position)
        distance_meters = distance_pixels * self.meters_per_pixel

        # Incorporate depth change
        depth_change = abs(depth - last_depth)

        # Pythagoras theorem to combine horizontal and vertical movement
        total_distance = np.sqrt(distance_meters ** 2 + depth_change ** 2)

        # Calculate time difference
        time_diff = current_time - last_time

        if time_diff > 0:
            speed_mps = total_distance / time_diff  # Speed in meters per second
            speed_km_h = speed_mps * 3.6  # Convert to km/h

            # Update position, depth, and time
            self.vehicle_positions[track_id] = {'last_position': ipm_position, 'last_time': current_time,
                                                'last_depth': depth}

            return speed_km_h

        return None

    def get_line_positions(self):
        return self.line1_y, self.line2_y


#speed_calculator.py
import numpy as np


class SpeedCalculator:
    def __init__(self, smoothing_window=5, max_history=100):
        self.speed_history = {}
        self.smoothing_window = smoothing_window
        self.max_history = max_history

    def update_history(self, track_id, speed):
        if track_id not in self.speed_history:
            self.speed_history[track_id] = []

        self.speed_history[track_id].append(speed)

        if len(self.speed_history[track_id]) > self.max_history:
            self.speed_history[track_id] = self.speed_history[track_id][-self.max_history:]

    def get_smoothed_speed(self, track_id):
        if track_id not in self.speed_history or len(self.speed_history[track_id]) < self.smoothing_window:
            return None

        return np.mean(self.speed_history[track_id][-self.smoothing_window:])

    def calculate_speed_confidence(self, track_id):
        if track_id not in self.speed_history or len(self.speed_history[track_id]) < self.smoothing_window:
            return 0.0

        recent_speeds = self.speed_history[track_id][-self.smoothing_window:]
        speed_std = np.std(recent_speeds)
        speed_mean = np.mean(recent_speeds)

        if speed_mean == 0:
            return 0.0

        coefficient_of_variation = speed_std / speed_mean
        confidence = 1 / (1 + coefficient_of_variation)

        return confidence




#bounding_box_constructor.py
import numpy as np
import cv2


class BoundingBoxConstructor:
    def __init__(self, vanishing_points, camera_matrix):
        self.vanishing_points = vanishing_points
        self.camera_matrix = camera_matrix

    def construct_3d_box(self, bbox_2d, depth, aspect_ratio=None):
        try:
            x1, y1, x2, y2 = bbox_2d
            if x1 >= x2 or y1 >= y2 or depth <= 0:
                raise ValueError("Invalid bounding box or depth")

            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            width = x2 - x1
            height = y2 - y1

            # Estimate 3D dimensions
            focal_length = self.camera_matrix[0, 0]
            width_3d = width * depth / focal_length
            height_3d = height * depth / focal_length

            if aspect_ratio is None:
                length_3d = max(width_3d, height_3d)
            else:
                length_3d = max(width_3d, height_3d) * aspect_ratio

            # Construct 3D bounding box corners
            corners_3d = np.array([
                [-width_3d / 2, -height_3d / 2, length_3d / 2],
                [width_3d / 2, -height_3d / 2, length_3d / 2],
                [width_3d / 2, height_3d / 2, length_3d / 2],
                [-width_3d / 2, height_3d / 2, length_3d / 2],
                [-width_3d / 2, -height_3d / 2, -length_3d / 2],
                [width_3d / 2, -height_3d / 2, -length_3d / 2],
                [width_3d / 2, height_3d / 2, -length_3d / 2],
                [-width_3d / 2, height_3d / 2, -length_3d / 2]
            ])

            # Align with vanishing points
            rotation_matrix = np.column_stack(self.vanishing_points)
            corners_3d = np.dot(corners_3d, rotation_matrix.T)

            # Translate to center position
            corners_3d += np.array([center[0], center[1], depth])

            return corners_3d
        except Exception as e:
            print(f"Error in constructing 3D box: {str(e)}")
            return None

    def project_3d_to_2d(self, points_3d):
        print(f"Projecting 3D points: {points_3d}")
        points_2d, _ = cv2.projectPoints(points_3d, np.zeros(3), np.zeros(3), self.camera_matrix, None)
        print(f"Projected 2D points: {points_2d}")
        return points_2d.reshape(-1, 2)

    def transform_vp2_vp3(self, point):
        # Transform point from VP2-VP3 plane to image coordinates
        vp2, vp3 = self.vanishing_points[1:3]
        basis = np.column_stack((vp2, vp3, np.cross(vp2, vp3)))
        return np.dot(basis, point)

    def transform_vp1_vp2(self, point):
        # Transform point from VP1-VP2 plane to image coordinates
        vp1, vp2 = self.vanishing_points[:2]
        basis = np.column_stack((vp1, vp2, np.cross(vp1, vp2)))
        return np.dot(basis, point)



#main.py:
from video_processor import VideoProcessor


def main():
    # Video path
    video_path = 'Input/Calibration_test2.mov'

    # Calibration file path
    calibration_file = 'Files/my_camera_calibration.json'

    # Mask file path
    road_mask_file = 'Files/my_road_mask.npy'

    # Initialize the video processor
    processor = VideoProcessor(video_path, calibration_file, road_mask_file, frame_skip=5)

    # Process the video
    processor.process_video()


if __name__ == "__main__":
    main()