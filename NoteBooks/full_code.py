import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

class CameraCalibration:
    def __init__(self, model_path='vanishing_point_detection_model_video_15min.keras'):
        self.focal_length = None
        self.principal_point = None
        self.vanishing_points = None
        self.model = keras.models.load_model(model_path)

    def calibrate_camera(self, frames):
        self.frames = frames
        self.height, self.width = frames[0].shape[:2]
        self.principal_point = (self.width / 2, self.height / 2)

        # Find the first vanishing point (vehicle movement direction)
        vp1 = self.find_vanishing_point(frames[0])

        # Find the second vanishing point (perpendicular to vehicle movement)
        vp2 = self.find_vanishing_point(frames[-1])  # Using the last frame for diversity

        # Ensure vanishing points are orthogonal
        vp2 = self.orthogonalize_vanishing_points(vp1, vp2)

        # Calculate focal length
        self.focal_length = np.sqrt(abs(np.dot(vp1, vp2)))

        # Calculate the third vanishing point
        vp3 = np.cross(vp1, vp2)
        vp3 /= np.linalg.norm(vp3)

        self.vanishing_points = [vp1, vp2, vp3]

        # Create camera matrix
        camera_matrix = np.array([
            [self.focal_length, 0, self.principal_point[0]],
            [0, self.focal_length, self.principal_point[1]],
            [0, 0, 1]
        ])

        return camera_matrix, np.zeros((5, 1))  # Assuming no distortion for simplicity

    def orthogonalize_vanishing_points(self, vp1, vp2):
        # Project vp2 onto the plane perpendicular to vp1
        vp2_ortho = vp2 - np.dot(vp2, vp1) * vp1
        vp2_ortho /= np.linalg.norm(vp2_ortho)
        return vp2_ortho

    def find_vanishing_point(self, frame):
        # Preprocess the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype('float32') / 255.0
        frame = np.expand_dims(frame, axis=0)

        # Predict vanishing point
        vp = self.model.predict(frame)[0]

        # Convert normalized coordinates back to image space
        vp = np.array([vp[0] * self.width, vp[1] * self.height, 1])
        return vp

    def get_scale_factor(self, detections):
        # Calculate mean dimensions of detected vehicles
        mean_width = np.mean([d[2] - d[0] for d in detections])
        mean_height = np.mean([d[3] - d[1] for d in detections])

        # Compare with statistical data (this is a placeholder, replace with actual data)
        average_vehicle_width = 1.8  # meters
        scale_factor = average_vehicle_width / mean_width

        return scale_factor

#car_detection.py:
from ultralytics import YOLO
import numpy as np

class CarDetection:
    def __init__(self):
        # Initialize YOLO model for object detection
        self.model = YOLO('models/yolov8m.pt')
        # Define vehicle classes we're interested in
        self.vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']

    def detect_cars(self, frame):
        # Perform object detection on the frame
        results = self.model(frame)
        detections = []
        for r in results:
            # Extract bounding box coordinates, confidence, and class
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                # If the detected object is a vehicle, add it to detections
                if self.model.names[int(cls)] in self.vehicle_classes:
                    detections.append([x1, y1, x2, y2, conf, cls])
        return np.array(detections)



#depth_estimation.py:
import torch
import cv2
import numpy as np
from transformers import DPTForDepthEstimation, DPTImageProcessor

class DepthEstimationModel:
    def __init__(self, model_type="Intel/dpt-large"):
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
        return cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

#depth_masker.py:
import cv2
import numpy as np

class DepthMasker:
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

#projective_transformation.py:
import cv2
import numpy as np


class ProjectiveTransformation:
    def __init__(self):
        pass

    def back_project(self, points_2d, camera_matrix, dist_coeffs, depth):
        # Undistort points
        points_2d_undistorted = cv2.undistortPoints(points_2d, camera_matrix, dist_coeffs)

        # Get focal length and principal point
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        # Back-project to 3D
        points_3d = []
        for point in points_2d_undistorted:
            x, y = point[0]
            X = (x - cx) * depth / fx
            Y = (y - cy) * depth / fy
            Z = depth
            points_3d.append([X, Y, Z])

        return np.array(points_3d)

#vehicle_tracker.py:
import numpy as np
class VehicleTracker:
    def __init__(self, max_frames_to_skip=10, min_hits=3):
        self.tracks = {}
        self.frame_count = 0
        self.max_frames_to_skip = max_frames_to_skip
        self.min_hits = min_hits
        self.track_id_count = 0

    def update(self, detections):
        self.frame_count += 1

        # Update existing tracks
        for track_id in list(self.tracks.keys()):
            if self.frame_count - self.tracks[track_id]['last_seen'] > self.max_frames_to_skip:
                del self.tracks[track_id]
            else:
                self.tracks[track_id]['missed_frames'] += 1

        # Match detections to tracks
        if len(detections) > 0:
            for detection in detections:
                matched = False
                for track_id, track in self.tracks.items():
                    if self.iou(detection[:4], track['bbox'][:4]) > 0.3:  # IOU threshold
                        self.tracks[track_id]['bbox'] = detection
                        self.tracks[track_id]['last_seen'] = self.frame_count
                        self.tracks[track_id]['missed_frames'] = 0
                        self.tracks[track_id]['hits'] += 1
                        matched = True
                        break

                if not matched:
                    self.track_id_count += 1
                    self.tracks[self.track_id_count] = {
                        'bbox': detection,
                        'last_seen': self.frame_count,
                        'missed_frames': 0,
                        'hits': 1
                    }

        return self.tracks

    def iou(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2

        xi1, yi1, xi2, yi2 = max(x1, x3), max(y1, y3), min(x2, x4), min(y2, y4)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersect_area = (xi2 - xi1) * (yi2 - yi1)
        bbox1_area = (x2 - x1) * (y2 - y1)
        bbox2_area = (x4 - x3) * (y4 - y3)

        iou = intersect_area / float(bbox1_area + bbox2_area - intersect_area)
        return iou

    def get_track_history(self, track_id):
        return self.track_history.get(track_id, [])


#video_processor.py:
import cv2
import numpy as np
from depth_estimation import DepthEstimationModel
from camera_calibration import CameraCalibration
from car_detection import CarDetection
from projective_transformation import ProjectiveTransformation
from vehicle_tracker import VehicleTracker
from depth_masker import DepthMasker
from position_calculator import PositionCalculator
from speed_calculator import SpeedCalculator
from roi_selector import ROISelector


class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.depth_model = DepthEstimationModel()
        self.car_detection = CarDetection()
        self.tracker = VehicleTracker()

        self.cap = cv2.VideoCapture(video_path)
        self.ret, self.frame = self.cap.read()
        if not self.ret:
            raise ValueError("Failed to read the video file.")
        self.height, self.width = self.frame.shape[:2]
        print(f"Video dimensions: {self.width}x{self.height}")

        self.depth_masker = DepthMasker(self.height, self.width)
        self.roi_selector = ROISelector(self.height, self.width)

        self.calibration = CameraCalibration()
        self.camera_matrix, self.dist_coeffs = None, None

        self.transformation = ProjectiveTransformation()

        self.output_path = 'Output/output_video.mp4'
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.out = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))

        self.frame_count = 0
        self.scale_factor = None

        self.position_calculator = None
        self.speed_calculator = SpeedCalculator()

        self.current_frame_time = 0
        self.previous_frame_time = 0

    def process_video(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = self.cap.read()
        if not ret:
            raise ValueError("Failed to read the first frame.")

        self.roi_selector.select_roi(first_frame)
        self.depth_masker.manual_road_selection(first_frame)

        calibration_frames = []
        all_detections = []

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            self.current_frame_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            masked_frame = self.depth_masker.apply_mask(frame)
            depth_map = self.depth_model.estimate_depth(masked_frame)
            detections = self.car_detection.detect_cars(masked_frame)

            # Filter detections based on ROI
            roi_detections = [det for det in detections if self.roi_selector.is_in_roi((det[0], det[1]))]
            all_detections.extend(roi_detections)

            if self.frame_count % 30 == 0 and len(calibration_frames) < 10:
                calibration_frames.append(masked_frame)

            if len(calibration_frames) == 10 and self.camera_matrix is None:
                self.camera_matrix, self.dist_coeffs = self.calibration.calibrate_camera(calibration_frames)
                self.scale_factor = self.calibration.get_scale_factor(all_detections)
                self.position_calculator = PositionCalculator(self.camera_matrix)
                print("Camera calibration completed.")
                print("Camera matrix:", self.camera_matrix)
                print("Scale factor:", self.scale_factor)

            tracks = self.tracker.update(roi_detections)

            display_frame = masked_frame.copy()

            if self.camera_matrix is not None:
                for track_id, track in tracks.items():
                    if track['hits'] >= self.tracker.min_hits and track['missed_frames'] == 0:
                        x1, y1, x2, y2, conf, cls = map(float, track['bbox'])
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                        vehicle_depth = np.mean(depth_map[y1:y2, x1:x2])

                        # Draw 2D bounding box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        # Calculate real-world dimensions
                        width_3d = (x2 - x1) * self.scale_factor
                        height_3d = (y2 - y1) * self.scale_factor
                        length_3d = max(width_3d, height_3d)  # Assume length is the larger of width or height

                        # Calculate 3D position
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        center_3d = self.position_calculator.calculate_3d_position(x_center, y_center, vehicle_depth)

                        # Calculate and display speed
                        speed, speed_conf = self.speed_calculator.calculate_speed(track_id, center_3d,
                                                                                  self.current_frame_time,
                                                                                  self.previous_frame_time)

                        # Define 3D bounding box corners in camera coordinates
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

                        # Transform 3D points to camera coordinate system
                        corners_3d = corners_3d + center_3d

                        # Project 3D corners to 2D image plane
                        corners_2d, _ = cv2.projectPoints(corners_3d, np.zeros(3), np.zeros(3),
                                                          self.camera_matrix, self.dist_coeffs)
                        corners_2d = corners_2d.reshape(-1, 2).astype(int)

                        # Draw 3D bounding box
                        self.draw_3d_box(display_frame, corners_2d, color=(0, 255, 0))

                        # Display information
                        info_text = f"ID: {track_id}, Conf: {conf:.2f}, Depth: {vehicle_depth:.2f}m"
                        cv2.putText(display_frame, info_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        if speed is not None:
                            speed_text = f"Speed: {speed:.2f} m/s, Conf: {speed_conf:.2f}"
                            cv2.putText(display_frame, speed_text, (x1, y1 - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw ROI on the frame (optional, for visualization)
            cv2.polylines(display_frame, [self.roi_selector.roi_points], True, (0, 255, 0), 2)

            self.out.write(display_frame)
            cv2.imshow('Processed Frame', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.previous_frame_time = self.current_frame_time
            self.frame_count += 1

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        print(f"Processing complete. Output saved to {self.output_path}")

    def draw_3d_box(self, img, corners, color=(0, 255, 0)):
        def draw_line(start, end):
            cv2.line(img, tuple(start), tuple(end), color, 2)

        # Draw front face
        for i in range(4):
            draw_line(corners[i], corners[(i + 1) % 4])

        # Draw back face
        for i in range(4):
            draw_line(corners[i + 4], corners[((i + 1) % 4) + 4])

        # Draw lines connecting front and back faces
        for i in range(4):
            draw_line(corners[i], corners[i + 4])

        return img


#speed_calculator.py
import numpy as np
from scipy.signal import savgol_filter


class SpeedCalculator:
    def __init__(self, smoothing_window=5, speed_confidence_threshold=0.8):
        self.previous_positions = {}
        self.speed_history = {}
        self.smoothing_window = smoothing_window
        self.speed_confidence_threshold = speed_confidence_threshold

    def calculate_speed(self, track_id, current_position, current_time, previous_time):
        if track_id in self.previous_positions:
            previous_position = self.previous_positions[track_id]
            time_diff = current_time - previous_time

            if time_diff > 0:
                displacement = np.linalg.norm(current_position - previous_position)
                speed = displacement / time_diff

                if track_id not in self.speed_history:
                    self.speed_history[track_id] = []
                self.speed_history[track_id].append(speed)

                # Apply smoothing if we have enough speed measurements
                if len(self.speed_history[track_id]) >= self.smoothing_window:
                    smoothed_speed = savgol_filter(self.speed_history[track_id], self.smoothing_window, 2)[-1]
                    speed_confidence = self.calculate_speed_confidence(self.speed_history[track_id])
                else:
                    smoothed_speed = speed
                    speed_confidence = 0.5  # Lower confidence for initial measurements

                self.previous_positions[track_id] = current_position
                return smoothed_speed, speed_confidence
        else:
            self.previous_positions[track_id] = current_position

        return None, 0.0

    def calculate_speed_confidence(self, speed_history):
        if len(speed_history) < 2:
            return 0.5

        # Calculate the coefficient of variation
        cv = np.std(speed_history) / np.mean(speed_history)

        # Convert CV to a confidence score (lower CV means higher confidence)
        confidence = 1 / (1 + cv)

        # Apply a threshold
        return min(confidence, self.speed_confidence_threshold)



# roi_selector.py
import cv2
import numpy as np


class ROISelector:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.roi_mask = None
        self.roi_points = None

    def select_roi(self, frame):
        roi_height = int(self.height * 0.6)  # 60% of the frame height
        top_left = (0, self.height - roi_height)
        bottom_right = (self.width, self.height)

        self.roi_points = np.array([
            top_left,
            (self.width, self.height - roi_height),
            bottom_right,
            (0, self.height)
        ], dtype=np.int32)

        # Create the ROI mask
        self.roi_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(self.roi_mask, [self.roi_points], 255)

        # Draw the ROI on the frame for visualization
        roi_frame = frame.copy()
        cv2.polylines(roi_frame, [self.roi_points], True, (0, 255, 0), 2)
        cv2.putText(roi_frame, "ROI", (10, self.height - roi_height + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with ROI
        cv2.imshow('ROI Selection', roi_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_roi(self, frame):
        if self.roi_mask is None:
            raise ValueError("ROI has not been selected. Call select_roi first.")
        return cv2.bitwise_and(frame, frame, mask=self.roi_mask)

    def is_in_roi(self, point):
        x, y = point
        return cv2.pointPolygonTest(self.roi_points, (x, y), False) >= 0

    def get_roi_mask(self):
        return self.roi_mask


#position_calculator.py:
import numpy as np

class PositionCalculator:
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix

    def calculate_3d_position(self, x, y, depth):
        # Convert image coordinates to camera coordinates
        x_cam = (x - self.camera_matrix[0, 2]) * depth / self.camera_matrix[0, 0]
        y_cam = (y - self.camera_matrix[1, 2]) * depth / self.camera_matrix[1, 1]
        return np.array([x_cam, y_cam, depth])


#main.py:
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
