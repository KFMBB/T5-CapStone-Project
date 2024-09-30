#camera_calibration.py
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


class VideoProcessor:
    def __init__(self, video_path, calibration_file='camera_calibration.json', road_mask_file='road_mask.npy',
                 detection_confidence=0.4):
        self.video_path = video_path
        self.calibration_file = calibration_file
        self.road_mask_file = road_mask_file
        self.detection_confidence = detection_confidence

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

        self.results = []  # To store results for JSON logging

    def process_video(self):
        # Road mask selection or loading
        if os.path.exists(self.road_mask_file):
            self.masker.load_road_mask(self.road_mask_file)
            print(f"Loaded existing road mask from {self.road_mask_file}")
        else:
            print("Please select the road area...")
            self.masker.manual_road_selection(self.frame)
            self.masker.save_road_mask(self.road_mask_file)
            print(f"Saved road mask to {self.road_mask_file}")

        # Camera calibration
        if os.path.exists(self.calibration_file):
            self.calibration.load_calibration(self.calibration_file)
            print(f"Loaded existing camera calibration from {self.calibration_file}")
        else:
            print("Performing camera calibration...")
            calibration_frames = [self.masker.apply_mask(self.cap.read()[1]) for _ in range(10)]
            self.calibration.calibrate_camera(calibration_frames)
            self.calibration.save_calibration(self.calibration_file)
            print(f"Saved camera calibration to {self.calibration_file}")

        # Get necessary matrices
        self.ipm_matrix = self.calibration.ipm_matrix
        camera_matrix = self.calibration.get_camera_matrix()
        vanishing_points = self.calibration.vanishing_points

        # Initialize BoundingBoxConstructor with calibration results
        self.bbox_constructor = BoundingBoxConstructor(vanishing_points, camera_matrix)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cv2.namedWindow('Processed Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Processed Frame', 1280, 720)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / self.fps

            # Create copies for visualization
            vis_frame = frame.copy()
            vis_ipm_frame = self.calibration.apply_ipm(frame.copy())

            # Apply depth masking
            masked_frame = self.masker.apply_mask(frame)

            # Apply IPM
            ipm_frame = self.calibration.apply_ipm(masked_frame)

            # Estimate depth
            depth_map = self.depth_model.estimate_depth(ipm_frame)

            # Detect vehicles
            detections = self.car_detection.detect_cars(ipm_frame, self.ipm_matrix, self.detection_confidence)

            # Construct 3D bounding boxes
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

            # Track vehicles
            tracks = self.tracker.update(bboxes_3d)

            # Visualize results
            for track_id, track in tracks.items():
                corners_3d = track['bbox_3d']
                corners_2d = self.bbox_constructor.project_3d_to_2d(corners_3d)

                # Draw 3D bounding box in original frame
                self.draw_3d_box(vis_frame, corners_2d, color=(0, 255, 0))

                # Calculate speed
                current_position = np.mean(corners_3d, axis=0)
                speed, confidence = self.speed_calculator.calculate_speed(
                    track_id, current_position, current_time, current_time - 1 / self.fps, unit='km/h'
                )

                # Prepare text to display
                speed_text = f"ID: {track_id}, Speed: {speed:.2f} km/h, Conf: {confidence:.2f}" if speed is not None else f"ID: {track_id}, Speed: N/A"

                # Display in original frame
                x1, y1 = corners_2d.min(axis=0).astype(int)
                cv2.putText(vis_frame, speed_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Transform corners to IPM view
                ipm_corners = self.calibration.apply_ipm(corners_2d.reshape(-1, 1, 2)).reshape(-1, 2)

                # Draw 3D bounding box in IPM view
                self.draw_3d_box(vis_ipm_frame, ipm_corners, color=(0, 255, 0))

                # Display text in IPM view
                ipm_x1, ipm_y1 = ipm_corners.min(axis=0).astype(int)
                cv2.putText(vis_ipm_frame, speed_text, (ipm_x1, ipm_y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Store results for JSON logging
                self.results.append({
                    'frame': frame_count,
                    'track_id': track_id,
                    'speed': speed if speed is not None else 'N/A',
                    'confidence': confidence if confidence is not None else 'N/A',
                    'position': current_position.tolist()
                })

            # Combine visualizations
            mask_vis = cv2.addWeighted(frame, 0.7, cv2.cvtColor(self.masker.get_mask(), cv2.COLOR_GRAY2BGR), 0.3, 0)
            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            top_row = np.hstack((vis_frame, mask_vis))
            bottom_row = np.hstack((depth_vis, vis_ipm_frame))
            combined_vis = np.vstack((top_row, bottom_row))

            # Resize the combined visualization to fit the output video dimensions
            combined_vis = cv2.resize(combined_vis, (self.width, self.height))

            # Add debug information
            cv2.putText(combined_vis, f"Frame: {frame_count}/{total_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Write frame to output video
            self.out.write(combined_vis)

            # Display the frame
            cv2.imshow('Processed Frame', combined_vis)

            # Print debug information
            print(f"Displayed frame {frame_count}/{total_frames}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):  # Add a pause functionality
                cv2.waitKey(0)

            if frame_count % 100 == 0:
                print(f"Processed frame {frame_count}/{total_frames} ({frame_count / total_frames * 100:.2f}%)")

        # Save results to JSON file
        with open('Output/speed_estimation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        print("Video processing completed.")

    def draw_3d_box(self, img, corners, color=(0, 255, 0)):
        def draw_line(start, end):
            cv2.line(img, tuple(map(int, start)), tuple(map(int, end)), color, 2)

        # Draw bottom face
        for i in range(4):
            draw_line(corners[i], corners[(i + 1) % 4])

        # Draw top face
        for i in range(4):
            draw_line(corners[i + 4], corners[(i + 1) % 4 + 4])

        # Draw vertical lines
        for i in range(4):
            draw_line(corners[i], corners[i + 4])

        return img

    def __del__(self):
        cv2.destroyAllWindows()


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



#vehicle_tracker.py:
import numpy as np
from scipy.optimize import linear_sum_assignment

class VehicleTracker:
    def __init__(self, max_frames_to_skip=10, min_hits=3, max_track_length=30):
        self.tracks = {}
        self.frame_count = 0
        self.max_frames_to_skip = max_frames_to_skip
        self.min_hits = min_hits
        self.max_track_length = max_track_length
        self.track_id_count = 0

    def update(self, detections_3d):
        self.frame_count += 1

        # Update existing tracks
        for track_id in list(self.tracks.keys()):
            if self.frame_count - self.tracks[track_id]['last_seen'] > self.max_frames_to_skip or \
               self.frame_count - self.tracks[track_id]['first_seen'] > self.max_track_length:
                del self.tracks[track_id]
            else:
                self.tracks[track_id]['missed_frames'] += 1

        # Match detections to tracks
        if len(detections_3d) > 0 and len(self.tracks) > 0:
            # Compute IoU matrix
            iou_matrix = np.zeros((len(detections_3d), len(self.tracks)))
            for i, detection in enumerate(detections_3d):
                for j, (track_id, track) in enumerate(self.tracks.items()):
                    iou_matrix[i, j] = self.iou_3d(detection, track['bbox_3d'])

            # Handle NaN and inf values
            iou_matrix = np.nan_to_num(iou_matrix, nan=0.0, posinf=0.0, neginf=0.0)

            # Match using Hungarian algorithm
            detection_indices, track_indices = linear_sum_assignment(-iou_matrix)

            matched_indices = np.column_stack((detection_indices, track_indices))

            for d, t in matched_indices:
                if iou_matrix[d, t] > 0.3:  # IOU threshold
                    track_id = list(self.tracks.keys())[t]
                    self.tracks[track_id]['bbox_3d'] = detections_3d[d]
                    self.tracks[track_id]['last_seen'] = self.frame_count
                    self.tracks[track_id]['missed_frames'] = 0
                    self.tracks[track_id]['hits'] += 1
                else:
                    self.create_new_track(detections_3d[d])

            # Create new tracks for unmatched detections
            unmatched_detections = set(range(len(detections_3d))) - set(detection_indices)
            for d in unmatched_detections:
                self.create_new_track(detections_3d[d])
        elif len(detections_3d) > 0:
            # If there are no existing tracks, create new tracks for all detections
            for detection in detections_3d:
                self.create_new_track(detection)

        return self.tracks

    def create_new_track(self, detection):
        self.track_id_count += 1
        self.tracks[self.track_id_count] = {
            'bbox_3d': detection,
            'last_seen': self.frame_count,
            'first_seen': self.frame_count,
            'missed_frames': 0,
            'hits': 1
        }

    def iou_3d(self, bbox1, bbox2):
        try:
            def volume(bbox):
                return np.prod(np.max(bbox, axis=0) - np.min(bbox, axis=0))

            intersection = np.minimum(np.max(bbox1, axis=0), np.max(bbox2, axis=0)) - np.maximum(np.min(bbox1, axis=0), np.min(bbox2, axis=0))
            intersection = np.maximum(intersection, 0)
            intersection_volume = np.prod(intersection)

            volume1 = volume(bbox1)
            volume2 = volume(bbox2)

            iou = intersection_volume / (volume1 + volume2 - intersection_volume)
            return np.clip(iou, 0, 1)  # Ensure IoU is between 0 and 1
        except Exception as e:
            print(f"Error in IOU calculation: {str(e)}")
            return 0


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


class VideoProcessor:
    def __init__(self, video_path, calibration_file='camera_calibration.json', road_mask_file='road_mask.npy',
                 detection_confidence=0.4):
        self.video_path = video_path
        self.calibration_file = calibration_file
        self.road_mask_file = road_mask_file
        self.detection_confidence = detection_confidence

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

        self.results = []  # To store results for JSON logging

    def process_video(self):
        # Road mask selection or loading
        if os.path.exists(self.road_mask_file):
            self.masker.load_road_mask(self.road_mask_file)
            print(f"Loaded existing road mask from {self.road_mask_file}")
        else:
            print("Please select the road area...")
            self.masker.manual_road_selection(self.frame)
            self.masker.save_road_mask(self.road_mask_file)
            print(f"Saved road mask to {self.road_mask_file}")

        # Camera calibration
        if os.path.exists(self.calibration_file):
            self.calibration.load_calibration(self.calibration_file)
            print(f"Loaded existing camera calibration from {self.calibration_file}")
        else:
            print("Performing camera calibration...")
            calibration_frames = [self.masker.apply_mask(self.cap.read()[1]) for _ in range(10)]
            self.calibration.calibrate_camera(calibration_frames)
            self.calibration.save_calibration(self.calibration_file)
            print(f"Saved camera calibration to {self.calibration_file}")

        # Get necessary matrices
        self.ipm_matrix = self.calibration.ipm_matrix
        camera_matrix = self.calibration.get_camera_matrix()
        vanishing_points = self.calibration.vanishing_points

        # Initialize BoundingBoxConstructor with calibration results
        self.bbox_constructor = BoundingBoxConstructor(vanishing_points, camera_matrix)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cv2.namedWindow('Processed Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Processed Frame', 1280, 720)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / self.fps

            # Create copies for visualization
            vis_frame = frame.copy()
            vis_ipm_frame = self.calibration.apply_ipm(frame.copy())

            # Apply depth masking
            masked_frame = self.masker.apply_mask(frame)

            # Apply IPM
            ipm_frame = self.calibration.apply_ipm(masked_frame)

            # Estimate depth
            depth_map = self.depth_model.estimate_depth(ipm_frame)

            # Detect vehicles
            detections = self.car_detection.detect_cars(ipm_frame, self.ipm_matrix, self.detection_confidence)

            # Construct 3D bounding boxes
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

            # Track vehicles
            tracks = self.tracker.update(bboxes_3d)

            # Visualize results
            for track_id, track in tracks.items():
                corners_3d = track['bbox_3d']
                corners_2d = self.bbox_constructor.project_3d_to_2d(corners_3d)

                # Draw 3D bounding box in original frame
                self.draw_3d_box(vis_frame, corners_2d, color=(0, 255, 0))

                # Calculate speed
                current_position = np.mean(corners_3d, axis=0)
                speed, confidence = self.speed_calculator.calculate_speed(
                    track_id, current_position, current_time, current_time - 1 / self.fps, unit='km/h'
                )

                # Prepare text to display
                speed_text = f"ID: {track_id}, Speed: {speed:.2f} km/h, Conf: {confidence:.2f}" if speed is not None else f"ID: {track_id}, Speed: N/A"

                # Display in original frame
                x1, y1 = corners_2d.min(axis=0).astype(int)
                cv2.putText(vis_frame, speed_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Transform corners to IPM view
                ipm_corners = self.calibration.apply_ipm(corners_2d.reshape(-1, 1, 2)).reshape(-1, 2)

                # Draw 3D bounding box in IPM view
                self.draw_3d_box(vis_ipm_frame, ipm_corners, color=(0, 255, 0))

                # Display text in IPM view
                ipm_x1, ipm_y1 = ipm_corners.min(axis=0).astype(int)
                cv2.putText(vis_ipm_frame, speed_text, (ipm_x1, ipm_y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Store results for JSON logging
                self.results.append({
                    'frame': frame_count,
                    'track_id': track_id,
                    'speed': speed if speed is not None else 'N/A',
                    'confidence': confidence if confidence is not None else 'N/A',
                    'position': current_position.tolist()
                })

            # Combine visualizations
            mask_vis = cv2.addWeighted(frame, 0.7, cv2.cvtColor(self.masker.get_mask(), cv2.COLOR_GRAY2BGR), 0.3, 0)
            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            top_row = np.hstack((vis_frame, mask_vis))
            bottom_row = np.hstack((depth_vis, vis_ipm_frame))
            combined_vis = np.vstack((top_row, bottom_row))

            # Resize the combined visualization to fit the output video dimensions
            combined_vis = cv2.resize(combined_vis, (self.width, self.height))

            # Add debug information
            cv2.putText(combined_vis, f"Frame: {frame_count}/{total_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Write frame to output video
            self.out.write(combined_vis)

            # Display the frame
            cv2.imshow('Processed Frame', combined_vis)

            # Print debug information
            print(f"Displayed frame {frame_count}/{total_frames}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):  # Add a pause functionality
                cv2.waitKey(0)

            if frame_count % 100 == 0:
                print(f"Processed frame {frame_count}/{total_frames} ({frame_count / total_frames * 100:.2f}%)")

        # Save results to JSON file
        with open('Output/speed_estimation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        print("Video processing completed.")

    def draw_3d_box(self, img, corners, color=(0, 255, 0)):
        def draw_line(start, end):
            cv2.line(img, tuple(map(int, start)), tuple(map(int, end)), color, 2)

        # Draw bottom face
        for i in range(4):
            draw_line(corners[i], corners[(i + 1) % 4])

        # Draw top face
        for i in range(4):
            draw_line(corners[i + 4], corners[(i + 1) % 4 + 4])

        # Draw vertical lines
        for i in range(4):
            draw_line(corners[i], corners[i + 4])

        return img

    def __del__(self):
        cv2.destroyAllWindows()


#speed_calculator.py
import numpy as np
from scipy.signal import savgol_filter

class SpeedCalculator:
    def __init__(self, smoothing_window=5, speed_confidence_threshold=0.8, max_history=100):
        self.previous_positions = {}
        self.speed_history = {}
        self.smoothing_window = smoothing_window
        self.speed_confidence_threshold = speed_confidence_threshold
        self.max_history = max_history

    def calculate_speed(self, track_id, current_position, current_time, previous_time, unit='m/s'):
        try:
            if track_id in self.previous_positions:
                previous_position = self.previous_positions[track_id]
                time_diff = current_time - previous_time

                if time_diff > 0:
                    displacement = np.linalg.norm(current_position - previous_position)
                    speed = displacement / time_diff

                    # Apply a simple low-pass filter
                    if track_id not in self.speed_history:
                        self.speed_history[track_id] = []

                    if len(self.speed_history[track_id]) > 0:
                        speed = 0.7 * speed + 0.3 * self.speed_history[track_id][-1]

                    self.speed_history[track_id].append(speed)

                    # Limit history size
                    if len(self.speed_history[track_id]) > self.max_history:
                        self.speed_history[track_id] = self.speed_history[track_id][-self.max_history:]

                    if len(self.speed_history[track_id]) >= self.smoothing_window:
                        smoothed_speed = np.median(self.speed_history[track_id][-self.smoothing_window:])
                        speed_confidence = self.calculate_speed_confidence(self.speed_history[track_id])
                    else:
                        smoothed_speed = speed
                        speed_confidence = 0.5

                    self.previous_positions[track_id] = current_position
                    return self.convert_speed(smoothed_speed, unit), speed_confidence
            else:
                self.previous_positions[track_id] = current_position

            return None, 0.0
        except Exception as e:
            print(f"Error in speed calculation: {str(e)}")
            return None, 0.0

    def calculate_speed_confidence(self, speed_history):
        if len(speed_history) < 2:
            return 0.5

        cv = np.std(speed_history) / np.mean(speed_history)
        confidence = 1 / (1 + cv)
        return min(confidence, self.speed_confidence_threshold)

    def convert_speed(self, speed, unit):
        if unit == 'm/s':
            return speed
        elif unit == 'km/h':
            return speed * 3.6
        elif unit == 'mph':
            return speed * 2.237
        else:
            raise ValueError(f"Unsupported unit: {unit}")



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
                [-width_3d/2, -height_3d/2, length_3d/2],
                [width_3d/2, -height_3d/2, length_3d/2],
                [width_3d/2, height_3d/2, length_3d/2],
                [-width_3d/2, height_3d/2, length_3d/2],
                [-width_3d/2, -height_3d/2, -length_3d/2],
                [width_3d/2, -height_3d/2, -length_3d/2],
                [width_3d/2, height_3d/2, -length_3d/2],
                [-width_3d/2, height_3d/2, -length_3d/2]
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

    def project_3d_to_2d(self, corners_3d):
        # Project 3D corners to 2D image plane
        corners_2d, _ = cv2.projectPoints(corners_3d, np.zeros(3), np.zeros(3), self.camera_matrix, None)
        return corners_2d.reshape(-1, 2)

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
    processor = VideoProcessor(video_path, calibration_file, road_mask_file)

    # Process the video
    processor.process_video()


if __name__ == "__main__":
    main()