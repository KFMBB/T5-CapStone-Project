#camera_calibration.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json

class CameraCalibration:
    def __init__(self, model_path='models/best_model.keras'):
        self.focal_length = None
        self.principal_point = None
        self.vanishing_points = None
        self.model = keras.models.load_model(model_path)
        self.ipm_matrix = None

    def calibrate_camera(self, frames):
        self.frames = frames
        self.height, self.width = frames[0].shape[:2]
        self.principal_point = (self.width / 2, self.height / 2)

        # Find the vanishing points
        vp1 = self.find_vanishing_point(frames[0])
        vp2 = self.find_vanishing_point(frames[-1])
        vp2 = self.orthogonalize_vanishing_points(vp1, vp2)

        self.focal_length = np.sqrt(abs(np.dot(vp1, vp2)))
        vp3 = np.cross(vp1, vp2)
        vp3 /= np.linalg.norm(vp3)

        self.vanishing_points = [vp1, vp2, vp3]

        # Compute IPM matrix
        self.compute_ipm_matrix()

        return self.ipm_matrix

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


    def compute_ipm_matrix(self):
        # Define source points (in image plane)
        src_points = np.float32([
            [0, self.height],
            [self.width, self.height],
            [self.width, 0],
            [0, 0]
        ])

        # Define destination points (in bird's eye view)
        dst_points = np.float32([
            [0, self.height],
            [self.width, self.height],
            [self.width * 0.75, 0],
            [self.width * 0.25, 0]
        ])

        # Compute the IPM matrix
        self.ipm_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    def apply_ipm(self, frame):
        if self.ipm_matrix is None:
            raise ValueError("IPM matrix has not been computed. Call calibrate_camera first.")
        return cv2.warpPerspective(frame, self.ipm_matrix, (self.width, self.height))

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
            'focal_length': self.focal_length.tolist() if isinstance(self.focal_length,
                                                                     np.ndarray) else self.focal_length,
            'principal_point': self.principal_point.tolist() if isinstance(self.principal_point,
                                                                           np.ndarray) else self.principal_point,
            'vanishing_points': [vp.tolist() for vp in self.vanishing_points],
            'ipm_matrix': self.ipm_matrix.tolist()
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

#car_detection.py:
from ultralytics import YOLO
import numpy as np
import cv2


class CarDetection:
    def __init__(self):
        self.model = YOLO('models/yolov8m.pt')
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
import json

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

# #projective_transformation.py:
# import cv2
# import numpy as np
#
#
# class ProjectiveTransformation:
#     def __init__(self, camera_calibration):
#         self.camera_calibration = camera_calibration
#         self.transformation_matrix = None
#         self.inverse_transformation_matrix = None
#
#     def construct_transformation(self, mask):
#         height, width = mask.shape[:2]
#
#         # Define source points (trapezoid)
#         src_points = np.float32([
#             [width * 0.1, height * 0.8],  # Bottom left
#             [width * 0.9, height * 0.8],  # Bottom right
#             [width * 0.6, height * 0.4],  # Top right
#             [width * 0.4, height * 0.4]  # Top left
#         ])
#
#         # Define destination points (rectangle)
#         dst_points = np.float32([
#             [width * 0.2, height],  # Bottom left
#             [width * 0.8, height],  # Bottom right
#             [width * 0.8, 0],  # Top right
#             [width * 0.2, 0]  # Top left
#         ])
#
#         # Compute the perspective transformation matrix
#         self.transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
#         self.inverse_transformation_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
#
#         # Visualize source points on original image
#         original_vis = self._visualize_points(mask, src_points)
#         cv2.imshow('Original with Source Points', original_vis)
#
#         # Apply transformation and visualize destination points
#         transformed_mask = self.apply_transformation(mask)
#         transformed_vis = self._visualize_points(transformed_mask, dst_points, color=(0, 0, 255))
#         cv2.imshow('Transformed with Destination Points', transformed_vis)
#
#         cv2.waitKey(1)
#
#         return original_vis, transformed_vis
#
#     def apply_transformation(self, image):
#         if self.transformation_matrix is None:
#             raise ValueError("Transformation matrix has not been constructed")
#         return cv2.warpPerspective(image, self.transformation_matrix, (image.shape[1], image.shape[0]))
#
#     def apply_inverse_transformation(self, points):
#         if self.inverse_transformation_matrix is None:
#             raise ValueError("Inverse transformation matrix has not been constructed")
#
#         # Reshape points to (N, 1, 2) format
#         points = points.reshape(-1, 1, 2)
#
#         # Apply perspective transformation
#         transformed_points = cv2.perspectiveTransform(points.astype(np.float32), self.inverse_transformation_matrix)
#
#         # Reshape back to original format
#         return transformed_points.reshape(-1, 2)
#
#     def _visualize_points(self, image, points, color=(0, 255, 0)):
#         vis_image = image.copy()
#         for i, point in enumerate(points):
#             x, y = int(point[0]), int(point[1])
#             cv2.circle(vis_image, (x, y), 5, color, -1)
#             cv2.putText(vis_image, str(i), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#         return vis_image

#vehicle_tracker.py:
import numpy as np

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
        if len(detections_3d) > 0:
            # Compute IoU matrix
            iou_matrix = np.zeros((len(detections_3d), len(self.tracks)))
            for i, detection in enumerate(detections_3d):
                for j, (track_id, track) in enumerate(self.tracks.items()):
                    iou_matrix[i, j] = self.iou_3d(detection, track['bbox_3d'])

            # Match using Hungarian algorithm (you'll need to install scipy for this)
            from scipy.optimize import linear_sum_assignment
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
            return iou
        except Exception as e:
            print(f"Error in IOU calculation: {str(e)}")
            return 0


#video_processor.py:
import cv2
import numpy as np
import os
from camera_calibration import CameraCalibration
from car_detection import CarDetection
from bounding_box_constructor import BoundingBoxConstructor
from vehicle_tracker import VehicleTracker
from speed_calculator import SpeedCalculator
from depth_estimation import DepthEstimationModel
from depth_masker import DepthMasker


class VideoProcessor:
    def __init__(self, video_path, calibration_file='camera_calibration.json', road_mask_file='road_mask.npy', detection_confidence=0.4):
        self.video_path = video_path
        self.calibration_file = calibration_file
        self.road_mask_file = road_mask_file
        self.calibration = CameraCalibration()
        self.car_detection = CarDetection()
        self.detection_confidence = detection_confidence
        self.depth_model = DepthEstimationModel()
        self.tracker = VehicleTracker(max_frames_to_skip=10, min_hits=3, max_track_length=30)
        self.speed_calculator = SpeedCalculator(smoothing_window=5, speed_confidence_threshold=0.8, max_history=100)


        self.cap = cv2.VideoCapture(video_path)
        self.ret, self.frame = self.cap.read()
        if not self.ret:
            raise ValueError("Failed to read the video file.")
        self.height, self.width = self.frame.shape[:2]

        self.depth_masker = DepthMasker(self.height, self.width)
        self.ipm_matrix = None
        self.bbox_constructor = None

        self.output_path = 'Output/output_video.mp4'
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.out = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))

    def process_video(self):
        # Road mask selection or loading
        if os.path.exists(self.road_mask_file):
            # Load existing road mask
            self.depth_masker.load_road_mask(self.road_mask_file)
            print(f"Loaded existing road mask from {self.road_mask_file}")
        else:
            # Perform manual road selection
            print("Please select the road area...")
            self.depth_masker.manual_road_selection(self.frame)

            # Save the road mask for future use
            self.depth_masker.save_road_mask(self.road_mask_file)
            print(f"Saved road mask to {self.road_mask_file}")

        # Camera calibration
        if os.path.exists(self.calibration_file):
            self.calibration.load_calibration(self.calibration_file)
            print(f"Loaded existing camera calibration from {self.calibration_file}")
        else:
            print("Performing camera calibration...")
            calibration_frames = [self.depth_masker.apply_mask(self.cap.read()[1]) for _ in range(10)]
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
        print(f"Processing video with {total_frames} frames...")

        ipm_view_saved = False

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / self.fps

            # Create a copy of the original frame for visualization
            vis_frame = frame.copy()

            # Apply depth masking
            masked_frame = self.depth_masker.apply_mask(frame)

            # Visualize the mask
            mask_vis = cv2.addWeighted(frame, 0.7, cv2.cvtColor(self.depth_masker.get_mask(), cv2.COLOR_GRAY2BGR), 0.3,
                                       0)

            # Apply IPM
            ipm_frame = self.calibration.apply_ipm(masked_frame)

            # Save one frame of the IPM view
            if not ipm_view_saved:
                cv2.imwrite('Output/ipm_view.jpg', ipm_frame)
                ipm_view_saved = True

            # Estimate depth
            depth_map = self.depth_model.estimate_depth(ipm_frame)

            # Normalize depth map for visualization
            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # Apply depth-based masking
            depth_masked_frame = self.depth_masker.apply_depth_mask(ipm_frame, depth_map)

            # Detect vehicles
            detections = self.car_detection.detect_cars(depth_masked_frame, self.ipm_matrix, self.detection_confidence)

            # Construct 3D bounding boxes
            bboxes_3d = []
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                center_depth = np.mean(depth_map[int(y1):int(y2), int(x1):int(x2)])
                bbox_3d = self.bbox_constructor.construct_3d_box([x1, y1, x2, y2], center_depth, aspect_ratio=1.5)
                if bbox_3d is not None:
                    bboxes_3d.append(bbox_3d)

            # Track vehicles
            tracks = self.tracker.update(bboxes_3d)

            # Calculate speeds and visualize results
            for track_id, track in tracks.items():
                if track['hits'] >= self.tracker.min_hits and track['missed_frames'] == 0:
                    current_position = np.mean(track['bbox_3d'], axis=0)
                    speed, confidence = self.speed_calculator.calculate_speed(
                        track_id, current_position, current_time, current_time - 1 / self.fps, unit='km/h'
                    )

                    # Visualize results
                    corners_2d = self.bbox_constructor.project_3d_to_2d(track['bbox_3d'])

                    # Draw 2D bounding box
                    x1, y1 = corners_2d.min(axis=0).astype(int)
                    x2, y2 = corners_2d.max(axis=0).astype(int)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw 3D bounding box
                    self.draw_3d_box(vis_frame, corners_2d)

                    if speed is not None:
                        # Display ID, speed, and confidence
                        cv2.putText(vis_frame, f"ID: {track_id}", (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(vis_frame, f"Speed: {speed:.2f} km/h", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(vis_frame, f"Conf: {confidence:.2f}", (x1, y1 + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Combine visualizations
            top_row = np.hstack((vis_frame, mask_vis))
            bottom_row = np.hstack((depth_vis, cv2.resize(ipm_frame, (vis_frame.shape[1], vis_frame.shape[0]))))
            combined_vis = np.vstack((top_row, bottom_row))

            # Resize the combined visualization to fit the output video dimensions
            combined_vis = cv2.resize(combined_vis, (self.width, self.height))

            # Write frame to output video
            self.out.write(combined_vis)

            # Display the frame (optional)
            cv2.imshow('Processed Frame', combined_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if frame_count % 100 == 0:
                print(f"Processed frame {frame_count}/{total_frames} ({frame_count / total_frames * 100:.2f}%)")

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        print("Video processing completed.")

    def draw_3d_box(self, img, corners):
        # Draw the base of the 3D box
        for i in range(4):
            cv2.line(img, tuple(corners[i].astype(int)), tuple(corners[(i + 1) % 4].astype(int)), (0, 255, 0), 2)

        # Draw the top of the 3D box
        for i in range(4):
            cv2.line(img, tuple(corners[i + 4].astype(int)), tuple(corners[(i + 1) % 4 + 4].astype(int)), (0, 255, 0),
                     2)

        # Draw the vertical lines
        for i in range(4):
            cv2.line(img, tuple(corners[i].astype(int)), tuple(corners[i + 4].astype(int)), (0, 255, 0), 2)

        return img


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

                    if track_id not in self.speed_history:
                        self.speed_history[track_id] = []
                    self.speed_history[track_id].append(speed)

                    # Limit history size
                    if len(self.speed_history[track_id]) > self.max_history:
                        self.speed_history[track_id] = self.speed_history[track_id][-self.max_history:]

                    if len(self.speed_history[track_id]) >= self.smoothing_window:
                        smoothed_speed = savgol_filter(self.speed_history[track_id], self.smoothing_window, 2)[-1]
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



# # roi_selector.py
# import cv2
# import numpy as np
#
#
# class ROISelector:
#     def __init__(self, height, width):
#         self.height = height
#         self.width = width
#         self.roi_mask = None
#         self.roi_points = None
#
#     def select_roi(self, frame):
#         roi_height = int(self.height * 0.6)  # 60% of the frame height
#         top_left = (0, self.height - roi_height)
#         bottom_right = (self.width, self.height)
#
#         self.roi_points = np.array([
#             top_left,
#             (self.width, self.height - roi_height),
#             bottom_right,
#             (0, self.height)
#         ], dtype=np.int32)
#
#         # Create the ROI mask
#         self.roi_mask = np.zeros((self.height, self.width), dtype=np.uint8)
#         cv2.fillPoly(self.roi_mask, [self.roi_points], 255)
#
#         # Draw the ROI on the frame for visualization
#         roi_frame = frame.copy()
#         cv2.polylines(roi_frame, [self.roi_points], True, (0, 255, 0), 2)
#         cv2.putText(roi_frame, "ROI", (10, self.height - roi_height + 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#         # Display the frame with ROI
#         cv2.imshow('ROI Selection', roi_frame)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#     def apply_roi(self, frame):
#         if self.roi_mask is None:
#             raise ValueError("ROI has not been selected. Call select_roi first.")
#         return cv2.bitwise_and(frame, frame, mask=self.roi_mask)
#
#     def is_in_roi(self, point):
#         x, y = point
#         return cv2.pointPolygonTest(self.roi_points, (x, y), False) >= 0
#
#     def get_roi_mask(self):
#         return self.roi_mask


# #position_calculator.py:
# import numpy as np
#
# class PositionCalculator:
#     def __init__(self, camera_matrix):
#         self.camera_matrix = camera_matrix
#
#     def calculate_3d_position(self, x, y, depth):
#         # Convert image coordinates to camera coordinates
#         x_cam = (x - self.camera_matrix[0, 2]) * depth / self.camera_matrix[0, 0]
#         y_cam = (y - self.camera_matrix[1, 2]) * depth / self.camera_matrix[1, 1]
#         return np.array([x_cam, y_cam, depth])


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
            width_3d = width * depth / self.camera_matrix[0, 0]
            height_3d = height * depth / self.camera_matrix[1, 1]

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