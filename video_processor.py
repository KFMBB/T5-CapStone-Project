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
        self.projective_transform = None

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

            if self.frame_count % 30 == 0 and len(calibration_frames) < 10:
                calibration_frames.append(masked_frame)

            if len(calibration_frames) == 10 and self.camera_matrix is None:
                self.camera_matrix, self.dist_coeffs = self.calibration.calibrate_camera(calibration_frames)
                self.projective_transform = ProjectiveTransformation(self.calibration)

                binary_mask = (self.depth_masker.get_mask() > 0).astype(np.uint8) * 255
                print(f"Binary mask shape: {binary_mask.shape}")
                print(f"Binary mask non-zero elements: {np.count_nonzero(binary_mask)}")

                original_vis, transformed_vis = self.projective_transform.construct_transformation(binary_mask)

                # Apply transformation to the actual frame
                transformed_frame = self.projective_transform.apply_transformation(masked_frame)

                # Display all visualizations
                cv2.imshow('Original with Source Points', original_vis)
                cv2.imshow('Transformed with Destination Points', transformed_vis)
                cv2.imshow('Transformed Frame', transformed_frame)
                cv2.waitKey(1)

                self.scale_factor = self.calibration.get_scale_factor(all_detections)
                self.position_calculator = PositionCalculator(self.camera_matrix)
                print("Camera calibration completed.")
                print("Camera matrix:", self.camera_matrix)
                print("Scale factor:", self.scale_factor)

            if self.projective_transform is not None:
                transformed_frame = self.projective_transform.apply_transformation(masked_frame)
            else:
                transformed_frame = masked_frame

            depth_map = self.depth_model.estimate_depth(transformed_frame)
            detections = self.car_detection.detect_cars(transformed_frame)

            # Filter detections based on ROI
            roi_detections = [det for det in detections if self.roi_selector.is_in_roi((det[0], det[1]))]
            all_detections.extend(roi_detections)

            tracks = self.tracker.update(roi_detections)

            display_frame = masked_frame.copy()

            if self.camera_matrix is not None:
                for track_id, track in tracks.items():
                    if track['hits'] >= self.tracker.min_hits and track['missed_frames'] == 0:
                        x1, y1, x2, y2, conf, cls = map(float, track['bbox'])
                        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                        # Transform bounding box back to original frame
                        if self.projective_transform is not None:
                            [[x1, y1], [x2, y2]] = self.projective_transform.apply_inverse_transformation(np.array([[x1, y1], [x2, y2]])).astype(int)
                            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

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