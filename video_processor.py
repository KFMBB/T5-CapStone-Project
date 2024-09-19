import cv2
import numpy as np
from depth_estimation import DepthEstimationModel
from camera_calibration import CameraCalibration
from car_detection import CarDetection
from projective_transformation import ProjectiveTransformation
from vehicle_tracker import VehicleTracker
from depth_masker import DepthMasker


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

        # Initialize depth masker
        self.depth_masker = DepthMasker(self.height, self.width)

        # Initialize camera calibration
        self.calibration = CameraCalibration()
        self.camera_matrix, self.dist_coeffs = None, None

        # Initialize Projective Transformation
        self.transformation = ProjectiveTransformation()

        # Set up video writer for output
        self.output_path = 'Output/output_video.mp4'
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.out = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))

        self.frame_count = 0

    def process_video(self):
        # Manual road selection
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = self.cap.read()
        if not ret:
            raise ValueError("Failed to read the first frame.")
        self.depth_masker.manual_road_selection(first_frame)

        calibration_frames = []

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Apply road mask
            masked_frame = self.depth_masker.apply_mask(frame)

            # Estimate depth for masked frame
            depth_map = self.depth_model.estimate_depth(masked_frame)

            # Detect vehicles in the masked frame
            detections = self.car_detection.detect_cars(masked_frame)

            # Update tracks
            tracks = self.tracker.update(detections)

            # Collect frames for calibration
            if self.frame_count % 30 == 0 and len(calibration_frames) < 10:
                calibration_frames.append(masked_frame)

            # Perform camera calibration if we have enough frames
            if len(calibration_frames) == 10 and self.camera_matrix is None:
                self.camera_matrix, self.dist_coeffs = self.calibration.calibrate_camera(calibration_frames)
                print("Camera calibration completed.")
                print("Camera matrix:", self.camera_matrix)
                print("Distortion coefficients:", self.dist_coeffs)

            # Process tracked vehicles
            if self.camera_matrix is not None:
                for track_id, track in tracks.items():
                    if track['hits'] >= self.tracker.min_hits and track['missed_frames'] == 0:
                        x1, y1, x2, y2 = map(int, track['bbox'][:4])

                        # Calculate average depth for the vehicle
                        vehicle_depth = np.mean(depth_map[y1:y2, x1:x2])

                        # Draw 2D bounding box
                        cv2.rectangle(masked_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                        # Write depth on the bounding box
                        cv2.putText(masked_frame, f"Depth: {vehicle_depth:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                        # Calculate 3D bounding box dimensions
                        width_3d = (x2 - x1) * vehicle_depth / self.camera_matrix[0, 0]
                        height_3d = (y2 - y1) * vehicle_depth / self.camera_matrix[1, 1]
                        length_3d = (width_3d + height_3d) / 2  # Estimate length as average of width and height

                        # Define 3D bounding box corners
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        corners_3d = np.array([
                            [-width_3d / 2, -height_3d / 2, 0],
                            [width_3d / 2, -height_3d / 2, 0],
                            [width_3d / 2, -height_3d / 2, -length_3d],
                            [-width_3d / 2, -height_3d / 2, -length_3d],
                            [-width_3d / 2, height_3d / 2, 0],
                            [width_3d / 2, height_3d / 2, 0],
                            [width_3d / 2, height_3d / 2, -length_3d],
                            [-width_3d / 2, height_3d / 2, -length_3d]
                        ])

                        # Transform 3D points to camera coordinate system
                        corners_3d += np.array([(x_center - self.width / 2) * vehicle_depth / self.camera_matrix[0, 0],
                                                (y_center - self.height / 2) * vehicle_depth / self.camera_matrix[1, 1],
                                                vehicle_depth])

                        # Project 3D corners to 2D image plane
                        corners_2d, _ = cv2.projectPoints(corners_3d, np.zeros(3), np.zeros(3),
                                                          self.camera_matrix, self.dist_coeffs)
                        corners_2d = corners_2d.reshape(-1, 2).astype(int)

                        # Draw 3D bounding box in green
                        self.draw_3d_box(masked_frame, corners_2d, color=(0, 255, 0))

            # Write frame to output video
            self.out.write(masked_frame)

            # Display the processed frame
            cv2.imshow('Processed Frame', masked_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.frame_count += 1

        # Clean up
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        print(f"Processing complete. Output saved to {self.output_path}")

    def draw_3d_box(self, img, corners, color=(0, 255, 0)):
        # Draw 3D bounding box on the image
        def draw_line(start, end):
            cv2.line(img, tuple(start), tuple(end), color, 2)

        # Draw the front face
        for i in range(4):
            draw_line(corners[i], corners[(i + 1) % 4])

        # Draw the rear face
        for i in range(4):
            draw_line(corners[i + 4], corners[((i + 1) % 4) + 4])

        # Draw the lines connecting front and rear faces
        for i in range(4):
            draw_line(corners[i], corners[i + 4])

        return img