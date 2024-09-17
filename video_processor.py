import cv2
import numpy as np
from depth_estimation import DepthEstimationModel
from camera_calibration import CameraCalibration
from car_detection import CarDetection
from projective_transformation import ProjectiveTransformation

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.depth_model = DepthEstimationModel()
        self.calibration = CameraCalibration(video_path)
        self.car_detection = CarDetection()

        self.cap = cv2.VideoCapture(video_path)
        self.ret, self.frame = self.cap.read()
        if not self.ret:
            raise ValueError("Failed to read the video file.")
        self.height, self.width = self.frame.shape[:2]
        print(f"Video dimensions: {self.width}x{self.height}")

        # Perform camera calibration
        self.camera_matrix, self.dist_coeffs = self.calibrate_camera()

        # Initialize Projective Transformation
        self.transformation = ProjectiveTransformation(self.camera_matrix, self.dist_coeffs)

        # Set up video writer for output
        self.output_path = 'Output/output_video.mp4'
        self.transformed_output_path = 'Output/transformed_output_video.mp4'
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.out = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))
        self.transformed_out = cv2.VideoWriter(self.transformed_output_path, self.fourcc, self.fps, (self.width, self.height))

        self.frame_count = 0
        self.focal_length = self.camera_matrix[0, 0]

    def calibrate_camera(self):
        print("Attempting camera calibration...")
        calibration_result = self.calibration.calibrate_camera()
        if calibration_result[0] is None:
            print("Camera calibration failed. Using default camera matrix.")
            camera_matrix = np.array([
                [1000, 0, self.width / 2],
                [0, 1000, self.height / 2],
                [0, 0, 1]
            ])
            dist_coeffs = np.zeros((5,1))  # Assuming 5 distortion coefficients
        else:
            print("Camera calibration successful.")
            camera_matrix, dist_coeffs = calibration_result

        print("Camera matrix:")
        print(camera_matrix)
        return camera_matrix, dist_coeffs

    def draw_3d_box(self, img, corners, color=(0, 0, 255)):
        # Draw 3D bounding box on the image
        for i in range(4):
            cv2.line(img, tuple(corners[i]), tuple(corners[(i + 1) % 4]), color, 2)
            cv2.line(img, tuple(corners[i + 4]), tuple(corners[(i + 1) % 4 + 4]), color, 2)
            cv2.line(img, tuple(corners[i]), tuple(corners[i + 4]), color, 2)
        return img

    def estimate_3d_dimensions(self, width_2d, height_2d, depth):
        # Estimate 3D dimensions of the vehicle
        width_3d = width_2d * depth / self.focal_length
        height_3d = height_2d * depth / self.focal_length
        length_3d = (width_3d + height_3d) / 2
        return width_3d, height_3d, length_3d

    def process_video(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert frame to RGB for depth estimation
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth_map = self.depth_model.estimate_depth(img)

            # Normalize and colorize depth map for visualization
            depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_map_normalized = np.uint8(depth_map_normalized)
            depth_map_color = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

            # Detect vehicles in the frame
            detections = self.car_detection.detect_cars(frame)
            vehicle_detections = detections[detections[:, 5] == 2]

            for det in vehicle_detections:
                x1, y1, x2, y2, conf, cls = det[:6]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Get depth value for the detected vehicle
                depth_roi = depth_map[int(y1):int(y2), int(x1):int(x2)]
                depth_value = np.mean(depth_roi)

                # Estimate 3D dimensions
                width_3d, height_3d, length_3d = self.estimate_3d_dimensions(x2 - x1, y2 - y1, depth_value)

                # Define 3D bounding box corners
                corners_3d = np.array([
                    [-width_3d / 2, -height_3d / 2, -length_3d / 2],
                    [width_3d / 2, -height_3d / 2, -length_3d / 2],
                    [width_3d / 2, -height_3d / 2, length_3d / 2],
                    [-width_3d / 2, -height_3d / 2, length_3d / 2],
                    [-width_3d / 2, height_3d / 2, -length_3d / 2],
                    [width_3d / 2, height_3d / 2, -length_3d / 2],
                    [width_3d / 2, height_3d / 2, length_3d / 2],
                    [-width_3d / 2, height_3d / 2, length_3d / 2]
                ])

                # Transform 3D corners to camera space
                corners_3d = corners_3d + np.array([(center_x - self.width / 2) * depth_value / self.focal_length,
                                                    (center_y - self.height / 2) * depth_value / self.focal_length,
                                                    depth_value])

                # Project 3D corners to 2D image plane
                corners_2d, _ = cv2.projectPoints(corners_3d, np.zeros(3), np.zeros(3), self.camera_matrix, self.dist_coeffs)
                corners_2d = corners_2d.reshape(-1, 2).astype(int)

                # Draw 3D bounding box in green
                frame = self.draw_3d_box(frame, corners_2d, color=(0, 255, 0))

                # Draw 2D bounding box and depth information
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f"Depth: {depth_value:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

            # Perform projective transformation
            points_2d = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], dtype=np.float32)
            transformed_frame = self.transformation.perform_transformation(frame, points_2d)

            # Blend original frame with depth map for visualization
            blended_frame = cv2.addWeighted(frame, 0.7, depth_map_color, 0.3, 0)

            # Write frames to output videos
            self.out.write(blended_frame)
            self.transformed_out.write(transformed_frame)

            # Display the processed frame
            cv2.imshow('Frame with Depth', blended_frame)
            cv2.imshow('Transformed Frame', transformed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.frame_count += 1

        # Clean up
        self.cap.release()
        self.out.release()
        self.transformed_out.release()
        cv2.destroyAllWindows()
        print(f"Processing complete. Output saved to {self.output_path}")
        print(f"Transformed output saved to {self.transformed_output_path}")