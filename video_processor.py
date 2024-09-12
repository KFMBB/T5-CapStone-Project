import cv2
import numpy as np
from depth_estimation import DepthEstimationModel
from camera_calibration import CameraCalibration
from car_detection import CarDetection

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.depth_model = DepthEstimationModel()
        self.calibration = CameraCalibration(video_path)
        self.car_detection = CarDetection()
        
        # Perform camera calibration
        self.camera_matrix, self.dist_coeffs = self.calibration.calibrate_camera_with_aruco()
        
        if self.camera_matrix is None or self.dist_coeffs is None:
            print("Camera calibration failed. Exiting.")
            exit()
        
        self.cap = cv2.VideoCapture(video_path)
        self.ret, self.frame = self.cap.read()
        self.height, self.width = self.frame.shape[:2]
        print(f"Video dimensions: {self.width}x{self.height}")

        # Define the codec and create VideoWriter object
        self.output_path = 'output_video.mp4'
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
        self.out = cv2.VideoWriter(self.output_path, self.fourcc, 20.0, (self.width, self.height))

        # Initialize frame count
        self.frame_count = 0

    def process_video(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                break

            # Estimate depth
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth_map = self.depth_model.estimate_depth(img)

            # YOLOv5 car detection
            detections = self.car_detection.detect_cars(frame)

            # Filter out cars (YOLOv5 class 'car' is class 2)
            car_detections = detections[detections[:, 5] == 2]

            # Create a blank canvas for 3D visualization
            canvas_3d = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Loop over detected cars
            for det in car_detections:
                x1, y1, x2, y2, conf, cls = map(int, det[:6])

                # Calculate the center of the car
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Get the depth value at the center of the car
                depth_value = depth_map[center_y, center_x]

                # Convert 2D point to 3D using depth information
                point_2d = np.array([[center_x, center_y]], dtype=np.float32)
                point_3d = cv2.undistortPoints(point_2d, self.camera_matrix, self.dist_coeffs)
                point_3d = point_3d[0][0]
                point_3d = np.array([point_3d[0], point_3d[1], 1.0]) * depth_value

                # Convert camera matrix to 4x4 for homogeneous coordinates
                camera_matrix_homogeneous = np.hstack((self.camera_matrix, np.zeros((3, 1))))
                camera_matrix_homogeneous = np.vstack((camera_matrix_homogeneous, [0, 0, 0, 1]))

                # Project 3D point back to 2D for visualization
                projected_point_2d = np.dot(camera_matrix_homogeneous, np.array([point_3d[0], point_3d[1], point_3d[2], 1.0]))
                projected_point_2d /= projected_point_2d[2]
                projected_point_2d = tuple(map(int, projected_point_2d[:2]))

                # Draw detection and depth information on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Depth: {depth_value:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, projected_point_2d, 5, (255, 0, 0), -1)

                # Draw on 3D canvas
                cv2.circle(canvas_3d, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

            # Write frame to output video
            self.out.write(frame)

            # Show frame
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.frame_count += 1

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        print(f"Processing complete. Output saved to {self.output_path}")