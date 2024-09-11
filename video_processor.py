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

                # Project 3D point back to 2D for visualization
                rvec = np.zeros(3, dtype=np.float32)
                tvec = np.zeros(3, dtype=np.float32)
                point_2d_proj, _ = cv2.projectPoints(point_3d.reshape(1, 1, 3), rvec, tvec, self.camera_matrix, self.dist_coeffs)
                x_proj, y_proj = point_2d_proj[0][0]

                # Draw original bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw projected point on the 3D canvas
                cv2.circle(canvas_3d, (int(x_proj), int(y_proj)), 5, (0, 0, 255), -1)

                # Add depth information
                cv2.putText(frame, f"Depth: {depth_value:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Combine original frame and 3D canvas
            combined_frame = cv2.addWeighted(frame, 0.7, canvas_3d, 0.3, 0)

            # Write the frame to the output video
            self.out.write(combined_frame)

            # Display frame count every 30 frames (optional for logging purposes)
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                print(f"Processed {self.frame_count} frames")

        # Release resources
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        print("Video processing complete. Output saved to", self.output_path)
