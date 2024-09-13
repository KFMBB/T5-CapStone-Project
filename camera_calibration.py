import cv2
import numpy as np
from absl.testing.parameterized import parameters

print(cv2.__version__)

class CameraCalibration:
    def __init__(self, video_path):
        self.video_path = video_path

    def calibrate_camera_with_aruco(self):
        # Use getPredefinedDictionary instead of Dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

        # Using DetectorParameters_create to ensure compatibility
        aruco_params = cv2.aruco.DetectorParameters()


        cap = cv2.VideoCapture(self.video_path)
        all_corners = []
        all_ids = []
        obj_points = []

        marker_length = 0.05  # Marker length in meters
        obj_point = np.zeros((4, 3), np.float32)
        obj_point[:, :2] = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32) * marker_length

        frame_count = 0
        gray = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None:
                for i in range(len(ids)):
                    all_corners.append(corners[i])
                    all_ids.append(ids[i])
                    obj_points.append(obj_point)

            frame_count += 1
            if frame_count >= 600:  # Process 600 frames
                break

        cap.release()

        if gray is None:
            print("Error: Could not obtain calibration frames.")
            return None, None

        # Camera calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, all_corners, gray.shape[::-1], None, None
        )

        if ret:
            print("Camera calibration successful!")
            return camera_matrix, dist_coeffs
        else:
            print("Camera calibration failed.")
            return None, None

