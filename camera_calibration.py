import cv2
import numpy as np

class CameraCalibration:
    def __init__(self, video_path):
        self.video_path = video_path

    def calibrate_camera_with_aruco(self):
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        cap = cv2.VideoCapture(self.video_path)
        all_corners = []
        all_ids = []
        obj_points = []

        marker_length = 0.05
        obj_point = np.array([[0, 0, 0],
                              [marker_length, 0, 0],
                              [marker_length, marker_length, 0],
                              [0, marker_length, 0]], dtype=np.float32)

        frame_count = 0
        gray = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None:
                for i in range(len(ids)):
                    all_corners.append(corners[i])
                    all_ids.append(ids[i])
                    obj_points.append(obj_point)

            frame_count += 1
            if frame_count >= 600:
                break

        cap.release()

        if gray is None:
            print("Error: Could not obtain calibration frames.")
            return None, None

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, all_corners, gray.shape[::-1], None, None
        )

        if ret:
            print("Camera calibration successful!")
            return camera_matrix, dist_coeffs
        else:
            print("Camera calibration failed.")
            return None, None
