import cv2
import numpy as np

class CameraCalibration:
    def __init__(self):
        pass

    def calibrate_camera(self, frames):
        # Prepare object points
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all frames
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        # Calibrate camera
        if len(objpoints) > 0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            return mtx, dist
        else:
            print("Not enough corner detections for calibration. Using default values.")
            return self.get_default_camera_matrix(frame.shape), np.zeros((5,1))

    def get_default_camera_matrix(self, frame_shape):
        height, width = frame_shape[:2]
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        return camera_matrix