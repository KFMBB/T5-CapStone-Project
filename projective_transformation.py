import cv2
import numpy as np


class ProjectiveTransformation:
    def __init__(self):
        pass

    def back_project(self, points_2d, camera_matrix, dist_coeffs, depth):
        # Undistort points
        points_2d_undistorted = cv2.undistortPoints(points_2d, camera_matrix, dist_coeffs)

        # Get focal length and principal point
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        # Back-project to 3D
        points_3d = []
        for point in points_2d_undistorted:
            x, y = point[0]
            X = (x - cx) * depth / fx
            Y = (y - cy) * depth / fy
            Z = depth
            points_3d.append([X, Y, Z])

        return np.array(points_3d)