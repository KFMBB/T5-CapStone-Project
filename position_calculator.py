import numpy as np

class PositionCalculator:
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix

    def calculate_3d_position(self, x, y, depth):
        # Convert image coordinates to camera coordinates
        x_cam = (x - self.camera_matrix[0, 2]) * depth / self.camera_matrix[0, 0]
        y_cam = (y - self.camera_matrix[1, 2]) * depth / self.camera_matrix[1, 1]
        return np.array([x_cam, y_cam, depth])