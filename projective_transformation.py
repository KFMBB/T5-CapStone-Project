import cv2
import numpy as np


class ProjectiveTransformation:
    def __init__(self, camera_calibration):
        self.camera_calibration = camera_calibration
        self.transformation_matrix = None
        self.inverse_transformation_matrix = None

    def construct_transformation(self, mask):
        height, width = mask.shape[:2]

        # Define source points (trapezoid)
        src_points = np.float32([
            [width * 0.1, height * 0.8],  # Bottom left
            [width * 0.9, height * 0.8],  # Bottom right
            [width * 0.6, height * 0.4],  # Top right
            [width * 0.4, height * 0.4]  # Top left
        ])

        # Define destination points (rectangle)
        dst_points = np.float32([
            [width * 0.2, height],  # Bottom left
            [width * 0.8, height],  # Bottom right
            [width * 0.8, 0],  # Top right
            [width * 0.2, 0]  # Top left
        ])

        # Compute the perspective transformation matrix
        self.transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inverse_transformation_matrix = cv2.getPerspectiveTransform(dst_points, src_points)

        # Visualize source points on original image
        original_vis = self._visualize_points(mask, src_points)
        cv2.imshow('Original with Source Points', original_vis)

        # Apply transformation and visualize destination points
        transformed_mask = self.apply_transformation(mask)
        transformed_vis = self._visualize_points(transformed_mask, dst_points, color=(0, 0, 255))
        cv2.imshow('Transformed with Destination Points', transformed_vis)

        cv2.waitKey(1)

        return original_vis, transformed_vis

    def apply_transformation(self, image):
        if self.transformation_matrix is None:
            raise ValueError("Transformation matrix has not been constructed")
        return cv2.warpPerspective(image, self.transformation_matrix, (image.shape[1], image.shape[0]))

    def apply_inverse_transformation(self, points):
        if self.inverse_transformation_matrix is None:
            raise ValueError("Inverse transformation matrix has not been constructed")

        # Reshape points to (N, 1, 2) format
        points = points.reshape(-1, 1, 2)

        # Apply perspective transformation
        transformed_points = cv2.perspectiveTransform(points.astype(np.float32), self.inverse_transformation_matrix)

        # Reshape back to original format
        return transformed_points.reshape(-1, 2)

    def _visualize_points(self, image, points, color=(0, 255, 0)):
        vis_image = image.copy()
        for i, point in enumerate(points):
            x, y = int(point[0]), int(point[1])
            cv2.circle(vis_image, (x, y), 5, color, -1)
            cv2.putText(vis_image, str(i), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return vis_image