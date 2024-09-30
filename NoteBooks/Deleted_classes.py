
# #projective_transformation.py:
# import cv2
# import numpy as np
#
#
# class ProjectiveTransformation:
#     def __init__(self, camera_calibration):
#         self.camera_calibration = camera_calibration
#         self.transformation_matrix = None
#         self.inverse_transformation_matrix = None
#
#     def construct_transformation(self, mask):
#         height, width = mask.shape[:2]
#
#         # Define source points (trapezoid)
#         src_points = np.float32([
#             [width * 0.1, height * 0.8],  # Bottom left
#             [width * 0.9, height * 0.8],  # Bottom right
#             [width * 0.6, height * 0.4],  # Top right
#             [width * 0.4, height * 0.4]  # Top left
#         ])
#
#         # Define destination points (rectangle)
#         dst_points = np.float32([
#             [width * 0.2, height],  # Bottom left
#             [width * 0.8, height],  # Bottom right
#             [width * 0.8, 0],  # Top right
#             [width * 0.2, 0]  # Top left
#         ])
#
#         # Compute the perspective transformation matrix
#         self.transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
#         self.inverse_transformation_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
#
#         # Visualize source points on original image
#         original_vis = self._visualize_points(mask, src_points)
#         cv2.imshow('Original with Source Points', original_vis)
#
#         # Apply transformation and visualize destination points
#         transformed_mask = self.apply_transformation(mask)
#         transformed_vis = self._visualize_points(transformed_mask, dst_points, color=(0, 0, 255))
#         cv2.imshow('Transformed with Destination Points', transformed_vis)
#
#         cv2.waitKey(1)
#
#         return original_vis, transformed_vis
#
#     def apply_transformation(self, image):
#         if self.transformation_matrix is None:
#             raise ValueError("Transformation matrix has not been constructed")
#         return cv2.warpPerspective(image, self.transformation_matrix, (image.shape[1], image.shape[0]))
#
#     def apply_inverse_transformation(self, points):
#         if self.inverse_transformation_matrix is None:
#             raise ValueError("Inverse transformation matrix has not been constructed")
#
#         # Reshape points to (N, 1, 2) format
#         points = points.reshape(-1, 1, 2)
#
#         # Apply perspective transformation
#         transformed_points = cv2.perspectiveTransform(points.astype(np.float32), self.inverse_transformation_matrix)
#
#         # Reshape back to original format
#         return transformed_points.reshape(-1, 2)
#
#     def _visualize_points(self, image, points, color=(0, 255, 0)):
#         vis_image = image.copy()
#         for i, point in enumerate(points):
#             x, y = int(point[0]), int(point[1])
#             cv2.circle(vis_image, (x, y), 5, color, -1)
#             cv2.putText(vis_image, str(i), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#         return vis_image



# # roi_selector.py
# import cv2
# import numpy as np
#
#
# class ROISelector:
#     def __init__(self, height, width):
#         self.height = height
#         self.width = width
#         self.roi_mask = None
#         self.roi_points = None
#
#     def select_roi(self, frame):
#         roi_height = int(self.height * 0.6)  # 60% of the frame height
#         top_left = (0, self.height - roi_height)
#         bottom_right = (self.width, self.height)
#
#         self.roi_points = np.array([
#             top_left,
#             (self.width, self.height - roi_height),
#             bottom_right,
#             (0, self.height)
#         ], dtype=np.int32)
#
#         # Create the ROI mask
#         self.roi_mask = np.zeros((self.height, self.width), dtype=np.uint8)
#         cv2.fillPoly(self.roi_mask, [self.roi_points], 255)
#
#         # Draw the ROI on the frame for visualization
#         roi_frame = frame.copy()
#         cv2.polylines(roi_frame, [self.roi_points], True, (0, 255, 0), 2)
#         cv2.putText(roi_frame, "ROI", (10, self.height - roi_height + 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#         # Display the frame with ROI
#         cv2.imshow('ROI Selection', roi_frame)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#
#     def apply_roi(self, frame):
#         if self.roi_mask is None:
#             raise ValueError("ROI has not been selected. Call select_roi first.")
#         return cv2.bitwise_and(frame, frame, mask=self.roi_mask)
#
#     def is_in_roi(self, point):
#         x, y = point
#         return cv2.pointPolygonTest(self.roi_points, (x, y), False) >= 0
#
#     def get_roi_mask(self):
#         return self.roi_mask


# #position_calculator.py:
# import numpy as np
#
# class PositionCalculator:
#     def __init__(self, camera_matrix):
#         self.camera_matrix = camera_matrix
#
#     def calculate_3d_position(self, x, y, depth):
#         # Convert image coordinates to camera coordinates
#         x_cam = (x - self.camera_matrix[0, 2]) * depth / self.camera_matrix[0, 0]
#         y_cam = (y - self.camera_matrix[1, 2]) * depth / self.camera_matrix[1, 1]
#         return np.array([x_cam, y_cam, depth])
