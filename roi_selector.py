import cv2
import numpy as np


class ROISelector:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.roi_mask = None
        self.roi_points = None

    def select_roi(self, frame):
        roi_height = int(self.height * 0.6)  # 60% of the frame height
        top_left = (0, self.height - roi_height)
        bottom_right = (self.width, self.height)

        self.roi_points = np.array([
            top_left,
            (self.width, self.height - roi_height),
            bottom_right,
            (0, self.height)
        ], dtype=np.int32)

        # Create the ROI mask
        self.roi_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(self.roi_mask, [self.roi_points], 255)

        # Draw the ROI on the frame for visualization
        roi_frame = frame.copy()
        cv2.polylines(roi_frame, [self.roi_points], True, (0, 255, 0), 2)
        cv2.putText(roi_frame, "ROI", (10, self.height - roi_height + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with ROI
        cv2.imshow('ROI Selection', roi_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def apply_roi(self, frame):
        if self.roi_mask is None:
            raise ValueError("ROI has not been selected. Call select_roi first.")
        return cv2.bitwise_and(frame, frame, mask=self.roi_mask)

    def is_in_roi(self, point):
        x, y = point
        return cv2.pointPolygonTest(self.roi_points, (x, y), False) >= 0

    def get_roi_mask(self):
        return self.roi_mask