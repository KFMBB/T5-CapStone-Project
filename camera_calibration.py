import cv2
import numpy as np
from scipy.optimize import least_squares


import cv2
import numpy as np
from scipy.optimize import least_squares

class CameraCalibration:
    def __init__(self):
        self.focal_length = None
        self.principal_point = None
        self.vanishing_points = None

    def calibrate_camera(self, frames):
        self.frames = frames
        self.height, self.width = frames[0].shape[:2]
        self.principal_point = (self.width / 2, self.height / 2)

        # Find the first vanishing point (vehicle movement direction)
        vp1 = self.find_first_vanishing_point()

        # Find the second vanishing point (perpendicular to vehicle movement)
        vp2 = self.find_second_vanishing_point()

        # Ensure vanishing points are orthogonal
        vp2 = self.orthogonalize_vanishing_points(vp1, vp2)

        # Calculate focal length
        self.focal_length = np.sqrt(abs(np.dot(vp1, vp2)))

        # Calculate the third vanishing point
        vp3 = np.cross(vp1, vp2)
        vp3 /= np.linalg.norm(vp3)

        self.vanishing_points = [vp1, vp2, vp3]

        # Create camera matrix
        camera_matrix = np.array([
            [self.focal_length, 0, self.principal_point[0]],
            [0, self.focal_length, self.principal_point[1]],
            [0, 0, 1]
        ])

        return camera_matrix, np.zeros((5, 1))  # Assuming no distortion for simplicity

    def orthogonalize_vanishing_points(self, vp1, vp2):
        # Project vp2 onto the plane perpendicular to vp1
        vp2_ortho = vp2 - np.dot(vp2, vp1) * vp1
        vp2_ortho /= np.linalg.norm(vp2_ortho)
        return vp2_ortho

    def find_first_vanishing_point(self):
        # Use KLT tracker to find motion lines
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        old_frame = cv2.cvtColor(self.frames[0], cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_frame, mask=None, **feature_params)

        lines = []
        for frame in self.frames[1:]:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame_gray, p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                lines.append([c, d, a, b])

            old_frame = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        # Transform lines to diamond space
        diamond_space = self.transform_to_diamond_space(lines)

        # Find the peak in diamond space
        peak = np.unravel_index(np.argmax(diamond_space), diamond_space.shape)
        vp1 = self.diamond_to_image_space(peak, self.width, self.height)

        return vp1

    def find_second_vanishing_point(self):
        # Detect edges perpendicular to vehicle movement
        edges = []
        for frame in self.frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges_frame = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges_frame, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    if 80 < abs(angle) < 100:  # Approximately perpendicular to vehicle movement
                        edges.append([x1, y1, x2, y2])

        # Transform edges to diamond space
        diamond_space = self.transform_to_diamond_space(edges)

        # Find the peak in diamond space
        peak = np.unravel_index(np.argmax(diamond_space), diamond_space.shape)
        vp2 = self.diamond_to_image_space(peak, self.width, self.height)

        return vp2

    def transform_to_diamond_space(self, lines):
        diamond_space = np.zeros((self.width * 2, self.height * 2))
        for line in lines:
            x1, y1, x2, y2 = line
            if x2 != x1:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                u = int(m * self.width + self.width)
                v = int(b + self.height)
                if 0 <= u < diamond_space.shape[0] and 0 <= v < diamond_space.shape[1]:
                    diamond_space[u, v] += 1
        return diamond_space

    def diamond_to_image_space(self, peak, width, height):
        u, v = peak
        m = (u - width) / width
        b = v - height
        x = width / 2
        y = m * x + b
        return np.array([x, y, 1])

    def get_scale_factor(self, detections):
        # Calculate mean dimensions of detected vehicles
        mean_width = np.mean([d[2] - d[0] for d in detections])
        mean_height = np.mean([d[3] - d[1] for d in detections])

        # Compare with statistical data (this is a placeholder, replace with actual data)
        average_vehicle_width = 1.8  # meters
        scale_factor = average_vehicle_width / mean_width

        return scale_factor