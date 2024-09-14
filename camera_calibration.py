import cv2
import numpy as np
from absl.testing.parameterized import parameters
from scipy.optimize import minimize
from car_detection import CarDetection

print(cv2.__version__)

class CameraCalibration:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.principal_point = (self.frame_width // 2, self.frame_height // 2)

    def find_vanishing_point(self, lines, diamond_size=1000):
        # Find vanishing point using the diamond space method
        if not lines:
            print("No lines provided to find_vanishing_point")
            return None

        diamond_space = np.zeros((diamond_size, diamond_size), dtype=np.uint8)

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1

                u = int((m + 1) / 2 * diamond_size)
                v = int((b + 1) / 2 * diamond_size)

                cv2.line(diamond_space, (u, 0), (u, diamond_size - 1), 255, 1)
                cv2.line(diamond_space, (0, v), (diamond_size - 1, v), 255, 1)

        accumulator = cv2.HoughLines(diamond_space, 1, np.pi / 180, 100)

        if accumulator is None:
            print("No vanishing point found in diamond space")
            return None

        vp_rho, vp_theta = accumulator[0][0]

        x = int(vp_rho * np.cos(vp_theta))
        y = int(vp_rho * np.sin(vp_theta))

        return (x, y)

    def calibrate_camera(self):
        car_detection = CarDetection()

        # Find first vanishing point (movement direction)
        movement_lines = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            detections = car_detection.detect_cars(frame)
            for det in detections:
                x1, y1, x2, y2 = det[:4]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                movement_lines.append([[center_x, center_y, center_x + 10, center_y]])  # Assume horizontal movement

        vp1 = self.find_vanishing_point(movement_lines)
        if vp1 is None:
            print("Failed to find first vanishing point")
            return None, None

        # Find second vanishing point (perpendicular to movement)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        perpendicular_lines = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    if abs(angle) > 80 and abs(angle) < 100:
                        perpendicular_lines.append(line)

        vp2 = self.find_vanishing_point(perpendicular_lines)
        if vp2 is None:
            print("Failed to find second vanishing point")
            return None, None

        # Calculate focal length
        def focal_length_error(f):
            K = np.array([[f, 0, self.principal_point[0]],
                          [0, f, self.principal_point[1]],
                          [0, 0, 1]])
            K_inv = np.linalg.inv(K)

            v1 = np.array([vp1[0], vp1[1], 1])
            v2 = np.array([vp2[0], vp2[1], 1])

            v1_normalized = K_inv.dot(v1)
            v2_normalized = K_inv.dot(v2)

            return abs(np.dot(v1_normalized, v2_normalized))

        result = minimize(focal_length_error, 1000, method='nelder-mead')
        focal_length = result.x[0]

        # Calculate third vanishing point
        v1 = np.array([vp1[0], vp1[1], 1])
        v2 = np.array([vp2[0], vp2[1], 1])
        v3 = np.cross(v1, v2)
        v3 /= v3[2]
        vp3 = (int(v3[0]), int(v3[1]))

        # Construct camera matrix
        camera_matrix = np.array([
            [focal_length, 0, self.principal_point[0]],
            [0, focal_length, self.principal_point[1]],
            [0, 0, 1]
        ])

        return camera_matrix, None  # Return None for distortion coefficients as they're not calculated in this method


