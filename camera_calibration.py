import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json

class CameraCalibration:
    def __init__(self, model_path='models/vp_using_seg_model_best.keras'):
        self.focal_length = None
        self.principal_point = None
        self.vanishing_points = None
        self.model_path = model_path
        self.model = None
        self.ipm_matrix = None
        self.width = None
        self.height = None

    def load_model(self):
        if self.model is None:
            self.model = keras.models.load_model(self.model_path)

    def calibrate_camera(self, frames):
        self.load_model()  # Ensure model is loaded
        self.frames = frames
        self.height, self.width = frames[0].shape[:2]
        self.principal_point = (self.width / 2, self.height / 2)

        # Find the vanishing points
        vp1 = self.find_vanishing_point(frames[0])
        vp2 = self.find_vanishing_point(frames[-1])
        vp2 = self.orthogonalize_vanishing_points(vp1, vp2)

        self.focal_length = np.sqrt(abs(np.dot(vp1, vp2)))
        vp3 = np.cross(vp1, vp2)
        vp3 /= np.linalg.norm(vp3)

        self.vanishing_points = [vp1, vp2, vp3]

        # Compute IPM matrix
        self.compute_ipm_matrix()

        # Visualize vanishing points
        self.visualize_vanishing_points(frames[0], [vp1, vp2, vp3])

        return self.ipm_matrix

    def orthogonalize_vanishing_points(self, vp1, vp2):
        # Project vp2 onto the plane perpendicular to vp1
        vp2_ortho = vp2 - np.dot(vp2, vp1) * vp1
        vp2_ortho /= np.linalg.norm(vp2_ortho)
        return vp2_ortho

    def find_vanishing_point(self, frame):
        self.load_model()  # Ensure model is loaded
        # Preprocess the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype('float32') / 255.0
        frame = np.expand_dims(frame, axis=0)

        # Predict vanishing point
        segmentation, vp = self.model.predict(frame)
        vp = vp[0]  # Get the first (and only) prediction

        # Convert normalized coordinates back to image space
        vp = np.array([vp[0] * self.width, vp[1] * self.height, 1])
        return vp

    def compute_ipm_matrix(self):
        # Define source points (in image plane)
        src_points = np.float32([
            [0, self.height],
            [self.width, self.height],
            [self.width, 0],
            [0, 0]
        ])

        # Define destination points (in bird's eye view)
        dst_points = np.float32([
            [0, self.height],
            [self.width, self.height],
            [self.width * 0.75, 0],
            [self.width * 0.25, 0]
        ])

        # Compute the IPM matrix
        self.ipm_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    def get_camera_matrix(self):
        if self.focal_length is None or self.principal_point is None:
            raise ValueError("Camera not calibrated. Call calibrate_camera first.")
        return np.array([
            [self.focal_length, 0, self.principal_point[0]],
            [0, self.focal_length, self.principal_point[1]],
            [0, 0, 1]
        ])

    def save_calibration(self, filename):
        data = {
            'focal_length': self.focal_length.tolist() if isinstance(self.focal_length, np.ndarray) else self.focal_length,
            'principal_point': self.principal_point.tolist() if isinstance(self.principal_point, np.ndarray) else self.principal_point,
            'vanishing_points': [vp.tolist() for vp in self.vanishing_points],
            'ipm_matrix': self.ipm_matrix.tolist(),
            'width': self.width,
            'height': self.height
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load_calibration(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        self.focal_length = np.array(data['focal_length'])
        self.principal_point = np.array(data['principal_point'])
        self.vanishing_points = [np.array(vp) for vp in data['vanishing_points']]
        self.ipm_matrix = np.array(data['ipm_matrix'])
        self.width = data['width']
        self.height = data['height']

    def apply_ipm(self, frame):
        if self.ipm_matrix is None:
            raise ValueError("IPM matrix has not been computed. Call calibrate_camera first.")
        if self.width is None or self.height is None:
            self.height, self.width = frame.shape[:2]
        return cv2.warpPerspective(frame, self.ipm_matrix, (self.width, self.height))

    def visualize_vanishing_points(self, frame, vanishing_points):
        """
        Visualize vanishing points on the frame
        """
        frame_with_vps = frame.copy()

        # Draw vanishing points on the image
        for idx, vp in enumerate(vanishing_points):
            x, y = int(vp[0]), int(vp[1])
            cv2.circle(frame_with_vps, (x, y), 10, (0, 0, 255), -1)  # Red circles for vanishing points
            cv2.putText(frame_with_vps, f'VP{idx+1}', (x + 15, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Optionally: Show lines converging towards the vanishing point (e.g., horizon or lane markings)
        for vp in vanishing_points:
            # Draw lines converging to vanishing points from different positions in the image
            cv2.line(frame_with_vps, (self.width // 4, self.height), (int(vp[0]), int(vp[1])), (0, 255, 0), 2)
            cv2.line(frame_with_vps, (self.width // 2, self.height), (int(vp[0]), int(vp[1])), (0, 255, 0), 2)
            cv2.line(frame_with_vps, (3 * self.width // 4, self.height), (int(vp[0]), int(vp[1])), (0, 255, 0), 2)

        # Display the frame with vanishing points
        cv2.imshow('Vanishing Points Visualization', frame_with_vps)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
