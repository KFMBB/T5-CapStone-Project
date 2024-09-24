import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

class CameraCalibration:
    def __init__(self, model_path='models/vanishing_point_detection_model.h5'):
        self.focal_length = None
        self.principal_point = None
        self.vanishing_points = None
        self.model = keras.models.load_model(model_path)

    def calibrate_camera(self, frames):
        self.frames = frames
        self.height, self.width = frames[0].shape[:2]
        self.principal_point = (self.width / 2, self.height / 2)

        # Find the first vanishing point (vehicle movement direction)
        vp1 = self.find_vanishing_point(frames[0])

        # Find the second vanishing point (perpendicular to vehicle movement)
        vp2 = self.find_vanishing_point(frames[-1])  # Using the last frame for diversity

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

    def find_vanishing_point(self, frame):
        # Preprocess the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype('float32') / 255.0
        frame = np.expand_dims(frame, axis=0)

        # Predict vanishing point
        vp = self.model.predict(frame)[0]

        # Convert normalized coordinates back to image space
        vp = np.array([vp[0] * self.width, vp[1] * self.height, 1])
        return vp

    def get_scale_factor(self, detections):
        # Calculate mean dimensions of detected vehicles
        mean_width = np.mean([d[2] - d[0] for d in detections])
        mean_height = np.mean([d[3] - d[1] for d in detections])

        # Compare with statistical data (this is a placeholder, replace with actual data)
        average_vehicle_width = 1.8  # meters
        scale_factor = average_vehicle_width / mean_width

        return scale_factor