
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import random
from google.colab.patches import cv2_imshow
class CameraCalibration:
    def __init__(self, model_path='/content/drive/MyDrive/Speed_Detection_Project/VP_Model/vp_using_seg_model_best.keras'):
        """
        Initialize the CameraCalibration class.

        - model_path: Path to the pre-trained model used for vanishing point detection.
        - focal_length: The estimated distance between the camera and image plane.
        - principal_point: The image center where the camera's optical axis intersects.
        - vanishing_points: Stores the three computed vanishing points.
        """
        self.focal_length = None  # Focal length of the camera
        self.principal_point = None  # The center of the image (x, y)
        self.vanishing_points = None  # List of three vanishing points
        self.model_path = model_path  # Path to the neural network model
        self.model = None  # Placeholder for the loaded Keras model
        self.ipm_matrix = None  # Inverse Perspective Mapping (IPM) matrix for bird's-eye view
        self.width = None  # Image width (set when processing frames)
        self.height = None  # Image height (set when processing frames)

    def load_model(self):
        """
        Load the pre-trained Keras model for vanishing point detection.
        """
        try:
            if self.model is None:
                self.model = keras.models.load_model(self.model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise  # Re-raise the exception to halt execution

    def calibrate_camera(self, frames):
        """
        Perform camera calibration using multiple frames for vanishing point detection.

        - frames: A set of frames from the video that will be used to compute calibration parameters.
        """
        self.load_model()
        self.frames = frames
        self.height, self.width = frames[0].shape[:2]  # Get the frame dimensions
        self.principal_point = (self.width / 2, self.height / 2)  # Assume the principal point is at the image center

        # Robust Vanishing Point Estimation using RANSAC
        vps = []
        for frame in frames:  # Process all frames for robustness
            vp = self.find_vanishing_point(frame)
            if vp is not None:  # Add check for valid vanishing points
                vps.append(vp)

        if not vps:
            raise ValueError("No valid vanishing points detected.")

        # Apply RANSAC to find the best line fitting the vanishing points
        best_vp = self.ransac_line_fit(np.array(vps), threshold=10, iterations=100)

        # Find second vanishing point (from random frame for now)
        vp2 = self.find_vanishing_point(random.choice(frames))
        vp2 = self.orthogonalize_vanishing_points(best_vp, vp2)

        # Estimate focal length (using placeholder method for now)
        self.focal_length = self.estimate_focal_length(best_vp, vp2)

        # Calculate the third vanishing point (cross-product)
        vp3 = np.cross(best_vp, vp2)
        vp3 /= np.linalg.norm(vp3)

        # Store vanishing points
        self.vanishing_points = [best_vp, vp2, vp3]

        # Compute the Inverse Perspective Mapping (IPM) matrix
        self.compute_ipm_matrix()

        # Visualize the vanishing points in the frame
        self.visualize_vanishing_points(frames[0], [best_vp, vp2, vp3])

        return self.ipm_matrix  # Return the computed IPM matrix

    def ransac_line_fit(self, points, threshold, iterations):
        """
        Use RANSAC to robustly fit a line through noisy vanishing points.

        - points: Array of vanishing points.
        - threshold: Distance threshold for inlier points.
        - iterations: Number of iterations for RANSAC.
        """
        best_line = None
        best_inliers = 0
        for i in range(iterations):
            # Sample two random points
            sample = random.sample(range(len(points)), 2)
            p1, p2 = points[sample]

            # Compute the line defined by the cross-product of the two points
            line = np.cross(p1, p2)
            inliers = 0

            # Count inliers
            for p in points:
                dist = abs(np.dot(line, p)) / np.linalg.norm(line)  # Point-line distance
                if dist < threshold:
                    inliers += 1
            if inliers > best_inliers:
                best_inliers = inliers
                best_line = line

        # Return the best fit line (normalized)
        if best_line is not None:
            return best_line / np.linalg.norm(best_line)
        return None

    def orthogonalize_vanishing_points(self, vp1, vp2):
        """
        Ensure that the second vanishing point (vp2) is orthogonal to the first (vp1).
        """
        vp2_ortho = vp2 - np.dot(vp2, vp1) * vp1  # Project vp2 onto the plane orthogonal to vp1
        vp2_ortho /= np.linalg.norm(vp2_ortho)  # Normalize the vector
        return vp2_ortho

    def find_vanishing_point(self, frame):
        """
        Find the vanishing point using the neural network.

        - frame: Input frame for vanishing point detection.
        """
        self.load_model()  # Ensure the model is loaded
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format
        frame = cv2.resize(frame, (224, 224))  # Resize the frame for the model
        frame = frame.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension

        # Predict vanishing point using the model
        segmentation, vp = self.model.predict(frame)
        vp = vp[0]  # Extract the vanishing point from prediction

        # Convert normalized coordinates back to image space
        vp = np.array([vp[0] * self.width, vp[1] * self.height, 1])
        return vp

    def estimate_focal_length(self, vp1, vp2):
        """
        Estimate the camera's focal length using vanishing points.

        - vp1: First vanishing point.
        - vp2: Second vanishing point.
        """
        # Placeholder: Estimate focal length based on vp1 and vp2
        return np.sqrt(abs(np.dot(vp1, vp2)))

    def compute_ipm_matrix(self):
        """
        Compute the Inverse Perspective Mapping (IPM) matrix to transform the view into a bird's-eye view.
        """
        # Define source points (image corners)
        src_points = np.float32([
            [0, self.height],  # Bottom-left
            [self.width, self.height],  # Bottom-right
            [self.width, 0],  # Top-right
            [0, 0]  # Top-left
        ])

        # Define destination points for the bird's-eye view
        dst_points = np.float32([
            [0, self.height],  # Bottom-left
            [self.width, self.height],  # Bottom-right
            [self.width * 0.75, 0],  # Shifted inward for the top-right
            [self.width * 0.25, 0]  # Shifted inward for the top-left
        ])

        # Compute the IPM matrix using OpenCV
        self.ipm_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    def visualize_vanishing_points(self, frame, vanishing_points, output_path=None):
     """
     Visualize vanishing points on the frame, handling cases where VPs are outside the image bounds.

     - frame: The frame where vanishing points are to be visualized.
     - vanishing_points: List of vanishing points to display.
     - output_path: Path where the visualized frame should be saved (optional).
     """
     frame_with_vps = frame.copy()

     # Function to normalize vanishing points from homogeneous to Cartesian coordinates
     def normalize_vp(vp):
        if vp[2] != 0:
            return [vp[0] / vp[2], vp[1] / vp[2]]
        return [vp[0], vp[1]]  # In case the third component is 0 (shouldn't happen in most cases)

     # Normalize vanishing points to Cartesian coordinates
     normalized_vps = [normalize_vp(vp) for vp in vanishing_points]

     # Image boundaries
     img_h, img_w = frame.shape[:2]

     # Draw vanishing points and lines
     for idx, vp in enumerate(normalized_vps):
         x, y = int(vp[0]), int(vp[1])

         if 0 <= x < img_w and 0 <= y < img_h:
             # Draw a red circle for vanishing points inside the image bounds
             cv2.circle(frame_with_vps, (x, y), 10, (0, 0, 255), -1)  # Red circle
             cv2.putText(frame_with_vps, f'VP{idx+1}', (x + 15, y + 15), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
         else:
             # Handle vanishing points outside the image bounds
             if x < 0:
                 x = 0
             elif x >= img_w:
                 x = img_w - 1
             if y < 0:
                 y = 0
             elif y >= img_h:
                 y = img_h - 1
            
             #  Draw an arrow pointing toward the off-screen vanishing point
             cv2.arrowedLine(frame_with_vps, (img_w // 2, img_h), (x, y), (0, 255, 0), 2)

     # Draw green lines converging to vanishing points from the bottom of the image
     for vp in normalized_vps:
         x, y = int(vp[0]), int(vp[1])
        
         # Clip the vanishing point to the image bounds (if off-screen)
         if x < 0:
             x = 0
         elif x >= img_w:
             x = img_w - 1
         if y < 0:
             y = 0
         elif y >= img_h:
             y = img_h - 1

         # Draw lines converging to vanishing points from the bottom of the image
         cv2.line(frame_with_vps, (img_w // 4, img_h), (x, y), (0, 255, 0), 2)
         cv2.line(frame_with_vps, (img_w // 2, img_h), (x, y), (0, 255, 0), 2)
         cv2.line(frame_with_vps, (3 * img_w // 4, img_h), (x, y), (0, 255, 0), 2)

     # Save or display the frame
     if output_path:
         cv2.imwrite(output_path, frame_with_vps)
         print(f"Vanishing points visualization saved to {output_path}")
     else:
         # Display the frame with vanishing points in a window
         cv2_imshow(frame_with_vps)
         cv2.waitKey(0)
         cv2.destroyAllWindows()

