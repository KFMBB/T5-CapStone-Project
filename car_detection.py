from ultralytics import YOLO
import numpy as np
import cv2

class CarDetection:
    def __init__(self):
        self.model = YOLO('models/yolov9m.pt')
        self.vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']

    def detect_cars(self, frame, ipm_matrix, calibrator, conf_threshold=0.5):
        try:
            # Undistort and rectify the image
            undistorted = calibrator.undistort_image(frame)
            rectified = calibrator.rectify_image(undistorted)

            # Apply IPM to the rectified frame
            ipm_frame = cv2.warpPerspective(rectified, ipm_matrix, (frame.shape[1], frame.shape[0]))

            # Perform object detection on the IPM frame
            results = self.model(ipm_frame)
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    if self.model.names[int(cls)] in self.vehicle_classes and conf > conf_threshold:
                        # Transform bounding box back to original image space
                        original_box = self.transform_bbox_to_original(
                            [x1, y1, x2, y2],
                            cv2.invert(ipm_matrix)[1]
                        )
                        # Calculate 3D position (assuming flat ground plane)
                        z = self.estimate_distance(original_box, calibrator.focal_length)
                        x = (original_box[0] + original_box[2]) / 2 - calibrator.principal_point[0]
                        y = (original_box[1] + original_box[3]) / 2 - calibrator.principal_point[1]
                        x *= z / calibrator.focal_length
                        y *= z / calibrator.focal_length
                        detections.append([x, y, z, *original_box, conf, cls])
            return np.array(detections)
        except Exception as e:
            print(f"Error in car detection: {str(e)}")
            return np.array([])

    def transform_bbox_to_original(self, bbox, inverse_ipm_matrix):
        def transform_point(x, y):
            p = np.dot(inverse_ipm_matrix, [x, y, 1])
            return p[:2] / p[2]

        x1, y1, x2, y2 = bbox
        p1 = transform_point(x1, y1)
        p2 = transform_point(x2, y2)
        return [*p1, *p2]

    def estimate_distance(self, bbox, focal_length):
        # Simplified distance estimation based on bounding box height
        # Assumes known average height of vehicles
        avg_vehicle_height = 1.5  # meters
        bbox_height = bbox[3] - bbox[1]
        return (avg_vehicle_height * focal_length) / bbox_height