from ultralytics import YOLO
import cv2
import numpy as np


class CarDetection:
    def __init__(self):
        # Initialize YOLO model for object detection
        self.model = YOLO('models/yolov8n.pt')
        # Define vehicle classes we're interested in
        self.vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']

    def detect_cars(self, frame):
        # Perform object detection on the frame
        results = self.model(frame)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract bounding box coordinates, confidence, and class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                # If the detected object is a vehicle, add it to detections
                if self.model.names[int(cls)] in self.vehicle_classes:
                    detections.append([x1, y1, x2, y2, conf, 2])
        return np.array(detections)
