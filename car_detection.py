from ultralytics import YOLO
import cv2
import numpy as np


class CarDetection:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')  # Load the YOLOv8 model
        self.vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']

    def detect_cars(self, frame):
        results = self.model(frame)  # Perform detection

        # Extract detections
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # get box coordinates
                conf = box.conf[0].cpu().numpy()  # get confidence
                cls = box.cls[0].cpu().numpy()  # get class

                # Check if the detected class is a vehicle
                if self.model.names[int(cls)] in self.vehicle_classes:
                    detections.append([x1, y1, x2, y2, conf, 2])  # Use 2 as a generic vehicle class

        return np.array(detections)