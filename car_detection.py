import torch

class CarDetection:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.eval()

    def detect_cars(self, frame):
        results = self.model(frame)
        detections = results.xyxy[0]
        return detections
