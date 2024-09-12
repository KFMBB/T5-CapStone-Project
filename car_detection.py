import torch

class CarDetection:
    def __init__(self):
        self.model = torch.hub.load('models/yolov5s.pt', 'yolov5s')
        self.model.eval()

    def detect_cars(self, frame):
        results = self.model(frame)
        detections = results.xyxy[0]
        return detections
