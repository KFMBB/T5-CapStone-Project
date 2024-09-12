import torch
import cv2
import numpy as np

class DepthEstimationModel:
    def __init__(self, model_type="MiDaS_small"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device)
        self.model.eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    def estimate_depth(self, img):
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        return cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
