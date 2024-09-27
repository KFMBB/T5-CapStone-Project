import torch
import cv2
import numpy as np
from transformers import DPTForDepthEstimation, DPTImageProcessor

class DepthEstimationModel:
    def __init__(self, model_type="Intel/dpt-large"):
        # Initialize the DPT depth estimation model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DPTForDepthEstimation.from_pretrained(model_type)
        self.model.to(self.device).eval()
        self.processor = DPTImageProcessor.from_pretrained(model_type)

    def estimate_depth(self, img):
        # Preprocess the image
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        # Perform depth estimation
        with torch.no_grad():
            outputs = self.model(**inputs)
            prediction = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        # Normalize the depth map
        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX)

        # Apply additional smoothing
        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)

        return depth_map