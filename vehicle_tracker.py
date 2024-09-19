import numpy as np


class VehicleTracker:
    def __init__(self, max_frames_to_skip=10, min_hits=3):
        self.tracks = {}
        self.frame_count = 0
        self.max_frames_to_skip = max_frames_to_skip
        self.min_hits = min_hits

    def update(self, detections):
        self.frame_count += 1

        # Update existing tracks
        for track_id in list(self.tracks.keys()):
            if self.frame_count - self.tracks[track_id]['last_seen'] > self.max_frames_to_skip:
                del self.tracks[track_id]
            else:
                self.tracks[track_id]['missed_frames'] += 1

        # Match detections to tracks
        for detection in detections:
            matched = False
            for track_id, track in self.tracks.items():
                if self.iou(detection, track['bbox']) > 0.3:  # IOU threshold
                    self.tracks[track_id]['bbox'] = detection
                    self.tracks[track_id]['last_seen'] = self.frame_count
                    self.tracks[track_id]['missed_frames'] = 0
                    self.tracks[track_id]['hits'] += 1
                    matched = True
                    break

            if not matched:
                self.tracks[len(self.tracks)] = {
                    'bbox': detection,
                    'last_seen': self.frame_count,
                    'missed_frames': 0,
                    'hits': 1
                }

        return self.tracks

    def iou(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1[:4]
        x3, y3, x4, y4 = bbox2[:4]

        xi1, yi1, xi2, yi2 = max(x1, x3), max(y1, y3), min(x2, x4), min(y2, y4)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersect_area = (xi2 - xi1) * (yi2 - yi1)
        bbox1_area = (x2 - x1) * (y2 - y1)
        bbox2_area = (x4 - x3) * (y4 - y3)

        iou = intersect_area / float(bbox1_area + bbox2_area - intersect_area)
        return iou