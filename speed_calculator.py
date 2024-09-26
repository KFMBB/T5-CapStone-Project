import numpy as np
from scipy.signal import savgol_filter


class SpeedCalculator:
    def __init__(self, smoothing_window=5, speed_confidence_threshold=0.8):
        self.previous_positions = {}
        self.speed_history = {}
        self.smoothing_window = smoothing_window
        self.speed_confidence_threshold = speed_confidence_threshold

    def calculate_speed(self, track_id, current_position, current_time, previous_time):
        if track_id in self.previous_positions:
            previous_position = self.previous_positions[track_id]
            time_diff = current_time - previous_time

            if time_diff > 0:
                displacement = np.linalg.norm(current_position - previous_position)
                speed = displacement / time_diff

                if track_id not in self.speed_history:
                    self.speed_history[track_id] = []
                self.speed_history[track_id].append(speed)

                # Apply smoothing if we have enough speed measurements
                if len(self.speed_history[track_id]) >= self.smoothing_window:
                    smoothed_speed = savgol_filter(self.speed_history[track_id], self.smoothing_window, 2)[-1]
                    speed_confidence = self.calculate_speed_confidence(self.speed_history[track_id])
                else:
                    smoothed_speed = speed
                    speed_confidence = 0.5  # Lower confidence for initial measurements

                self.previous_positions[track_id] = current_position
                return smoothed_speed, speed_confidence
        else:
            self.previous_positions[track_id] = current_position

        return None, 0.0

    def calculate_speed_confidence(self, speed_history):
        if len(speed_history) < 2:
            return 0.5

        # Calculate the coefficient of variation
        cv = np.std(speed_history) / np.mean(speed_history)

        # Convert CV to a confidence score (lower CV means higher confidence)
        confidence = 1 / (1 + cv)

        # Apply a threshold
        return min(confidence, self.speed_confidence_threshold)