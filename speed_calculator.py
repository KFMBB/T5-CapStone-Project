import numpy as np
from scipy.signal import savgol_filter


class SpeedCalculator:
    def __init__(self):
        self.previous_positions = {}
        self.speed_history = {}

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
                if len(self.speed_history[track_id]) >= 5:
                    smoothed_speed = savgol_filter(self.speed_history[track_id], 5, 2)[-1]
                else:
                    smoothed_speed = speed

                self.previous_positions[track_id] = current_position
                return smoothed_speed
        else:
            self.previous_positions[track_id] = current_position

        return None