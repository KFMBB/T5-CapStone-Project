import cv2
import numpy as np
import os
import json
import logging
from camera_calibration import CameraCalibration
from car_detection import CarDetection
from bounding_box_constructor import BoundingBoxConstructor
from vehicle_tracker import VehicleTracker
from speed_calculator import SpeedCalculator
from depth_estimation import DepthEstimationModel
from masker import Masker

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoProcessor:
    def __init__(self, video_path, calibration_file='Files/my_camera_calibration.json', road_mask_file='Files/my_road_mask.npy',
                 detection_confidence=0.4, frame_skip=5):
        self.video_path = video_path
        self.calibration_file = calibration_file
        self.road_mask_file = road_mask_file
        self.detection_confidence = detection_confidence
        self.frame_skip = frame_skip

        self.calibration = CameraCalibration()
        self.car_detection = CarDetection()
        self.depth_model = DepthEstimationModel()
        self.tracker = VehicleTracker(max_frames_to_skip=10, min_hits=3, max_track_length=30)
        self.speed_calculator = SpeedCalculator(smoothing_window=5, speed_confidence_threshold=0.8, max_history=100)

        self.cap = cv2.VideoCapture(video_path)
        self.ret, self.frame = self.cap.read()
        if not self.ret:
            raise ValueError("Failed to read the video file.")
        self.height, self.width = self.frame.shape[:2]

        self.masker = Masker(self.height, self.width)
        self.ipm_matrix = None
        self.bbox_constructor = None

        self.output_path = 'Output/output_video.mp4'
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.out = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))

        self.results = []
        self.json_file_path = 'Output/speed_estimation_results.json'
        self.last_json_update = 0

    def process_video(self):
        # Road mask selection or loading
        if os.path.exists(self.road_mask_file):
            self.masker.load_road_mask(self.road_mask_file)
            logging.info(f"Loaded existing road mask from {self.road_mask_file}")
        else:
            logging.info("Please select the road area...")
            self.masker.manual_road_selection(self.frame)
            self.masker.save_road_mask(self.road_mask_file)
            logging.info(f"Saved road mask to {self.road_mask_file}")

        # Camera calibration
        if os.path.exists(self.calibration_file):
            self.calibration.load_calibration(self.calibration_file)
            logging.info(f"Loaded existing camera calibration from {self.calibration_file}")
        else:
            logging.info("Performing camera calibration...")
            calibration_frames = self._collect_calibration_frames()
            self.calibration.calibrate_camera(calibration_frames)
            self.calibration.save_calibration(self.calibration_file)
            logging.info(f"Saved camera calibration to {self.calibration_file}")

        # Reset video capture to the beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Get necessary matrices
        self.ipm_matrix = self.calibration.ipm_matrix
        camera_matrix = self.calibration.get_camera_matrix()
        vanishing_points = self.calibration.vanishing_points

        # Initialize BoundingBoxConstructor with calibration results
        self.bbox_constructor = BoundingBoxConstructor(vanishing_points, camera_matrix)

        frame_count = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cv2.namedWindow('Processed Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Processed Frame', 1280, 720)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % self.frame_skip != 0:
                continue

            current_time = frame_count / self.fps

            vis_frame = frame.copy()
            vis_ipm_frame = self.calibration.apply_ipm(frame.copy())

            masked_frame = self.masker.apply_mask(frame)
            ipm_frame = self.calibration.apply_ipm(masked_frame)

            depth_map = self.depth_model.estimate_depth(ipm_frame)

            # Debug: Print and visualize depth map statistics
            depth_min, depth_max, depth_mean, depth_std = depth_map.min(), depth_map.max(), depth_map.mean(), depth_map.std()
            logging.debug(f"Frame {frame_count}: Depth Map - min: {depth_min}, max: {depth_max}, mean: {depth_mean}, std: {depth_std}")

            # Visualize depth map (optional)
            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            # Uncomment to display depth map
            # cv2.imshow('Depth Map', depth_vis)

            detections = self.car_detection.detect_cars(ipm_frame, self.ipm_matrix, self.detection_confidence)

            # Debug: Print number of detections
            logging.debug(f"Frame {frame_count}: Number of detections: {len(detections)}")

            # Visualize 2D detections before 3D construction
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(vis_frame, f"{cls} {conf:.2f}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Optionally, save frames with 2D detections
            # cv2.imwrite(f"Output/detections_frame_{frame_count}.jpg", vis_frame)

            # Construct 3D bounding boxes
            bboxes_3d = []
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                # Ensure coordinates are within depth_map bounds
                x1_clamped = max(0, min(int(x1), self.width - 1))
                x2_clamped = max(0, min(int(x2), self.width - 1))
                y1_clamped = max(0, min(int(y1), self.height - 1))
                y2_clamped = max(0, min(int(y2), self.height - 1))

                center_depth = np.mean(depth_map[y1_clamped:y2_clamped, x1_clamped:x2_clamped])
                if np.isnan(center_depth) or np.isinf(center_depth):
                    logging.warning(f"Frame {frame_count}: Invalid depth value for detection {det}")
                    continue
                bbox_3d = self.bbox_constructor.construct_3d_box([x1, y1, x2, y2], center_depth, aspect_ratio=1.5)
                if bbox_3d is not None:
                    bboxes_3d.append(bbox_3d)

            # Print number of 3D bounding boxes
            logging.info(f"Frame {frame_count}: Number of 3D bounding boxes: {len(bboxes_3d)}")

            # Track vehicles
            tracks = self.tracker.update(bboxes_3d)

            # Print number of tracks
            logging.info(f"Frame {frame_count}: Number of tracks: {len(tracks)}")

            # Visualize results
            for track_id, track in tracks.items():
                corners_3d = track['bbox_3d']  # Assuming shape (8, 3)
                corners_2d = self.bbox_constructor.project_3d_to_2d(corners_3d)  # Should return shape (8, 2)

                # Debug: Print corners
                logging.debug(f"Track {track_id}: 3D Corners: {corners_3d}")
                logging.debug(f"Track {track_id}: 2D Corners: {corners_2d}")

                # Draw 2D projected points for verification
                for point in corners_2d:
                    cv2.circle(vis_frame, tuple(map(int, point)), 3, (0, 255, 255), -1)

                # Draw 3D bounding box in original frame
                self.draw_3d_box(vis_frame, corners_2d, color=(0, 255, 0))

                # Calculate speed
                current_position = np.mean(corners_3d, axis=0)
                speed, confidence = self.speed_calculator.calculate_speed(
                    track_id, current_position, frame_count, self.fps, unit='km/h'
                )

                # Prepare text to display
                if speed is not None:
                    speed_text = f"ID: {track_id}, Speed: {speed:.2f} km/h, Conf: {confidence:.2f}"
                else:
                    speed_text = f"ID: {track_id}, Speed: N/A"

                # Display in original frame
                x1, y1 = corners_2d.min(axis=0).astype(int)
                cv2.putText(vis_frame, speed_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Transform corners to IPM view
                ipm_corners = self.calibration.apply_ipm(corners_2d.reshape(-1, 1, 2)).reshape(-1, 2)

                # Debug: Print IPM projected corners
                logging.debug(f"Track {track_id}: IPM Projected Corners: {ipm_corners}")

                # Draw 3D bounding box in IPM view
                self.draw_3d_box(vis_ipm_frame, ipm_corners, color=(0, 255, 0))

                # Display text in IPM view
                ipm_x1, ipm_y1 = ipm_corners.min(axis=0).astype(int)
                cv2.putText(vis_ipm_frame, speed_text, (ipm_x1, ipm_y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Store results for JSON logging
                self._store_results(frame_count, track_id, speed, confidence, current_position)

            # Combine visualizations
            mask_vis = cv2.addWeighted(frame, 0.7, cv2.cvtColor(self.masker.get_mask(), cv2.COLOR_GRAY2BGR), 0.3, 0)
            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            top_row = np.hstack((vis_frame, mask_vis))
            bottom_row = np.hstack((depth_vis, vis_ipm_frame))
            combined_vis = np.vstack((top_row, bottom_row))

            # Resize the combined visualization to fit the output video dimensions
            combined_vis = cv2.resize(combined_vis, (self.width, self.height))

            # Add debug information
            cv2.putText(combined_vis, f"Frame: {frame_count}/{total_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Write frame to output video
            self.out.write(combined_vis)

            # Display the frame
            cv2.imshow('Processed Frame', combined_vis)

            # Print debug information
            logging.info(f"Processed frame {frame_count}/{total_frames}")

            # Save intermediate visualizations
            # Uncomment to save visualizations
            # cv2.imwrite(f"Output/vis_frame_{frame_count}.jpg", vis_frame)
            # cv2.imwrite(f"Output/ipm_frame_{frame_count}.jpg", vis_ipm_frame)
            # cv2.imwrite(f"Output/depth_map_frame_{frame_count}.jpg", depth_vis)

            # Update JSON file every 100 frames
            if frame_count % 100 == 0:
                self.write_results_to_json()
                self.last_json_update = frame_count

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):  # Add a pause functionality
                cv2.waitKey(0)

        # Write final results to JSON file
        self.write_results_to_json()
        logging.info(f"Total results stored: {len(self.results)}")

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        logging.info("Video processing completed.")

    def _collect_calibration_frames(self, num_frames=10):
        calibration_frames = []
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if ret:
                masked_frame = self.masker.apply_mask(frame)
                calibration_frames.append(masked_frame)
            else:
                break
        return calibration_frames

    def _store_results(self, frame_count, track_id, speed, confidence, position):
        """Store tracking results for JSON logging"""
        result = {
            'frame': int(frame_count),
            'track_id': int(track_id),
            'speed': float(speed) if speed is not None else None,
            'confidence': float(confidence) if confidence is not None else None,
            'position': position.tolist() if isinstance(position, np.ndarray) else position
        }
        self.results.append(result)
        logging.debug(f"Stored result for frame {frame_count}, track {track_id}")

    def write_results_to_json(self):
        """Write tracking results to a JSON file"""
        try:
            with open(self.json_file_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            logging.info(f"Updated JSON file at frame {self.last_json_update} with {len(self.results)} results")
        except Exception as e:
            logging.error(f"Error writing to JSON file: {e}")

    def draw_3d_box(self, img, corners, color=(0, 255, 0)):
        """
        Draw a 3D bounding box on the image.

        :param img: The image on which to draw.
        :param corners: A NumPy array of shape (8, 2) representing the 2D projections of the 8 corners of the 3D bounding box.
        :param color: The color of the bounding box lines.
        :return: The image with the bounding box drawn.
        """
        if corners.shape[0] != 8:
            logging.warning(f"Expected 8 corners for 3D box, got {corners.shape[0]}")
            return img

        # Define the connections between corners to draw the box
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]

        for start, end in connections:
            start_point = tuple(map(int, corners[start]))
            end_point = tuple(map(int, corners[end]))
            cv2.line(img, start_point, end_point, color, 2)

        return img

    def __del__(self):
        cv2.destroyAllWindows()
