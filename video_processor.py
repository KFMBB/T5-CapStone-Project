import cv2
import numpy as np
import os
from camera_calibration import CameraCalibration
from car_detection import CarDetection
from bounding_box_constructor import BoundingBoxConstructor
from vehicle_tracker import VehicleTracker
from speed_calculator import SpeedCalculator
from depth_estimation import DepthEstimationModel
from depth_masker import DepthMasker


class VideoProcessor:
    def __init__(self, video_path, calibration_file='camera_calibration.json', road_mask_file='road_mask.npy', detection_confidence=0.4):
        self.video_path = video_path
        self.calibration_file = calibration_file
        self.road_mask_file = road_mask_file
        self.calibration = CameraCalibration()
        self.car_detection = CarDetection()
        self.detection_confidence = detection_confidence
        self.depth_model = DepthEstimationModel()
        self.tracker = VehicleTracker(max_frames_to_skip=10, min_hits=3, max_track_length=30)
        self.speed_calculator = SpeedCalculator(smoothing_window=5, speed_confidence_threshold=0.8, max_history=100)

        self.cap = cv2.VideoCapture(video_path)
        self.ret, self.frame = self.cap.read()
        if not self.ret:
            raise ValueError("Failed to read the video file.")
        self.height, self.width = self.frame.shape[:2]

        self.calibration = CameraCalibration()
        if os.path.exists(calibration_file):
            self.calibration.load_calibration(calibration_file)
        self.calibration.width = self.width
        self.calibration.height = self.height

        self.depth_masker = DepthMasker(self.height, self.width)
        self.ipm_matrix = None
        self.bbox_constructor = None

        self.output_path = 'Output/output_video.mp4'
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.out = cv2.VideoWriter(self.output_path, self.fourcc, self.fps, (self.width, self.height))

    def process_video(self):
        # Road mask selection or loading
        if os.path.exists(self.road_mask_file):
            # Load existing road mask
            self.depth_masker.load_road_mask(self.road_mask_file)
            print(f"Loaded existing road mask from {self.road_mask_file}")
        else:
            # Perform manual road selection
            print("Please select the road area...")
            self.depth_masker.manual_road_selection(self.frame)

            # Save the road mask for future use
            self.depth_masker.save_road_mask(self.road_mask_file)
            print(f"Saved road mask to {self.road_mask_file}")

        # Camera calibration
        if os.path.exists(self.calibration_file):
            self.calibration.load_calibration(self.calibration_file)
            print(f"Loaded existing camera calibration from {self.calibration_file}")
        else:
            print("Performing camera calibration...")
            calibration_frames = [self.depth_masker.apply_mask(self.cap.read()[1]) for _ in range(10)]
            self.calibration.calibrate_camera(calibration_frames)
            self.calibration.save_calibration(self.calibration_file)
            print(f"Saved camera calibration to {self.calibration_file}")

        # Get necessary matrices
        self.ipm_matrix = self.calibration.ipm_matrix
        camera_matrix = self.calibration.get_camera_matrix()
        vanishing_points = self.calibration.vanishing_points

        # Initialize BoundingBoxConstructor with calibration results
        self.bbox_constructor = BoundingBoxConstructor(vanishing_points, camera_matrix)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video with {total_frames} frames...")

        ipm_view_saved = False

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / self.fps

            # Create a copy of the original frame for visualization
            vis_frame = frame.copy()

            # Apply depth masking
            masked_frame = self.depth_masker.apply_mask(frame)

            # Visualize the mask
            mask_vis = cv2.addWeighted(frame, 0.7, cv2.cvtColor(self.depth_masker.get_mask(), cv2.COLOR_GRAY2BGR), 0.3,
                                       0)

            # Apply IPM
            ipm_frame = self.calibration.apply_ipm(masked_frame)

            # Save one frame of the IPM view
            if not ipm_view_saved:
                cv2.imwrite('Output/ipm_view.jpg', ipm_frame)
                ipm_view_saved = True

            # Estimate depth
            depth_map = self.depth_model.estimate_depth(ipm_frame)

            # Normalize depth map for visualization
            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            # Apply depth-based masking
            depth_masked_frame = self.depth_masker.apply_depth_mask(ipm_frame, depth_map)

            # Detect vehicles
            detections = self.car_detection.detect_cars(depth_masked_frame, self.ipm_matrix, self.detection_confidence)

            # Construct 3D bounding boxes
            bboxes_3d = []
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                center_depth = np.mean(depth_map[int(y1):int(y2), int(x1):int(x2)])
                if np.isnan(center_depth) or np.isinf(center_depth):
                    print(f"Warning: Invalid depth value for detection {det}")
                    continue
                bbox_3d = self.bbox_constructor.construct_3d_box([x1, y1, x2, y2], center_depth, aspect_ratio=1.5)
                if bbox_3d is not None:
                    bboxes_3d.append(bbox_3d)

            # Track vehicles
            try:
                tracks = self.tracker.update(bboxes_3d)
            except Exception as e:
                print(f"Error in tracking: {str(e)}")
                tracks = {}

                # Calculate speeds and visualize results
                for track_id, track in tracks.items():
                    if track['hits'] >= self.tracker.min_hits and track['missed_frames'] == 0:
                        current_position = np.mean(track['bbox_3d'], axis=0)
                        speed, confidence = self.speed_calculator.calculate_speed(
                            track_id, current_position, current_time, current_time - 1 / self.fps, unit='km/h'
                        )

                        # Visualize results
                        corners_2d = self.bbox_constructor.project_3d_to_2d(track['bbox_3d'])
                        self.draw_3d_box(frame, corners_2d)
                        if speed is not None:
                            cv2.putText(frame, f"ID: {track_id}, Speed: {speed:.2f} km/h",
                                        (int(corners_2d[0][0]), int(corners_2d[0][1]) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Draw 2D bounding box
                    x1, y1 = corners_2d.min(axis=0).astype(int)
                    x2, y2 = corners_2d.max(axis=0).astype(int)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

                    # Draw 3D bounding box
                    self.draw_3d_box(vis_frame, corners_2d)

                    if speed is not None:
                        # Display ID, speed, and confidence
                        cv2.putText(vis_frame, f"ID: {track_id}", (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(vis_frame, f"Speed: {speed:.2f} km/h", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(vis_frame, f"Conf: {confidence:.2f}", (x1, y1 + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Combine visualizations
            top_row = np.hstack((vis_frame, mask_vis))
            bottom_row = np.hstack((depth_vis, cv2.resize(ipm_frame, (vis_frame.shape[1], vis_frame.shape[0]))))
            combined_vis = np.vstack((top_row, bottom_row))

            # Resize the combined visualization to fit the output video dimensions
            combined_vis = cv2.resize(combined_vis, (self.width, self.height))

            # Write frame to output video
            self.out.write(combined_vis)

            # Display the frame (optional)
            cv2.imshow('Processed Frame', combined_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if frame_count % 100 == 0:
                print(f"Processed frame {frame_count}/{total_frames} ({frame_count / total_frames * 100:.2f}%)")

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        print("Video processing completed.")

    def draw_3d_box(self, img, corners):
        # Draw the base of the 3D box
        for i in range(4):
            cv2.line(img, tuple(corners[i].astype(int)), tuple(corners[(i + 1) % 4].astype(int)), (0, 255, 0), 2)

        # Draw the top of the 3D box
        for i in range(4):
            cv2.line(img, tuple(corners[i + 4].astype(int)), tuple(corners[(i + 1) % 4 + 4].astype(int)), (0, 255, 0),
                     2)

        # Draw the vertical lines
        for i in range(4):
            cv2.line(img, tuple(corners[i].astype(int)), tuple(corners[i + 4].astype(int)), (0, 255, 0), 2)

        return img