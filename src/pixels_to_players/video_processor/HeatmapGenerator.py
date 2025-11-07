import json
import csv
import cv2
import numpy as np
import os
from datetime import datetime


class GazeHeatmapGenerator:
    """
    Generates heatmaps from gaze tracking data.
    Uses calibration data to map eye coordinates to screen/video coordinates.
    """

    def __init__(self, video_width, video_height, calibration_data,
                 blur_ksize=51, colormap=cv2.COLORMAP_JET):
        self.video_width = video_width
        self.video_height = video_height
        self.blur_ksize = blur_ksize
        self.colormap = colormap

        # Calibration data: list of dicts with screen_x/y (normalized) and eye_x/y
        self.calibration_data = calibration_data

        # Precompute mapping coefficients for linear approximation
        self._compute_mapping()

    def _compute_mapping(self):
        """Compute linear mapping from eye coordinates to screen coordinates."""
        eye_x = np.array([p["eye_x"] for p in self.calibration_data])
        eye_y = np.array([p["eye_y"] for p in self.calibration_data])
        screen_x = np.array([p["screen_x"] * self.video_width for p in self.calibration_data])
        screen_y = np.array([p["screen_y"] * self.video_height for p in self.calibration_data])

        # Fit linear model: screen = a * eye + b
        self.a_x, self.b_x = np.polyfit(eye_x, screen_x, 1)
        self.a_y, self.b_y = np.polyfit(eye_y, screen_y, 1)

    def map_to_screen(self, iris_x, iris_y):
        """Map raw eye coordinates to screen/video coordinates."""
        screen_x = int(np.clip(self.a_x * iris_x + self.b_x, 0, self.video_width - 1))
        screen_y = int(np.clip(self.a_y * iris_y + self.b_y, 0, self.video_height - 1))
        return screen_x, screen_y

    def create_heatmap(self, gaze_points, alpha=0.7):
        """create 1 heatmap"""
        if not gaze_points:
            return np.zeros((self.video_height, self.video_width, 4),
                            dtype=np.uint8)

        heatmap = np.zeros((self.video_height, self.video_width),
                           dtype=np.float32)

        for p in gaze_points:
            x, y = self.map_to_screen(p["iris_x"], p["iris_y"])
            if 0 <= x < self.video_width and 0 <= y < self.video_height:
                heatmap[y, x] += 1

        heatmap = cv2.GaussianBlur(heatmap, (self.blur_ksize, self.blur_ksize),
                                   0)

        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (
                        heatmap.max() - heatmap.min()) * 255
        heatmap = heatmap.astype(np.uint8)

        heatmap_color = cv2.applyColorMap(heatmap, self.colormap)

        heatmap_bgra = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2BGRA)

        # set alpha according to intensity
        alpha_channel = (
                    heatmap.astype(np.float32) / 255.0 * alpha * 255).astype(
            np.uint8)
        heatmap_bgra[:, :, 3] = alpha_channel

        return heatmap_bgra

    def generate_per_segment(self, gaze_data, segments, output_dir):
        """generate heatmap for each segment"""
        os.makedirs(output_dir, exist_ok=True)
        heatmaps = []

        for seg in segments:
            seg_gaze = [
                p for p in gaze_data
                if seg["start_sec"] <= (
                            p["timestamp"] - gaze_data[0]["timestamp"]) <= seg[
                       "end_sec"]
            ]
            if not seg_gaze:
                continue

            heatmap_img = self.create_heatmap(seg_gaze)

            filename = f"heatmap_{seg['category']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filepath = os.path.join(output_dir, filename)

            cv2.imwrite(filepath, heatmap_img)

            heatmaps.append({
                "category": seg["category"],
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "heatmap_path": filepath
            })

        return heatmaps

    def load_segments(self, csv_path):
        """Load video segment info (category, start_sec, end_sec)."""
        segments = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                segments.append({
                    "category": row["category"],
                    "start_sec": float(row["start_sec"]),
                    "end_sec": float(row["end_sec"]),
                    "frame_path": row["frame_path"]
                })
        return segments

        return heatmaps