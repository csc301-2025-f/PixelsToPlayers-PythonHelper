import json
import csv
import cv2
import numpy as np
import os
from datetime import datetime


class GazeHeatmapGenerator:
    """
    Accumulates gaze points from gaze tracking data into heatmaps
    corresponding to segmented video scenes.
    """

    def __init__(self, video_width, video_height, blur_ksize=51,
                 colormap=cv2.COLORMAP_JET):
        self.video_width = video_width
        self.video_height = video_height
        self.blur_ksize = blur_ksize
        self.colormap = colormap

    def load_gaze_data(self, gaze_json_path):
        """Load gaze tracking data (timestamp, iris_x, iris_y)."""
        with open(gaze_json_path, "r") as f:
            data = json.load(f)
        return data

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

    def create_heatmap(self, gaze_points):
        """Generate a normalized heatmap from gaze points."""
        heatmap = np.zeros((self.video_height, self.video_width),
                           dtype=np.float32)

        # Accumulate gaze points
        for p in gaze_points:
            x = int(np.clip(p["iris_x"], 0, self.video_width - 1))
            y = int(np.clip(p["iris_y"], 0, self.video_height - 1))
            heatmap[y, x] += 1

        # Normalize and apply blur
        heatmap = cv2.GaussianBlur(heatmap, (self.blur_ksize, self.blur_ksize),
                                   0)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = heatmap.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, self.colormap)
        return heatmap_color

    def generate_per_segment(self, gaze_data, segments, output_dir):
        """Generate a heatmap for each video segment."""
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
            filename = f"heatmap_{seg['category']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, heatmap_img)

            heatmaps.append({
                "category": seg["category"],
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "heatmap_path": filepath
            })

        return heatmaps