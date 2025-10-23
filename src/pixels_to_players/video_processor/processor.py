"""
Video Scene Segmentation and Clustering Module

This module provides functionality to automatically segment videos into scenes,
extract representative frames, and cluster similar scenes into categories.
Uses histogram analysis and similarity thresholds for scene detection and clustering.

This will be the one step in creating a Heatmap. The next step is to accumulate
the attention data within each scene.
"""

import cv2
import numpy as np
import os
import csv
from utils import read_video, save_image


class VideoProcessor:
    """
    A processor for segmenting videos into scenes and clustering similar scenes.
    Outputs sample images for each scene and CSV for detailed separation schema

    Attributes:
        hist_threshold: Threshold for histogram difference to detect scene changes
        similarity_threshold: Threshold for clustering similar scenes
        frame_sample_rate: Rate to sample frames, 1 frame per frame_sample_rate sec
        min_segment_sec: Minimum segment duration in seconds to be considered valid
    """

    def __init__(self, hist_threshold=0.1, similarity_threshold=0.1,
                 frame_sample_rate=0.1, min_segment_sec=0.1):
        self.hist_threshold = hist_threshold
        self.similarity_threshold = similarity_threshold
        self.frame_sample_rate = frame_sample_rate
        self.min_segment_sec = min_segment_sec

    def segment_video(self, video_path):
        """
        Segment video into scenes based on histogram differences.

        Args:
            video_path (str): Path to the input video file

        Returns:
            tuple: A tuple containing:
                - scenes (list): List of tuples (start_frame, end_frame) for each scene
                - fps (float): Frames per second of the video
                - frame_count (int): Total number of frames in the video
        """
        cap, fps, frame_count = read_video(video_path)
        prev_hist = None
        scenes = []
        current_scene_start = 0

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames based on frame_sample_rate to improve performance
            step = max(1, int(fps * self.frame_sample_rate))
            if i % step != 0:
                continue

            # Calculate and normalize 3D color histogram (8 bins per channel)
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8],
                                [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            # Compare with previous frame's histogram to detect scene changes
            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist,
                                       cv2.HISTCMP_BHATTACHARYYA)
                if diff > self.hist_threshold:
                    scenes.append((current_scene_start, i))
                    current_scene_start = i
            prev_hist = hist

        # Add the final scene
        scenes.append((current_scene_start, frame_count))
        cap.release()
        return scenes, fps, frame_count

    def extract_segment_frames(self, video_path, scenes, fps):
        """
        Extract frames from each video segment for further processing.

        Args:
            video_path (str): Path to the input video file
            scenes (list): List of scene tuples (start_frame, end_frame)
            fps (float): Frames per second of the video

        Returns:
            list: List of lists, where each inner list contains frames from one segment
        """
        cap, _, _ = read_video(video_path)
        all_segment_frames = []

        for start, end in scenes:
            frames = []
            # Set video position to the start of the current segment
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for i in range(start, end):
                ret, frame = cap.read()
                if not ret:
                    break
                # Sample frames to reduce processing load
                step = max(1, int(fps * self.frame_sample_rate))
                if i % step == 0:
                    frames.append(frame)
            all_segment_frames.append(frames)

        cap.release()
        return all_segment_frames

    def compute_segment_representatives(self, all_segment_frames):
        """
        Compute representative frames for each segment based on histogram similarity.

        Args:
            all_segment_frames (list): List of frame lists for each segment

        Returns:
            tuple: A tuple containing:
                - scene_features (list): Average histograms for each segment
                - scene_images (list): Representative frame for each segment

        Note:
            The representative frame is chosen as the frame closest to the segment's
            average histogram, representing the most typical appearance of the segment.
        """
        scene_features = []
        scene_images = []

        for frames in all_segment_frames:
            if not frames:
                scene_features.append(None)
                scene_images.append(None)
                continue

            # Calculate histograms for all frames in the segment
            hists = [cv2.calcHist([f], [0, 1, 2], None, [8, 8, 8],
                                  [0, 256, 0, 256, 0, 256]) for f in frames]
            hists = [cv2.normalize(h, h).flatten() for h in hists]

            # Find the frame closest to the average histogram
            avg_hist = np.mean(hists, axis=0)
            distances = [cv2.norm(h - avg_hist) for h in hists]
            rep_frame = frames[np.argmin(distances)]

            scene_features.append(avg_hist)
            scene_images.append(rep_frame)

        return scene_features, scene_images

    def cluster_scenes(self, scenes, scene_features, scene_images, fps):
        """
        Cluster similar scenes into categories based on visual similarity.

        Args:
            scenes (list): List of scene tuples (start_frame, end_frame)
            scene_features (list): Feature vectors (histograms) for each scene
            scene_images (list): Representative frames for each scene
            fps (float): Frames per second of the video

        Returns:
            tuple: A tuple containing:
                - category_frames (list): Frames grouped by category
                - category_segments (list): Scene segments grouped by category
                - segment_info (list): Detailed information about each segment
        """
        categories = []  # Feature vectors for each category
        category_frames = []  # Representative frames for each category
        category_segments = []  # Time segments for each category
        others_frames = []  # Frames for short segments ("others" category)
        others_segments = []  # Time segments for short segments
        segment_info = []  # Metadata for all segments

        for idx, feat in enumerate(scene_features):
            start, end = scenes[idx]
            duration_sec = (end - start) / fps

            if feat is None:
                continue

            # Handle short segments by putting them in "others" category
            if duration_sec < self.min_segment_sec:
                others_frames.append(scene_images[idx])
                others_segments.append((start / fps, end / fps))
                segment_info.append({
                    "category": "others",
                    "start_sec": start / fps,
                    "end_sec": end / fps,
                    "frame_path": None
                })
                continue

            # Try to find existing category for this scene
            found = False
            for c_idx, cat_feat in enumerate(categories):
                dist = cv2.norm(feat - cat_feat)
                if dist < self.similarity_threshold:
                    category_frames[c_idx].append(scene_images[idx])
                    category_segments[c_idx].append((start / fps, end / fps))
                    segment_info.append({
                        "category": f"category_{c_idx + 1}",
                        "start_sec": start / fps,
                        "end_sec": end / fps,
                        "frame_path": None
                    })
                    found = True
                    break

            # Create new category if no similar category found
            if not found:
                categories.append(feat)
                category_frames.append([scene_images[idx]])
                category_segments.append([(start / fps, end / fps)])
                segment_info.append({
                    "category": f"category_{len(categories)}",
                    "start_sec": start / fps,
                    "end_sec": end / fps,
                    "frame_path": None
                })

        # Add "others" category if there are short segments
        if others_frames:
            category_frames.append(others_frames)
            category_segments.append(others_segments)

        return category_frames, category_segments, segment_info

    def process_video(self, video_path, output_dir, csv_path):
        """
        Segment video, cluster scenes, then generate outputs.

        Args:
            video_path (str): Path to input video file
            output_dir (str): Directory to save representative frame images
            csv_path (str): Path to output CSV file with segment metadata

        Returns:
            tuple: A tuple containing:
                - category_frames (list): Frames grouped by category
                - category_segments (list): Time segments grouped by category  
                - segment_info (list): Detailed information about each segment

        Note:
            Results:
            - Representative frame images for each category
            - CSV file with segment timing and category information
            - Create Console output showing category segments
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Execute processing pipeline
        scenes, fps, _ = self.segment_video(video_path)
        all_segment_frames = self.extract_segment_frames(video_path, scenes,
                                                         fps)
        scene_features, scene_images = self.compute_segment_representatives(
            all_segment_frames)
        category_frames, category_segments, segment_info = self.cluster_scenes(
            scenes, scene_features, scene_images, fps
        )

        # Save representative frames and update segment info with file paths
        for c_idx, frames in enumerate(category_frames):
            rep_frame = frames[0]

            # Determine category name and output filename
            if c_idx == len(category_frames) - 1 and len(category_frames) > 1:
                frame_file = os.path.join(output_dir,
                                          'category_others_representative.jpg')
                cat_name = "others"
            else:
                frame_file = os.path.join(output_dir,
                                          f'category_{c_idx + 1}_representative.jpg')
                cat_name = f"category_{c_idx + 1}"

            save_image(rep_frame, frame_file)

            # Update segment info with relative paths to representative frames
            for info in segment_info:
                if info["category"] == cat_name and info["frame_path"] is None:
                    info["frame_path"] = os.path.relpath(frame_file,
                                                         start=os.path.dirname(
                                                             csv_path))

        # Print segment information to console
        for c_idx, segments in enumerate(category_segments):
            cat_name = "others" if c_idx == len(category_segments) - 1 and len(
                category_segments) > 1 else f"category_{c_idx + 1}"
            print(f"{cat_name}: {segments}")

        # Write segment information to CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["category", "start_sec",
                                                   "end_sec", "frame_path"])
            writer.writeheader()
            writer.writerows(segment_info)

        return category_frames, category_segments, segment_info