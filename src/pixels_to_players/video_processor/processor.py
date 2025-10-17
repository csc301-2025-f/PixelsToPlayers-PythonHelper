import cv2
import numpy as np
import os
import csv
from utils import read_video, save_image

class VideoProcessor:
    def __init__(self, hist_threshold=0.1, similarity_threshold=0.1,
                 frame_sample_rate=0.1, min_segment_sec=0.1):
        self.hist_threshold = hist_threshold
        self.similarity_threshold = similarity_threshold
        self.frame_sample_rate = frame_sample_rate
        self.min_segment_sec = min_segment_sec

    def segment_video(self, video_path):
        cap, fps, frame_count = read_video(video_path)
        prev_hist = None
        scenes = []
        current_scene_start = 0

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            step = max(1, int(fps * self.frame_sample_rate))
            if i % step != 0:
                continue

            hist = cv2.calcHist([frame], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if diff > self.hist_threshold:
                    scenes.append((current_scene_start, i))
                    current_scene_start = i
            prev_hist = hist

        scenes.append((current_scene_start, frame_count))
        cap.release()
        return scenes, fps, frame_count

    def extract_segment_frames(self, video_path, scenes, fps):
        cap, _, _ = read_video(video_path)
        all_segment_frames = []

        for start, end in scenes:
            frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for i in range(start, end):
                ret, frame = cap.read()
                if not ret:
                    break
                step = max(1, int(fps * self.frame_sample_rate))
                if i % step == 0:
                    frames.append(frame)
            all_segment_frames.append(frames)

        cap.release()
        return all_segment_frames

    def compute_segment_representatives(self, all_segment_frames):
        scene_features = []
        scene_images = []

        for frames in all_segment_frames:
            if not frames:
                scene_features.append(None)
                scene_images.append(None)
                continue

            hists = [cv2.calcHist([f], [0,1,2], None, [8,8,8], [0,256,0,256,0,256]) for f in frames]
            hists = [cv2.normalize(h, h).flatten() for h in hists]
            avg_hist = np.mean(hists, axis=0)
            distances = [cv2.norm(h - avg_hist) for h in hists]
            rep_frame = frames[np.argmin(distances)]

            scene_features.append(avg_hist)
            scene_images.append(rep_frame)

        return scene_features, scene_images

    def cluster_scenes(self, scenes, scene_features, scene_images, fps):
        categories = []
        category_frames = []
        category_segments = []
        others_frames = []
        others_segments = []
        segment_info = []

        for idx, feat in enumerate(scene_features):
            start, end = scenes[idx]
            duration_sec = (end - start) / fps

            if feat is None:
                continue

            if duration_sec < self.min_segment_sec:
                # short scene into others
                others_frames.append(scene_images[idx])
                others_segments.append((start / fps, end / fps))
                segment_info.append({
                    "category": "others",
                    "start_sec": start / fps,
                    "end_sec": end / fps,
                    "frame_path": None
                })
                continue

            found = False
            for c_idx, cat_feat in enumerate(categories):
                dist = cv2.norm(feat - cat_feat)
                if dist < self.similarity_threshold:
                    category_frames[c_idx].append(scene_images[idx])
                    category_segments[c_idx].append((start / fps, end / fps))
                    segment_info.append({
                        "category": f"category_{c_idx+1}",
                        "start_sec": start / fps,
                        "end_sec": end / fps,
                        "frame_path": None
                    })
                    found = True
                    break
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

        if others_frames:
            category_frames.append(others_frames)
            category_segments.append(others_segments)

        return category_frames, category_segments, segment_info

    def process_video(self, video_path, output_dir, csv_path):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        scenes, fps, _ = self.segment_video(video_path)
        all_segment_frames = self.extract_segment_frames(video_path, scenes, fps)
        scene_features, scene_images = self.compute_segment_representatives(all_segment_frames)
        category_frames, category_segments, segment_info = self.cluster_scenes(
            scenes, scene_features, scene_images, fps
        )

        for c_idx, frames in enumerate(category_frames):
            rep_frame = frames[0]
            if c_idx == len(category_frames) - 1 and len(category_frames) > 1:
                frame_file = os.path.join(output_dir,
                                          f'category_others_representative.jpg')
            else:
                frame_file = os.path.join(output_dir,
                                          f'category_{c_idx + 1}_representative.jpg')

            save_image(rep_frame, frame_file)

            for info in segment_info:
                cat_name = info["category"]
                expected_name = "others" if c_idx == len(
                    category_frames) - 1 and len(
                    category_frames) > 1 else f"category_{c_idx + 1}"
                if cat_name == expected_name and info["frame_path"] is None:
                    info["frame_path"] = os.path.relpath(frame_file,
                                                         start=os.path.dirname(
                                                             csv_path))

        for c_idx, segments in enumerate(category_segments):
            cat_name = "others" if c_idx == len(category_segments)-1 and len(category_segments) > 1 else f"category_{c_idx+1}"
            print(f"{cat_name}: {segments}")

        # to csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["category", "start_sec", "end_sec", "frame_path"])
            writer.writeheader()
            writer.writerows(segment_info)

        return category_frames, category_segments, segment_info
