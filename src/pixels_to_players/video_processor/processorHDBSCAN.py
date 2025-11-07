"""
Video Scene Segmentation and Clustering Module

This module provides functionality to automatically segment videos into scenes,
extract representative frames, and cluster similar scenes into categories.
Uses histogram analysis and similarity thresholds for scene detection and clustering.

This will be the one step in creating a Heatmap. The next step is to accumulate
the attention data within each scene.
"""

import numpy as np
import json
import os
import csv
from sklearn.preprocessing import StandardScaler
import hdbscan
import cv2


class HDBSCANClusterAnalyzer:
    """
    A processor for segmenting videos into scenes and clustering similar scenes.
    Outputs sample images for each scene and CSV for detailed separation schema.
    """

    def __init__(self, min_cluster_size=100, min_samples=15,
                 cluster_selection_epsilon=1):
        """
        Args:
            min_cluster_size (int): Increase this value to make clustering more fuzzy and reduce cluster count
            min_samples (int): Increase this value to make clustering more stable
            cluster_selection_epsilon (float): Increase this value to merge similar clusters
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon

    def _load_features_from_json(self, json_path):
        """Load pre-computed features from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        embeddings = np.array(data['frame_features'])
        timestamps = np.array(data['frame_timestamps'])
        frame_indices = np.array(data['frame_indices'])

        metadata = data.get('metadata', {})
        recording_config = data.get('recording_config', {})

        return embeddings, timestamps, frame_indices, metadata, recording_config

    def _find_cluster_representative_frames(self, labels, embeddings,
                                            frame_indices, timestamps):
        """Find representative frame for each cluster (based on average feature vector within cluster)"""

        unique_clusters = set(labels) - {-1}

        cluster_rep_frames = {}
        cluster_stats = {}

        for cluster_label in sorted(unique_clusters):
            cluster_frame_indices = []
            for i, label in enumerate(labels):
                if label == cluster_label:
                    cluster_frame_indices.append(i)

            if not cluster_frame_indices:
                cluster_rep_frames[cluster_label] = None
                continue

            cluster_embeddings = embeddings[cluster_frame_indices]
            mean_features = np.mean(cluster_embeddings, axis=0)

            distances = []
            for i in cluster_frame_indices:
                dist = np.linalg.norm(embeddings[i] - mean_features)
                distances.append((i, dist))

            best_idx, min_dist = min(distances, key=lambda x: x[1])
            best_frame_index = frame_indices[best_idx]

            cluster_rep_frames[cluster_label] = best_frame_index

            cluster_timestamps = [timestamps[i] for i in cluster_frame_indices]
            cluster_stats[cluster_label] = {
                'frame_count': len(cluster_frame_indices),
                'time_span': (
                    float(min(cluster_timestamps)),
                    float(max(cluster_timestamps))),
                'duration': float(
                    max(cluster_timestamps) - min(cluster_timestamps)),
                'avg_distance': float(np.mean([d for _, d in distances])),
                'rep_frame_distance': float(min_dist),
                'rep_frame_timestamp': float(timestamps[best_idx])
            }

        return cluster_rep_frames, cluster_stats

    def _extract_cluster_representative_frames(self, cluster_rep_frames,
                                               video_path, output_dir, fps):
        """Extract and save representative frames for each cluster from video"""

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        extracted_frames = {}

        for cluster_label, frame_index in cluster_rep_frames.items():
            if frame_index is None:
                extracted_frames[cluster_label] = None
                continue

            if frame_index < 0 or frame_index >= total_frames:
                extracted_frames[cluster_label] = None
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()

            if ret:
                frame_filename = f"cluster_{cluster_label:02d}_frame_{frame_index:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)

                timestamp = frame_index / fps

                extracted_frames[cluster_label] = {
                    'cluster_label': cluster_label,
                    'frame_index': frame_index,
                    'frame_path': frame_path,
                    'timestamp': timestamp
                }
            else:
                extracted_frames[cluster_label] = None

        cap.release()
        return extracted_frames

    def _save_cluster_info_to_csv(self, cluster_stats, cluster_rep_frames,
                                  output_path):
        """Save cluster information to CSV file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'cluster_label',
                'frame_count',
                'start_time',
                'end_time',
                'duration',
                'representative_frame',
                'rep_frame_timestamp',
                'avg_distance'
            ])

            for cluster_label in sorted(cluster_stats.keys()):
                stats = cluster_stats[cluster_label]
                rep_frame = cluster_rep_frames.get(cluster_label, "N/A")

                writer.writerow([
                    cluster_label,
                    stats['frame_count'],
                    f"{stats['time_span'][0]:.2f}",
                    f"{stats['time_span'][1]:.2f}",
                    f"{stats['duration']:.2f}",
                    rep_frame,
                    f"{stats['rep_frame_timestamp']:.2f}",
                    f"{stats['avg_distance']:.6f}"
                ])

    def _save_clustering_summary(self, labels, cluster_stats, metadata,
                                 recording_config, output_path):
        """Save clustering analysis summary"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        unique_labels = set(labels)
        noise_count = list(labels).count(-1)
        valid_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        serializable_cluster_stats = {}
        for cluster_label, stats in cluster_stats.items():
            serializable_cluster_stats[str(cluster_label)] = stats

        summary = {
            'clustering_summary': {
                'total_frames': len(labels),
                'total_clusters': valid_clusters,
                'noise_frames': noise_count,
                'noise_percentage': f"{noise_count / len(labels) * 100:.1f}%",
                'average_cluster_size': float(np.mean(
                    [s['frame_count'] for s in cluster_stats.values()])),
                'min_cluster_size': int(
                    np.min([s['frame_count'] for s in cluster_stats.values()])),
                'max_cluster_size': int(
                    np.max([s['frame_count'] for s in cluster_stats.values()])),
                'clustering_time': metadata.get('feature_extraction_time',
                                                'N/A')
            },
            'cluster_details': serializable_cluster_stats,
            'original_metadata': metadata,
            'recording_config': recording_config
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def _save_temporal_segments(self, labels, timestamps, cluster_rep_frames,
                                output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sorted_indices = np.argsort(timestamps)
        sorted_labels = labels[sorted_indices]
        sorted_timestamps = timestamps[sorted_indices]
        rows = []
        start_time = sorted_timestamps[0]
        prev_label = sorted_labels[0]
        for i in range(1, len(sorted_labels)):
            cur_label = sorted_labels[i]
            cur_time = sorted_timestamps[i]
            if cur_label != prev_label:
                end_time = sorted_timestamps[i - 1]
                frame_index = cluster_rep_frames.get(prev_label, None)
                frame_path = (
                    f"heatmaps/category_{prev_label}_representative.jpg"
                    if frame_index is not None and prev_label != -1
                    else ""
                )
                if prev_label != -1:
                    rows.append([
                        f"category_{prev_label}",
                        round(float(start_time), 3),
                        round(float(end_time), 3),
                        frame_path
                    ])
                start_time = cur_time
                prev_label = cur_label
        end_time = sorted_timestamps[-1]
        if prev_label != -1:
            frame_index = cluster_rep_frames.get(prev_label, None)
            frame_path = (
                f"heatmaps/category_{prev_label}_representative.jpg"
                if frame_index is not None and prev_label != -1
                else ""
            )
            rows.append([
                f"category_{prev_label}",
                round(float(start_time), 3),
                round(float(end_time), 3),
                frame_path
            ])
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["category", "start_sec", "end_sec", "frame_path"])
            writer.writerows(rows)

    def analyze_clusters(self, json_path, video_path=None,
                         output_base_dir="./output"):
        """Main analysis pipeline: focus on cluster analysis only"""

        json_filename = os.path.basename(json_path)
        video_name = os.path.splitext(json_filename)[0].replace('features_', '')

        output_dir = os.path.join(output_base_dir, video_name)

        embeddings, timestamps, frame_indices, metadata, recording_config = self._load_features_from_json(
            json_path)

        fps = recording_config.get('fps', 15)

        scaler = StandardScaler()
        X = scaler.fit_transform(embeddings)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(X)

        cluster_rep_frames, cluster_stats = self._find_cluster_representative_frames(
            labels, embeddings, frame_indices, timestamps
        )

        csv_path = os.path.join(output_dir, "cluster_analysis.csv")
        self._save_cluster_info_to_csv(cluster_stats, cluster_rep_frames,
                                       csv_path)

        summary_path = os.path.join(output_dir, "clustering_summary.json")
        self._save_clustering_summary(labels, cluster_stats, metadata,
                                      recording_config, summary_path)

        extracted_frames_info = {}
        if video_path and os.path.exists(video_path):
            frames_dir = os.path.join(output_dir,
                                      "cluster_representative_frames")
            extracted_frames_info = self._extract_cluster_representative_frames(
                cluster_rep_frames, video_path, frames_dir, fps
            )

        temporal_csv_path = os.path.join(output_dir, "cluster_timeline.csv")
        self._save_temporal_segments(labels, timestamps, cluster_rep_frames,
                                     temporal_csv_path)

        return {
            'labels': labels,
            'cluster_stats': cluster_stats,
            'cluster_rep_frames': cluster_rep_frames,
            'extracted_frames': extracted_frames_info,
            'output_dir': output_dir,
            'metadata': metadata
        }

# for testing
if __name__ == "__main__":
    analyzer = HDBSCANClusterAnalyzer(
        min_cluster_size=100,
        min_samples=15,
        cluster_selection_epsilon=1
    )

    # change to test paths
    json_path = "../screen_recording/recordings/features_20251106_214227.json"
    video_path = "../screen_recording/recordings/screen_20251106_214227.mp4"

    try:
        results = analyzer.analyze_clusters(
            json_path=json_path,
            video_path=video_path,
            output_base_dir="./output"
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
