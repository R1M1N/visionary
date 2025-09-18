
"""
Tracking Utilities for Visionary

Provides visualization, history management, import/export,
and performance monitoring for tracking system.
"""

import json
import time
from typing import List, Dict, Optional
import numpy as np
import cv2

from .byte_track import Track, TrackState


class TrackVisualizer:
    """
    Track State Visualization - draw track boxes with IDs and colors.
    """
    def __init__(self, font_scale=0.5, thickness=1, color_palette=None):
        self.font_scale = font_scale
        self.thickness = thickness
        self.color_palette = color_palette
        if color_palette is None:
            # Fallback default colors
            self.color_palette = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (255, 255, 0), (255, 0, 255), (0, 255, 255)
            ]

    def visualize(self, image: np.ndarray, tracks: List[Track]) -> np.ndarray:
        for track in tracks:
            mean = track.mean
            x, y, w, h = int(mean[0]), int(mean[1]), int(mean[2]), int(mean[3])
            color = self.color_palette[track.track_id % len(self.color_palette)]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, self.thickness)
            label = f"ID:{track.track_id}"
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, self.thickness)
        return image


class TrackHistoryManager:
    """
    Maintain history of track positions for trajectory analysis.
    """
    def __init__(self, max_length=30):
        self.max_length = max_length
        self.history: Dict[int, List[np.ndarray]] = {}

    def update(self, tracks: List[Track]):
        for track in tracks:
            pos = track.mean[:4].copy()  # save bounding box
            self.history.setdefault(track.track_id, []).append(pos)
            if len(self.history[track.track_id]) > self.max_length:
                self.history[track.track_id].pop(0)

    def get_history(self, track_id: int) -> List[np.ndarray]:
        return self.history.get(track_id, [])


class TrackExporter:
    """
    Export and import tracking data to/from JSON format.
    """
    @staticmethod
    def export(tracks: List[Track]) -> str:
        export_data = []
        for track in tracks:
            data = {
                "track_id": track.track_id,
                "mean": track.mean.tolist(),
                "covariance": track.covariance.tolist(),
                "state": track.state,
                "age": track.age,
                "time_since_update": track.time_since_update
            }
            export_data.append(data)
        return json.dumps(export_data, indent=2)

    @staticmethod
    def import_data(json_string: str) -> List[Track]:
        import_data = json.loads(json_string)
        tracks = []
        for data in import_data:
            track = Track(
                mean=np.array(data["mean"]),
                covariance=np.array(data["covariance"]),
                track_id=data["track_id"],
                state=data["state"]
            )
            track.age = data["age"]
            track.time_since_update = data["time_since_update"]
            tracks.append(track)
        return tracks


class PerformanceMonitor:
    """
    Monitor runtime and memory usage during tracking.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.frame_times = []

    def start_frame(self):
        self.frame_start = time.time()

    def end_frame(self):
        duration = time.time() - self.frame_start
        self.frame_times.append(duration)

    def get_fps(self) -> float:
        if not self.frame_times:
            return 0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0

    def report(self):
        fps = self.get_fps()
        return {
            "average_fps": fps,
            "total_frames": len(self.frame_times),
            "total_time": time.time() - self.start_time
        }
