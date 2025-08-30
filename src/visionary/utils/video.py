
"""
Video Core Utilities for Visionary

Provides multi-format video support (MP4, AVI, MOV, WebM),
streaming capabilities for real-time processing,
and frame-by-frame processing with callbacks.
"""

import cv2
from typing import Callable, Optional

class VideoProcessor:
    def __init__(self, video_path: Optional[str] = None, stream_source: Optional[int] = None):
        """
        Initialize video processor.
        Args:
            video_path: Path to video file (MP4, AVI, MOV, WebM etc.)
            stream_source: Camera index or stream URL for real-time processing
        """
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
        elif stream_source is not None:
            self.cap = cv2.VideoCapture(stream_source)
        else:
            raise ValueError("Either video_path or stream_source must be provided")

        if not self.cap.isOpened():
            raise IOError("Failed to open video source")

    def get_fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame_size(self) -> tuple:
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def process_frames(self, frame_callback: Callable[[int, any], None], max_frames: Optional[int] = None) -> None:
        """
        Process video frames one by one.
        Args:
            frame_callback: function to call on each frame (frame_index, frame)
            max_frames: Optional limit on number of frames to process
        """
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_callback(frame_idx, frame)
            frame_idx += 1
            if max_frames is not None and frame_idx >= max_frames:
                break

    def release(self) -> None:
        self.cap.release()
