
"""
Video Enhancement for Visionary

Provides custom codec support, frame interpolation,
resolution scaling with quality preservation,
and temporal filtering for noise reduction.
"""

import cv2
import numpy as np
from typing import Optional, Any

class VideoEnhancer:
    def __init__(self, codec: str = 'X264', output_path: Optional[str] = None, fps: Optional[float] = None):
        """
        Initialize video enhancer with custom codec and output path.
        Args:
            codec: FourCC code for video compression
            output_path: Path to save enhanced video
            fps: Frame rate for output video
        """
        self.codec = codec
        self.output_path = output_path
        self.fps = fps
        self.writer = None

    def initialize_writer(self, frame_size: tuple):
        if self.output_path is None or self.fps is None:
            raise ValueError('Output path and fps must be specified to initialize writer')
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, frame_size)

    def write_frame(self, frame: np.ndarray):
        if self.writer is None:
            raise RuntimeError('VideoWriter not initialized')
        self.writer.write(frame)

    def release_writer(self):
        if self.writer:
            self.writer.release()

    @staticmethod
    def interpolate_frames(frame1: np.ndarray, frame2: np.ndarray, alpha: float) -> np.ndarray:
        """
        Interpolate between two frames using alpha blending.
        Args:
            frame1: First frame
            frame2: Second frame
            alpha: Interpolation factor between 0 and 1
        Returns:
            Interpolated frame
        """
        return cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)

    @staticmethod
    def scale_resolution(frame: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Scale resolution with quality preservation using cubic interpolation.
        Args:
            frame: Input frame
            scale_factor: Scaling factor
        Returns:
            Scaled frame
        """
        new_size = (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor))
        scaled_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_CUBIC)
        return scaled_frame

    @staticmethod
    def temporal_filter(frames: list, kernel_size: int = 5) -> np.ndarray:
        """
        Perform temporal filtering (moving average) over frames to reduce noise.
        Args:
            frames: List of frames (numpy arrays) to filter
            kernel_size: Number of frames to average
        Returns:
            Filtered frame
        """
        if len(frames) < kernel_size:
            kernel_size = len(frames)
        sum_frames = np.zeros_like(frames[0], dtype=np.float32)
        for f in frames[-kernel_size:]:
            sum_frames += f.astype(np.float32)
        avg_frame = (sum_frames / kernel_size).astype(np.uint8)
        return avg_frame
