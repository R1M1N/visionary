"""
Video Core Utilities for Visionary

Provides multi-format video support, streaming capabilities, 
and frame-by-frame processing with callbacks.
"""

import cv2
import numpy as np
from typing import Callable, Optional, Generator, Tuple
from dataclasses import dataclass


@dataclass
class VideoInfo:
    """Container for video metadata."""
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float  # in seconds
    
    @classmethod
    def from_video_capture(cls, cap: cv2.VideoCapture) -> 'VideoInfo':
        """Create VideoInfo from OpenCV VideoCapture object."""
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        return cls(
            width=width,
            height=height, 
            fps=fps,
            frame_count=frame_count,
            duration=duration
        )


class VideoSink:
    """Video writer for saving processed frames to video file."""
    
    def __init__(self, output_path: str, fps: float, frame_size: Tuple[int, int], 
                 fourcc: str = 'mp4v'):
        """
        Initialize video writer.
        
        Args:
            output_path: Path to output video file
            fps: Frames per second for output video
            frame_size: (width, height) of output frames
            fourcc: FourCC codec code
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.writer = None
        self._is_opened = False
    
    def open(self):
        """Open the video writer."""
        self.writer = cv2.VideoWriter(
            self.output_path, 
            self.fourcc, 
            self.fps, 
            self.frame_size
        )
        self._is_opened = self.writer.isOpened()
        return self._is_opened
    
    def write(self, frame: np.ndarray):
        """Write a frame to the video."""
        if not self._is_opened:
            raise RuntimeError("VideoSink not opened. Call open() first.")
        self.writer.write(frame)
    
    def release(self):
        """Release the video writer."""
        if self.writer:
            self.writer.release()
            self._is_opened = False
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def get_video_frames_generator(video_path: str, 
                              max_frames: Optional[int] = None) -> Generator[Tuple[int, np.ndarray], None, VideoInfo]:
    """
    Generator that yields video frames with frame indices.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to yield (None for all)
        
    Yields:
        Tuple of (frame_index, frame_array)
        
    Returns:
        VideoInfo object containing video metadata
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")
    
    try:
        video_info = VideoInfo.from_video_capture(cap)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
                
            yield frame_idx, frame
            frame_idx += 1
            
            if max_frames is not None and frame_idx >= max_frames:
                break
                
    finally:
        cap.release()
    
    return video_info


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
    
    def get_video_info(self) -> VideoInfo:
        """Get video metadata as VideoInfo object."""
        return VideoInfo.from_video_capture(self.cap)

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
