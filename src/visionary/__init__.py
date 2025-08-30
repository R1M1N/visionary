"""
Visionary: A comprehensive computer vision library for object detection,
tracking, annotation, and video processing.
"""

__version__ = "0.1.0"
__author__ = "Visionary Team"
__email__ = "team@visionary.ai"

# Core imports
from visionary.detection.core import Detections
# visionary/__init__.py

try:
    from visionary.trackers import ByteTracker
except ImportError:
    ByteTracker = None

# import other modules

# Annotators
from visionary.annotators import (
    BoxAnnotator,
    LabelAnnotator,
    MaskAnnotator,
    TraceAnnotator,
    CircleAnnotator,
    EllipseAnnotator,
    TriangleAnnotator,
    DotAnnotator,
    RoundBoxAnnotator,
    BoxCornerAnnotator,
    OrientedBoxAnnotator,
    PolygonAnnotator,
    HaloAnnotator,
    ColorAnnotator,
    PercentageBarAnnotator,
    RichLabelAnnotator,
    HeatMapAnnotator,
    BackgroundOverlayAnnotator,
    BlurAnnotator,
    PixelateAnnotator,
    IconAnnotator
)

# Detection tools
from visionary.detection.tools import (
    PolygonZone,
    LineZone,
    InferenceSlicer,
    DetectionSmoother
)

# Datasets
from visionary.datasets import DetectionDataset

# Metrics
from visionary.metrics import (
    MeanAveragePrecision,
    MeanAverageRecall,
    Precision,
    Recall,
    F1Score
)

# Utilities
from visionary.utils.video import VideoInfo, VideoSink, get_video_frames_generator
from visionary.utils.image import ImageSink
from visionary.utils.file import list_files_with_extensions

__all__ = [
    "Detections",
    "ByteTracker",
    "BoxAnnotator",
    "LabelAnnotator", 
    "MaskAnnotator",
    "TraceAnnotator",
    "PolygonZone",
    "LineZone",
    "DetectionDataset",
    "MeanAveragePrecision",
    "VideoInfo",
    "VideoSink",
]
