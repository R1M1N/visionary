"""
Visionary: A comprehensive computer vision library for object detection,
tracking, annotation, and video processing.
"""

import warnings

__version__ = "0.1.0"
__author__ = "R1M1"
__email__ = "r1m1@example.com"

# Main API - always available
from .unified_interface import VisionaryAPI

# Core imports
from .detection.core import Detections

# Optional imports with error handling
try:
    from .trackers import ByteTracker
except ImportError:
    ByteTracker = None

# Annotators - comprehensive set of visualization tools
try:
    from .annotators import (
        BaseAnnotator,
        # Box and shape annotators
        BoxCornerAnnotator,
        CircleAnnotator,
        DotAnnotator,
        EllipseAnnotator,
        PolygonAnnotator,
        RoundBoxAnnotator,
        TriangleAnnotator,
        OrientedBoxAnnotator,
        # Text and label annotators  
        LabelAnnotator,
        RichLabelAnnotator,
        # Mask and overlay annotators
        MaskAnnotator,
        HaloAnnotator,
        # Color and visual effects
        ColorAnnotator,
        BackgroundOverlayAnnotator,
        # Privacy and blur annotators
        BlurAnnotator,
        PixelAnnotator,  # From your PixelateAnnotator
        # Temporal and tracking annotators
        TraceAnnotator,
        HeatmapAnnotator,  # From your HeatMapAnnotator
        # Interactive annotators
        IconAnnotator,
        PercentageBarAnnotator,
        # Advanced visual annotators
        # AdvancedVisualAnnotator,  # Comment out if not implemented
    )
except ImportError as e:
    warnings.warn(f"Some annotators could not be imported: {e}", ImportWarning)
    # Define fallbacks for essential annotators
    (BaseAnnotator, BoxCornerAnnotator, CircleAnnotator, DotAnnotator, 
     EllipseAnnotator, PolygonAnnotator, RoundBoxAnnotator, TriangleAnnotator, 
     OrientedBoxAnnotator, LabelAnnotator, RichLabelAnnotator, MaskAnnotator, 
     HaloAnnotator, ColorAnnotator, BackgroundOverlayAnnotator, BlurAnnotator, 
     PixelAnnotator, TraceAnnotator, HeatmapAnnotator, IconAnnotator, 
     PercentageBarAnnotator) = [None] * 21

# Detection tools
try:
    from .detection.tools import (
        PolygonZone,
        LineZone,
        InferenceSlicer,
        Smoother as DetectionSmoother
    )
except ImportError:
    PolygonZone = LineZone = InferenceSlicer = DetectionSmoother = None

# Datasets
try:
    from .datasets import DetectionDataset
except ImportError:
    DetectionDataset = None

# Metrics
try:
    from .metrics import (
        MeanAveragePrecision,
        AdditionalMetrics
    )
except ImportError:
    MeanAveragePrecision = AdditionalMetrics = None

# Utilities
try:
    from .utils.video import VideoInfo, VideoSink, get_video_frames_generator
    from .utils.image import ImageSink  
    from .utils.file import list_files_with_extensions
    from .utils.geometry import calculate_iou
    from .utils.draw import draw_bounding_box, draw_mask
except ImportError:
    VideoInfo = VideoSink = get_video_frames_generator = None
    ImageSink = list_files_with_extensions = None
    calculate_iou = draw_bounding_box = draw_mask = None

# Unified API and core types
try:
    from .model_files.task_types import TaskType
    from .model_files.task_config import TaskConfig
    from .models import ModelType, ModelConfig
    from .utils.input_handler import InputType
except ImportError as e:
    warnings.warn(f"Core API components could not be imported: {e}", ImportWarning)
    TaskType = TaskConfig = ModelType = ModelConfig = InputType = None

# Key components for supervision-like workflow
__all__ = [
    # Main API (first priority)
    "VisionaryAPI",
    
    # Core data structures
    "Detections",
    
    # Configuration types
    "TaskType",
    "TaskConfig", 
    "ModelType",
    "ModelConfig",
    "InputType",
    
    # Tracking
    "ByteTracker",
    
    # Annotators (visualization)
    "BaseAnnotator",
    "BoxCornerAnnotator", 
    "CircleAnnotator",
    "DotAnnotator",
    "EllipseAnnotator",
    "PolygonAnnotator",
    "RoundBoxAnnotator",
    "TriangleAnnotator",
    "OrientedBoxAnnotator",
    "LabelAnnotator",
    "RichLabelAnnotator",
    "MaskAnnotator",
    "HaloAnnotator",
    "ColorAnnotator",
    "BackgroundOverlayAnnotator",
    "BlurAnnotator",
    "PixelAnnotator",
    "TraceAnnotator",
    "HeatmapAnnotator",
    "IconAnnotator",
    "PercentageBarAnnotator",
    
    # Detection tools
    "PolygonZone",
    "LineZone", 
    "InferenceSlicer",
    "DetectionSmoother",
    
    # Datasets
    "DetectionDataset",
    
    # Metrics
    "MeanAveragePrecision",
    "AdditionalMetrics",
    
    # Utilities
    "VideoInfo",
    "VideoSink",
    "get_video_frames_generator",
    "ImageSink", 
    "list_files_with_extensions",
    "calculate_iou",
    "draw_bounding_box",
    "draw_mask",
]

# Backward compatibility aliases
Task = TaskType  # For backward compatibility

# Filter out None values from __all__ for cleaner imports
__all__ = [name for name in __all__ if globals().get(name) is not None]

def check_dependencies():
    """Check if optional dependencies are available."""
    missing = []
    components = [
        ("ByteTracker", ByteTracker),
        ("BaseAnnotator", BaseAnnotator),
        ("DetectionDataset", DetectionDataset),
        ("MeanAveragePrecision", MeanAveragePrecision),
        ("VideoInfo", VideoInfo),
    ]
    
    for name, component in components:
        if component is None:
            missing.append(name)
    
    if missing:
        warnings.warn(f"Missing optional components: {', '.join(missing)}", ImportWarning)
        print("💡 Install with: pip install visionary[all]")
        return False
    else:
        print("✅ All components loaded successfully!")
        return True

def __getattr__(name):
    """Provide helpful error messages for missing components."""
    if name == "VisionaryAPI":
        return VisionaryAPI
    
    # Provide helpful error messages for common components
    common_missing = {
        "ByteTracker": "Install tracking dependencies: pip install visionary[tracking]",
        "BaseAnnotator": "Install annotation dependencies: pip install visionary[viz]", 
        "DetectionDataset": "Install dataset dependencies: pip install visionary[datasets]",
    }
    
    if name in common_missing:
        raise ImportError(f"'{name}' is not available. {common_missing[name]}")
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Make the main API easily discoverable
def get_api():
    """Get the main VisionaryAPI instance."""
    return VisionaryAPI()

# Version info function
def version_info():
    """Print version and component information."""
    print(f"Visionary v{__version__}")
    print(f"Author: {__author__}")
    print(f"Available components: {len(__all__)}")
    check_dependencies()
