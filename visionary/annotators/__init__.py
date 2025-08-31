"""
Annotators for drawing bounding boxes, labels, masks, and other visual elements.
"""

from .base import BaseAnnotator, ColorPalette, Position

# Basic shape annotators
from .boxcorner import BoxCornerAnnotator  
from .circle import CircleAnnotator
from .dot import DotAnnotator
from .ellipse import EllipseAnnotator
from .polygon import PolygonAnnotator
from .roundbox import RoundBoxAnnotator
from .triangle import TriangleAnnotator
from .orientedbox import OrientedBoxAnnotator

# Text and label annotators
from .label import LabelAnnotator
from .richlabel import RichLabelAnnotator

# Mask and overlay annotators
from .mask import MaskAnnotator
from .halo import HaloAnnotator

# Color and visual effects
from .color import ColorAnnotator
from .bgoverlay import BackgroundOverlayAnnotator

# Privacy annotators from temporal_privacy.py
from .temporal_privacy import (
    TraceAnnotator,
    HeatMapAnnotator as HeatmapAnnotator,
    BlurAnnotator,
    PixelateAnnotator as PixelAnnotator,
    IconAnnotator
)

# Other annotators
from .percentagebar import PercentageBarAnnotator

# Try to import optional annotators
try:
    from .advanced_visual import AdvancedVisualAnnotator
except ImportError:
    AdvancedVisualAnnotator = None

__all__ = [
    "BaseAnnotator",
    "ColorPalette", 
    "Position",
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
    "TraceAnnotator",
    "HeatmapAnnotator",
    "BlurAnnotator",
    "PixelAnnotator", 
    "IconAnnotator",
    "PercentageBarAnnotator",
]

# Add optional annotators if available
if AdvancedVisualAnnotator is not None:
    __all__.append("AdvancedVisualAnnotator")
