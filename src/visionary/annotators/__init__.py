"""
Visionary Annotators Package

This package provides a comprehensive annotation system for computer vision applications,
including base classes, geometric annotators, text annotators, and specialized visualization tools.
"""

from .base import (
    BaseAnnotator,
    ColorPalette,
    Position,
    AnnotatorConfig,
    calculate_optimal_text_color,
    blend_colors,
    adjust_color_brightness
)

from .geometry import (
    BoxAnnotator,
)

from .label import LabelAnnotator
from .mask import MaskAnnotator
from .trace import TraceAnnotator
from .circle import CircleAnnotator
from .ellipse import EllipseAnnotator
from .triangle import TriangleAnnotator
from .dot import DotAnnotator
from .roundbox import RoundBoxAnnotator
from .boxcorner import BoxCornerAnnotator
from .orientedbox import OrientedBoxAnnotator
from .polygon import PolygonAnnotator
from .halo import HaloAnnotator
from .color import ColorAnnotator
from .percentagebar import PercentageBarAnnotator
from .richlabel import RichLabelAnnotator
from .heatmap import HeatMapAnnotator
from .bgoverlay import BackgroundOverlayAnnotator
from .blur import BlurAnnotator
from .pixels import PixelateAnnotator
from .icon import IconAnnotator

__all__ = [
    # Base classes and utilities
    'BaseAnnotator',
    'ColorPalette',
    'Position',
    'AnnotatorConfig',
    'calculate_optimal_text_color',
    'blend_colors',
    'adjust_color_brightness',
    
    # All annotators
    'BoxAnnotator',
    'LabelAnnotator',
    'MaskAnnotator',
    'TraceAnnotator',
    'CircleAnnotator',
    'EllipseAnnotator',
    'TriangleAnnotator',
    'DotAnnotator',
    'RoundBoxAnnotator',
    'BoxCornerAnnotator',
    'OrientedBoxAnnotator',
    'PolygonAnnotator',
    'HaloAnnotator',
    'ColorAnnotator',
    'PercentageBarAnnotator',
    'RichLabelAnnotator',
    'HeatMapAnnotator',
    'BackgroundOverlayAnnotator',
    'BlurAnnotator',
    'PixelateAnnotator',
    'IconAnnotator'
]
