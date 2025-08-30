"""
Visionary Annotators Package

This package provides a comprehensive annotation system for computer vision
applications, including base classes, geometric annotators, text annotators,
and specialized visualization tools.
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

__all__ = [
    'BaseAnnotator',
    'ColorPalette', 
    'Position',
    'AnnotatorConfig',
    'calculate_optimal_text_color',
    'blend_colors',
    'adjust_color_brightness'
]
