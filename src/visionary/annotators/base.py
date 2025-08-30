"""
Base Annotator Framework for Visionary

This module provides the foundational annotator architecture including
abstract base classes, color management, positioning utilities, and
comprehensive configuration system for all annotation types.
"""

import abc
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union, List
from enum import Enum
import warnings


class ColorPalette:
    """Predefined color palettes for consistent annotation styling."""
    
    # Basic colors (BGR format for OpenCV)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    
    # Extended palette for multi-class detection
    COLORS = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green  
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (255, 192, 203), # Pink
        (0, 128, 0),    # Dark Green
        (139, 69, 19),  # Brown
        (255, 20, 147), # Deep Pink
        (0, 191, 255),  # Deep Sky Blue
        (50, 205, 50),  # Lime Green
        (255, 69, 0),   # Red Orange
    ]
    
    @classmethod
    def get_color(cls, index: int) -> Tuple[int, int, int]:
        """Get color by index, cycling through available colors."""
        return cls.COLORS[index % len(cls.COLORS)]


class Position(Enum):
    """Standard positioning options for annotations."""
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    TOP_CENTER = "top_center"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    BOTTOM_CENTER = "bottom_center"
    CENTER = "center"
    CENTER_LEFT = "center_left"
    CENTER_RIGHT = "center_right"


class BaseAnnotator(abc.ABC):
    """
    Abstract base class for all annotators in Visionary.
    
    This class provides common functionality including:
    - Color management and palettes
    - Positioning utilities
    - Configuration management
    - Text rendering utilities
    - Common drawing operations
    """
    
    def __init__(
        self, 
        color: Optional[Union[Tuple[int, int, int], str]] = None,
        thickness: int = 2,
        text_color: Optional[Tuple[int, int, int]] = None,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        **kwargs
    ):
        """
        Initialize base annotator.
        
        Args:
            color: Annotation color (BGR tuple) or color name
            thickness: Line/border thickness
            text_color: Text color (BGR tuple)
            text_scale: Text scale factor
            text_thickness: Text thickness
            text_padding: Padding around text
            **kwargs: Additional configuration parameters
        """
        # Color management
        self.color = self._parse_color(color) if color else ColorPalette.RED
        self.text_color = text_color if text_color else ColorPalette.WHITE
        
        # Drawing parameters  
        self.thickness = thickness
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.text_padding = text_padding
        
        # Configuration storage
        self.config = kwargs
        
        # Font configuration
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def _parse_color(self, color: Union[Tuple[int, int, int], str]) -> Tuple[int, int, int]:
        """Parse color from various input formats."""
        if isinstance(color, str):
            color_map = {
                'red': ColorPalette.RED,
                'green': ColorPalette.GREEN,
                'blue': ColorPalette.BLUE,
                'yellow': ColorPalette.YELLOW,
                'cyan': ColorPalette.CYAN,
                'magenta': ColorPalette.MAGENTA,
                'white': ColorPalette.WHITE,
                'black': ColorPalette.BLACK,
            }
            return color_map.get(color.lower(), ColorPalette.RED)
        
        elif isinstance(color, (tuple, list)) and len(color) == 3:
            return tuple(color)
        
        else:
            warnings.warn(f"Invalid color format: {color}. Using default red.")
            return ColorPalette.RED
    
    @abc.abstractmethod
    def annotate(
        self, 
        scene: np.ndarray, 
        detections: Any,
        **kwargs
    ) -> np.ndarray:
        """
        Abstract method to be implemented by all annotators.
        
        Args:
            scene: Input image/scene to annotate
            detections: Detection data to visualize
            **kwargs: Additional parameters specific to annotator
            
        Returns:
            Annotated image
        """
        pass
    
    def calculate_text_position(
        self, 
        xyxy: np.ndarray, 
        text: str, 
        position: Union[Position, str] = Position.TOP_LEFT
    ) -> Tuple[int, int]:
        """
        Calculate optimal text position relative to bounding box.
        
        Args:
            xyxy: Bounding box coordinates [x1, y1, x2, y2]
            text: Text to be positioned
            position: Position strategy
            
        Returns:
            (x, y) coordinates for text placement
        """
        x1, y1, x2, y2 = xyxy.astype(int)
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.text_scale, self.text_thickness
        )
        
        if isinstance(position, str):
            position = Position(position)
        
        # Calculate positions
        if position == Position.TOP_LEFT:
            return (x1, y1 - self.text_padding)
        elif position == Position.TOP_RIGHT:
            return (x2 - text_width, y1 - self.text_padding)
        elif position == Position.TOP_CENTER:
            return (x1 + (x2 - x1 - text_width) // 2, y1 - self.text_padding)
        elif position == Position.BOTTOM_LEFT:
            return (x1, y2 + text_height + self.text_padding)
        elif position == Position.BOTTOM_RIGHT:
            return (x2 - text_width, y2 + text_height + self.text_padding)
        elif position == Position.BOTTOM_CENTER:
            return (x1 + (x2 - x1 - text_width) // 2, y2 + text_height + self.text_padding)
        elif position == Position.CENTER:
            return (x1 + (x2 - x1 - text_width) // 2, y1 + (y2 - y1 + text_height) // 2)
        elif position == Position.CENTER_LEFT:
            return (x1, y1 + (y2 - y1 + text_height) // 2)
        elif position == Position.CENTER_RIGHT:
            return (x2 - text_width, y1 + (y2 - y1 + text_height) // 2)
        else:
            return (x1, y1 - self.text_padding)  # Default to top-left
    
    def draw_text_with_background(
        self, 
        scene: np.ndarray, 
        text: str, 
        position: Tuple[int, int],
        background_color: Optional[Tuple[int, int, int]] = None,
        opacity: float = 0.8
    ) -> np.ndarray:
        """
        Draw text with optional background for better visibility.
        
        Args:
            scene: Input image
            text: Text to draw
            position: Text position (x, y)
            background_color: Background color (if None, uses annotation color)
            opacity: Background opacity (0.0 to 1.0)
            
        Returns:
            Image with text drawn
        """
        if not text:
            return scene
            
        x, y = position
        bg_color = background_color if background_color else self.color
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, self.text_scale, self.text_thickness
        )
        
        # Draw background rectangle
        if opacity > 0:
            overlay = scene.copy()
            cv2.rectangle(
                overlay,
                (x - self.text_padding // 2, y - text_height - self.text_padding // 2),
                (x + text_width + self.text_padding // 2, y + self.text_padding // 2),
                bg_color,
                -1
            )
            scene = cv2.addWeighted(overlay, opacity, scene, 1 - opacity, 0)
        
        # Draw text
        cv2.putText(
            scene,
            text,
            (x, y),
            self.font,
            self.text_scale,
            self.text_color,
            self.text_thickness,
            cv2.LINE_AA
        )
        
        return scene
    
    def get_color_by_class(self, class_id: int) -> Tuple[int, int, int]:
        """Get consistent color for a given class ID."""
        return ColorPalette.get_color(class_id)
    
    def set_color(self, color: Union[Tuple[int, int, int], str]) -> None:
        """Update annotation color."""
        self.color = self._parse_color(color)
    
    def set_text_color(self, color: Tuple[int, int, int]) -> None:
        """Update text color."""
        self.text_color = color
    
    def set_thickness(self, thickness: int) -> None:
        """Update line thickness."""
        self.thickness = max(1, thickness)
    
    def set_text_scale(self, scale: float) -> None:
        """Update text scale."""
        self.text_scale = max(0.1, scale)
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        self.config.update(kwargs)
        
        # Apply configuration updates
        if 'color' in kwargs:
            self.set_color(kwargs['color'])
        if 'text_color' in kwargs:
            self.set_text_color(kwargs['text_color'])
        if 'thickness' in kwargs:
            self.set_thickness(kwargs['thickness'])
        if 'text_scale' in kwargs:
            self.set_text_scale(kwargs['text_scale'])
        if 'text_thickness' in kwargs:
            self.text_thickness = kwargs['text_thickness']
        if 'text_padding' in kwargs:
            self.text_padding = kwargs['text_padding']
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            'color': self.color,
            'text_color': self.text_color,
            'thickness': self.thickness,
            'text_scale': self.text_scale,
            'text_thickness': self.text_thickness,
            'text_padding': self.text_padding,
            **self.config
        }
    
    def copy(self) -> 'BaseAnnotator':
        """Create a copy of the annotator with same configuration."""
        return self.__class__(**self.get_config())
    
    def __repr__(self) -> str:
        """String representation of the annotator."""
        return f"{self.__class__.__name__}(color={self.color}, thickness={self.thickness})"


class AnnotatorConfig:
    """Configuration management utility for annotators."""
    
    def __init__(self, **defaults):
        """Initialize with default configuration."""
        self.defaults = defaults
        self.current = defaults.copy()
    
    def update(self, **kwargs):
        """Update configuration."""
        self.current.update(kwargs)
    
    def reset(self):
        """Reset to default configuration."""
        self.current = self.defaults.copy()
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.current.get(key, default)
    
    def __getitem__(self, key):
        """Dictionary-style access."""
        return self.current[key]
    
    def __setitem__(self, key, value):
        """Dictionary-style setting."""
        self.current[key] = value


# Advanced Utility Functions
def calculate_optimal_text_color(background_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Calculate optimal text color based on background color."""
    b, g, r = background_color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return ColorPalette.WHITE if luminance < 128 else ColorPalette.BLACK


def blend_colors(
    color1: Tuple[int, int, int], 
    color2: Tuple[int, int, int], 
    ratio: float = 0.5
) -> Tuple[int, int, int]:
    """Blend two colors with given ratio."""
    ratio = max(0.0, min(1.0, ratio))
    blended = tuple(
        int(c1 * (1 - ratio) + c2 * ratio)
        for c1, c2 in zip(color1, color2)
    )
    return blended


def adjust_color_brightness(
    color: Tuple[int, int, int], 
    factor: float = 1.2
) -> Tuple[int, int, int]:
    """Adjust color brightness by given factor."""
    adjusted = tuple(
        max(0, min(255, int(c * factor)))
        for c in color
    )
    return adjusted


# Additional Color Utilities
def generate_color_gradient(
    start_color: Tuple[int, int, int], 
    end_color: Tuple[int, int, int], 
    steps: int
) -> List[Tuple[int, int, int]]:
    """Generate color gradient between two colors."""
    if steps < 2:
        return [start_color, end_color]
    
    gradient = []
    for i in range(steps):
        ratio = i / (steps - 1)
        color = blend_colors(start_color, end_color, ratio)
        gradient.append(color)
    
    return gradient


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert hexadecimal color to BGR format."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])  # RGB to BGR


def bgr_to_hex(bgr_color: Tuple[int, int, int]) -> str:
    """Convert BGR color to hexadecimal format."""
    r, g, b = bgr_color[2], bgr_color[1], bgr_color[0]  # BGR to RGB
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


class ColorValidator:
    """Utility class for color validation and normalization."""
    
    @staticmethod
    def validate_color(color: Any) -> bool:
        """Validate if input is a valid color format."""
        if isinstance(color, (tuple, list)) and len(color) == 3:
            return all(isinstance(c, int) and 0 <= c <= 255 for c in color)
        
        if isinstance(color, str):
            if color.lower() in ['red', 'green', 'blue', 'yellow', 'cyan', 
                               'magenta', 'white', 'black']:
                return True
            
            # Check hex format
            hex_pattern = color.lstrip('#')
            if len(hex_pattern) == 6:
                try:
                    int(hex_pattern, 16)
                    return True
                except ValueError:
                    pass
        
        return False
    
    @staticmethod
    def normalize_color(color: Any) -> Tuple[int, int, int]:
        """Normalize color to BGR tuple format."""
        if isinstance(color, str):
            if color.startswith('#') or len(color) == 6:
                return hex_to_bgr(color)
            else:
                color_map = {
                    'red': ColorPalette.RED,
                    'green': ColorPalette.GREEN,
                    'blue': ColorPalette.BLUE,
                    'yellow': ColorPalette.YELLOW,
                    'cyan': ColorPalette.CYAN,
                    'magenta': ColorPalette.MAGENTA,
                    'white': ColorPalette.WHITE,
                    'black': ColorPalette.BLACK,
                }
                return color_map.get(color.lower(), ColorPalette.RED)
        
        elif isinstance(color, (tuple, list)) and len(color) == 3:
            return tuple(max(0, min(255, int(c))) for c in color)
        
        else:
            warnings.warn(f"Invalid color format: {color}. Using default red.")
            return ColorPalette.RED
