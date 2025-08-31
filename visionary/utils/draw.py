
"""
Drawing Utilities for Visionary

Provides custom drawing functions, color management,
and text rendering utilities.
"""

import cv2
from typing import Tuple, List

class DrawUtils:
    @staticmethod
    def draw_rectangle(image, top_left: Tuple[int, int], bottom_right: Tuple[int, int], 
                       color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> None:
        """
        Draw rectangle on image.
        """
        cv2.rectangle(image, top_left, bottom_right, color, thickness)

    @staticmethod
    def draw_circle(image, center: Tuple[int, int], radius: int = 5, 
                    color: Tuple[int, int, int] = (0, 255, 0), thickness: int = -1) -> None:
        """
        Draw circle on image.
        """
        cv2.circle(image, center, radius, color, thickness)

    @staticmethod
    def draw_line(image, pt1: Tuple[int, int], pt2: Tuple[int, int], 
                  color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> None:
        """
        Draw line on image.
        """
        cv2.line(image, pt1, pt2, color, thickness)

    @staticmethod
    def put_text(image, text: str, org: Tuple[int, int], 
                 font: int = cv2.FONT_HERSHEY_SIMPLEX, font_scale: float = 0.6,
                 color: Tuple[int, int, int] = (255, 255, 255), thickness: int = 1) -> None:
        """
        Render text on image.
        """
        cv2.putText(image, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

class ColorManager:
    COLORS = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 0),    # Maroon
        (0, 128, 0),    # Dark Green
        (0, 0, 128),    # Navy
        (128, 128, 0),  # Olive
    ]

    @staticmethod
    def get_color(index: int) -> Tuple[int, int, int]:
        """
        Get color by index, cycling through predefined colors.
        """
        return ColorManager.COLORS[index % len(ColorManager.COLORS)]
