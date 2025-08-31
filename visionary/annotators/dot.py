
import cv2
import numpy as np

class DotAnnotator:
    """Annotator for drawing dots/points on images."""

    def __init__(self, color=(255, 0, 255), radius=3, thickness=-1):
        """
        Args:
            color (tuple): Color of the dots in BGR format
            radius (int): Radius of the dots
            thickness (int): Thickness of the dot outline (-1 for filled)
        """
        self.color = color
        self.radius = radius
        self.thickness = thickness

    def annotate(self, image, dots):
        """
        Draw dots on the image.

        Args:
            image (np.ndarray): Input image
            dots (list): List of dot positions [(x1,y1), (x2,y2), ...] or 
                        list of dicts with 'position' and optional 'radius'

        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = image.copy()

        for dot in dots:
            if isinstance(dot, dict):
                position = tuple(map(int, dot['position']))
                radius = dot.get('radius', self.radius)
            else:
                # Assume tuple format (x, y)
                position = tuple(map(int, dot))
                radius = self.radius

            cv2.circle(annotated_image, position, radius, self.color, self.thickness)

        return annotated_image
