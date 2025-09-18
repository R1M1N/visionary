
import cv2
import numpy as np

class CircleAnnotator:
    """Annotator for drawing circles on images."""

    def __init__(self, color=(0, 255, 0), thickness=2, line_type=cv2.LINE_AA):
        """
        Args:
            color (tuple): Color of the circle in BGR format
            thickness (int): Thickness of the circle line (-1 for filled)
            line_type: OpenCV line type
        """
        self.color = color
        self.thickness = thickness
        self.line_type = line_type

    def annotate(self, image, circles):
        """
        Draw circles on the image.

        Args:
            image (np.ndarray): Input image
            circles (list): List of circles, where each circle is dict with 'center' and 'radius'
                           or tuple (center_x, center_y, radius)

        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = image.copy()

        for circle in circles:
            if isinstance(circle, dict):
                center = tuple(map(int, circle['center']))
                radius = int(circle['radius'])
            else:
                # Assume tuple format (x, y, radius)
                center = (int(circle[0]), int(circle[1]))
                radius = int(circle[2])

            cv2.circle(annotated_image, center, radius, self.color, self.thickness, self.line_type)

        return annotated_image
