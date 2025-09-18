
import cv2
import numpy as np

class EllipseAnnotator:
    """Annotator for drawing ellipses on images."""

    def __init__(self, color=(0, 0, 255), thickness=2, line_type=cv2.LINE_AA):
        """
        Args:
            color (tuple): Color of the ellipse in BGR format
            thickness (int): Thickness of the ellipse line (-1 for filled)
            line_type: OpenCV line type
        """
        self.color = color
        self.thickness = thickness
        self.line_type = line_type

    def annotate(self, image, ellipses):
        """
        Draw ellipses on the image.

        Args:
            image (np.ndarray): Input image
            ellipses (list): List of ellipses, where each ellipse is dict with:
                           'center', 'axes', 'angle' or tuple (center, axes, angle)

        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = image.copy()

        for ellipse in ellipses:
            if isinstance(ellipse, dict):
                center = tuple(map(int, ellipse['center']))
                axes = tuple(map(int, ellipse['axes']))
                angle = ellipse.get('angle', 0)
            else:
                # Assume tuple format ((center_x, center_y), (axis1, axis2), angle)
                center = tuple(map(int, ellipse[0]))
                axes = tuple(map(int, ellipse[1]))
                angle = ellipse[2] if len(ellipse) > 2 else 0

            cv2.ellipse(annotated_image, center, axes, angle, 0, 360, 
                       self.color, self.thickness, self.line_type)

        return annotated_image
