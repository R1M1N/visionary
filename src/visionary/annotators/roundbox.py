
import cv2
import numpy as np

class RoundBoxAnnotator:
    """Annotator for drawing rounded rectangles/boxes on images."""

    def __init__(self, color=(0, 255, 255), thickness=2, corner_radius=10):
        """
        Args:
            color (tuple): Color of the rounded box in BGR format
            thickness (int): Thickness of the box lines (-1 for filled)
            corner_radius (int): Radius of the rounded corners
        """
        self.color = color
        self.thickness = thickness
        self.corner_radius = corner_radius

    def _draw_rounded_rectangle(self, image, pt1, pt2, radius, color, thickness):
        """Helper method to draw a rounded rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2

        # Draw the main rectangle body
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, thickness)

        # Draw the corner arcs
        cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
        cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)

    def annotate(self, image, boxes):
        """
        Draw rounded boxes on the image.

        Args:
            image (np.ndarray): Input image
            boxes (list): List of boxes [x1, y1, x2, y2] or dicts with 'bbox' and optional 'radius'

        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = image.copy()

        for box in boxes:
            if isinstance(box, dict):
                x1, y1, x2, y2 = map(int, box['bbox'])
                radius = box.get('radius', self.corner_radius)
            else:
                x1, y1, x2, y2 = map(int, box)
                radius = self.corner_radius

            # Ensure valid rectangle
            if x2 <= x1 or y2 <= y1:
                continue

            # Limit radius to half the smaller dimension
            radius = min(radius, min(x2-x1, y2-y1) // 2)

            self._draw_rounded_rectangle(annotated_image, (x1, y1), (x2, y2), 
                                       radius, self.color, self.thickness)

        return annotated_image
