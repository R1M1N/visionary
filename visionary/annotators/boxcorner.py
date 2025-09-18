
import cv2
import numpy as np

class BoxCornerAnnotator:
    """Annotator for drawing corner markers on boxes."""

    def __init__(self, color=(255, 255, 255), thickness=3, corner_length=20):
        """
        Args:
            color (tuple): Color of the corner markers in BGR format
            thickness (int): Thickness of the corner lines
            corner_length (int): Length of each corner line
        """
        self.color = color
        self.thickness = thickness
        self.corner_length = corner_length

    def annotate(self, image, boxes):
        """
        Draw corner markers on boxes.

        Args:
            image (np.ndarray): Input image
            boxes (list): List of boxes [x1, y1, x2, y2] or dicts with 'bbox'

        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = image.copy()

        for box in boxes:
            if isinstance(box, dict):
                x1, y1, x2, y2 = map(int, box['bbox'])
                length = box.get('corner_length', self.corner_length)
            else:
                x1, y1, x2, y2 = map(int, box)
                length = self.corner_length

            # Top-left corner
            cv2.line(annotated_image, (x1, y1), (x1 + length, y1), self.color, self.thickness)
            cv2.line(annotated_image, (x1, y1), (x1, y1 + length), self.color, self.thickness)

            # Top-right corner
            cv2.line(annotated_image, (x2, y1), (x2 - length, y1), self.color, self.thickness)
            cv2.line(annotated_image, (x2, y1), (x2, y1 + length), self.color, self.thickness)

            # Bottom-right corner
            cv2.line(annotated_image, (x2, y2), (x2 - length, y2), self.color, self.thickness)
            cv2.line(annotated_image, (x2, y2), (x2, y2 - length), self.color, self.thickness)

            # Bottom-left corner
            cv2.line(annotated_image, (x1, y2), (x1 + length, y2), self.color, self.thickness)
            cv2.line(annotated_image, (x1, y2), (x1, y2 - length), self.color, self.thickness)

        return annotated_image
