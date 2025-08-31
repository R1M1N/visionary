
import cv2
import numpy as np

class TriangleAnnotator:
    """Annotator for drawing triangles on images."""

    def __init__(self, color=(255, 255, 0), thickness=2, line_type=cv2.LINE_AA):
        """
        Args:
            color (tuple): Color of the triangle in BGR format
            thickness (int): Thickness of the triangle lines (-1 for filled)
            line_type: OpenCV line type
        """
        self.color = color
        self.thickness = thickness
        self.line_type = line_type

    def annotate(self, image, triangles):
        """
        Draw triangles on the image.

        Args:
            image (np.ndarray): Input image
            triangles (list): List of triangles, where each triangle is list of 3 points
                            [[x1,y1], [x2,y2], [x3,y3]] or dict with 'points'

        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = image.copy()

        for triangle in triangles:
            if isinstance(triangle, dict):
                points = np.array(triangle['points'], dtype=np.int32)
            else:
                points = np.array(triangle, dtype=np.int32)

            if len(points) != 3:
                continue

            if self.thickness == -1:
                # Filled triangle
                cv2.fillPoly(annotated_image, [points], self.color)
            else:
                # Triangle outline
                cv2.polylines(annotated_image, [points], True, self.color, 
                             self.thickness, self.line_type)

        return annotated_image
