
import cv2
import numpy as np

class PolygonAnnotator:
    """Annotator for drawing polygons on images."""

    def __init__(self, color=(128, 0, 128), thickness=2, line_type=cv2.LINE_AA):
        """
        Args:
            color (tuple): Color of the polygon in BGR format
            thickness (int): Thickness of the polygon lines (-1 for filled)
            line_type: OpenCV line type
        """
        self.color = color
        self.thickness = thickness
        self.line_type = line_type

    def annotate(self, image, polygons):
        """
        Draw polygons on the image.

        Args:
            image (np.ndarray): Input image
            polygons (list): List of polygons, where each polygon is:
                           - List of points [[x1,y1], [x2,y2], ...]
                           - Dict with 'points' key
                           - Numpy array of shape (n, 2)

        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = image.copy()

        for polygon in polygons:
            if isinstance(polygon, dict):
                points = np.array(polygon['points'], dtype=np.int32)
            else:
                points = np.array(polygon, dtype=np.int32)

            if len(points) < 3:
                continue

            # Reshape for OpenCV if needed
            if points.ndim == 2:
                points = points.reshape((-1, 1, 2))

            if self.thickness == -1:
                # Filled polygon
                cv2.fillPoly(annotated_image, [points], self.color, self.line_type)
            else:
                # Polygon outline
                cv2.polylines(annotated_image, [points], True, self.color, 
                             self.thickness, self.line_type)

        return annotated_image
