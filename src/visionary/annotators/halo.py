
import cv2
import numpy as np

class HaloAnnotator:
    """Annotator for drawing halos/outlines around objects."""

    def __init__(self, halo_color=(255, 255, 255), halo_thickness=3, inner_color=(0, 0, 0), inner_thickness=1):
        """
        Args:
            halo_color (tuple): Color of the outer halo in BGR format
            halo_thickness (int): Thickness of the halo outline
            inner_color (tuple): Color of the inner line in BGR format
            inner_thickness (int): Thickness of the inner line
        """
        self.halo_color = halo_color
        self.halo_thickness = halo_thickness
        self.inner_color = inner_color
        self.inner_thickness = inner_thickness

    def annotate(self, image, objects):
        """
        Draw halos around objects.

        Args:
            image (np.ndarray): Input image
            objects (list): List of objects to add halos to. Can be:
                          - Bounding boxes [x1, y1, x2, y2]
                          - Polygons as list of points
                          - Dicts with 'bbox', 'polygon', or 'contour'

        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = image.copy()

        for obj in objects:
            if isinstance(obj, dict):
                if 'bbox' in obj:
                    # Draw halo around bounding box
                    x1, y1, x2, y2 = map(int, obj['bbox'])
                    # Draw outer halo
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), 
                                self.halo_color, self.halo_thickness)
                    # Draw inner line
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), 
                                self.inner_color, self.inner_thickness)

                elif 'polygon' in obj or 'contour' in obj:
                    # Draw halo around polygon
                    points_key = 'polygon' if 'polygon' in obj else 'contour'
                    points = np.array(obj[points_key], dtype=np.int32)
                    if points.ndim == 2:
                        points = points.reshape((-1, 1, 2))

                    # Draw outer halo
                    cv2.polylines(annotated_image, [points], True, 
                                self.halo_color, self.halo_thickness)
                    # Draw inner line
                    cv2.polylines(annotated_image, [points], True, 
                                self.inner_color, self.inner_thickness)

            elif len(obj) == 4:
                # Assume bounding box format
                x1, y1, x2, y2 = map(int, obj)
                # Draw outer halo
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), 
                            self.halo_color, self.halo_thickness)
                # Draw inner line
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), 
                            self.inner_color, self.inner_thickness)

            else:
                # Assume polygon format
                points = np.array(obj, dtype=np.int32)
                if points.ndim == 2:
                    points = points.reshape((-1, 1, 2))

                # Draw outer halo
                cv2.polylines(annotated_image, [points], True, 
                            self.halo_color, self.halo_thickness)
                # Draw inner line
                cv2.polylines(annotated_image, [points], True, 
                            self.inner_color, self.inner_thickness)

        return annotated_image
