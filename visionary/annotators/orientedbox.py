
import cv2
import numpy as np

class OrientedBoxAnnotator:
    """Annotator for drawing oriented/rotated bounding boxes on images."""

    def __init__(self, color=(255, 165, 0), thickness=2, line_type=cv2.LINE_AA):
        """
        Args:
            color (tuple): Color of the oriented box in BGR format
            thickness (int): Thickness of the box lines
            line_type: OpenCV line type
        """
        self.color = color
        self.thickness = thickness
        self.line_type = line_type

    def annotate(self, image, oriented_boxes):
        """
        Draw oriented boxes on the image.

        Args:
            image (np.ndarray): Input image
            oriented_boxes (list): List of oriented boxes, each containing:
                                  - 'center': (x, y) center point
                                  - 'size': (width, height)
                                  - 'angle': rotation angle in degrees
                                  OR 4 corner points as array

        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = image.copy()

        for box in oriented_boxes:
            if isinstance(box, dict) and 'center' in box:
                # RotatedRect format
                center = box['center']
                size = box['size']
                angle = box['angle']

                # Create rotated rectangle
                rect = (center, size, angle)
                box_points = cv2.boxPoints(rect)
                box_points = np.int0(box_points)
            else:
                # Assume 4 corner points
                box_points = np.array(box, dtype=np.int32)

            # Draw the oriented box
            cv2.drawContours(annotated_image, [box_points], -1, self.color, 
                           self.thickness, self.line_type)

        return annotated_image
