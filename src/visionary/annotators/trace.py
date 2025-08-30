
import cv2
import numpy as np

class TraceAnnotator:
    """Annotator for drawing traces/paths on images."""

    def __init__(self, color=(255, 0, 0), thickness=2, line_type=cv2.LINE_AA):
        """
        Args:
            color (tuple): Color of the trace line in BGR format
            thickness (int): Thickness of the line
            line_type: OpenCV line type
        """
        self.color = color
        self.thickness = thickness
        self.line_type = line_type

    def annotate(self, image, traces):
        """
        Draw traces on the image.

        Args:
            image (np.ndarray): Input image
            traces (list): List of traces, where each trace is a list of points [(x1,y1), (x2,y2), ...]

        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = image.copy()

        for trace in traces:
            if len(trace) < 2:
                continue

            # Convert points to numpy array
            points = np.array(trace, dtype=np.int32)

            # Draw lines connecting consecutive points
            for i in range(len(points) - 1):
                cv2.line(annotated_image, 
                        tuple(points[i]), 
                        tuple(points[i + 1]), 
                        self.color, 
                        self.thickness, 
                        self.line_type)

        return annotated_image
