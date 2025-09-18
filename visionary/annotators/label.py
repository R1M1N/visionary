
import cv2

class LabelAnnotator:
    """A basic implementation of a label annotator without external dependencies."""

    def __init__(self, label_color=(255, 0, 0), font_scale=0.5, font_color=(255, 255, 255), thickness=2):
        self.label_color = label_color
        self.font_scale = font_scale
        self.font_color = font_color
        self.thickness = thickness

    def annotate(self, image, detections):
        """
        Annotate the image with bounding boxes and labels.

        Args:
            image (numpy.ndarray): Image to annotate.
            detections (list): List of detection dicts with keys: 'bbox' and 'label'.
                'bbox' is [x_min, y_min, x_max, y_max]
                'label' is a string

        Returns:
            Annotated image (numpy.ndarray).
        """
        import cv2
        annotated_image = image.copy()
        for det in detections:
            bbox = det.get('bbox', None)
            label = det.get('label', '')
            if bbox is None:
                continue

            x_min, y_min, x_max, y_max = map(int, bbox)
            # Draw rectangle
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), self.label_color, self.thickness)

            # Calculate text size
            ((text_width, text_height), baseline) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.thickness)

            # Draw filled rectangle for text background
            cv2.rectangle(annotated_image, (x_min, y_min - text_height - baseline - 4), 
                          (x_min + text_width, y_min), self.label_color, -1)

            # Put label text
            cv2.putText(annotated_image, label, (x_min, y_min - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 
                        self.font_scale, self.font_color, self.thickness, lineType=cv2.LINE_AA)

        return annotated_image
