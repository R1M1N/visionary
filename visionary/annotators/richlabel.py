
import cv2
import numpy as np

class RichLabelAnnotator:
    """Annotator for drawing rich labels with background, borders, and styling."""

    def __init__(self, text_color=(255, 255, 255), bg_color=(0, 0, 0), 
                 border_color=None, border_thickness=1, padding=5, 
                 font_scale=0.6, font_thickness=1):
        """
        Args:
            text_color (tuple): Color of the text in BGR format
            bg_color (tuple): Background color of the label in BGR format
            border_color (tuple): Border color (None for no border)
            border_thickness (int): Thickness of the border
            padding (int): Padding around the text
            font_scale (float): Scale of the font
            font_thickness (int): Thickness of the font
        """
        self.text_color = text_color
        self.bg_color = bg_color
        self.border_color = border_color
        self.border_thickness = border_thickness
        self.padding = padding
        self.font_scale = font_scale
        self.font_thickness = font_thickness

    def annotate(self, image, rich_labels):
        """
        Draw rich labels on the image.

        Args:
            image (np.ndarray): Input image
            rich_labels (list): List of rich labels. Each should be dict with:
                              - 'text': text to display
                              - 'position': (x, y) position
                              - optional styling overrides

        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = image.copy()

        for label_info in rich_labels:
            text = label_info['text']
            x, y = map(int, label_info['position'])

            # Get styling (use defaults if not specified)
            text_color = label_info.get('text_color', self.text_color)
            bg_color = label_info.get('bg_color', self.bg_color)
            border_color = label_info.get('border_color', self.border_color)
            border_thickness = label_info.get('border_thickness', self.border_thickness)
            padding = label_info.get('padding', self.padding)
            font_scale = label_info.get('font_scale', self.font_scale)
            font_thickness = label_info.get('font_thickness', self.font_thickness)

            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            # Calculate label rectangle
            label_x1 = x - padding
            label_y1 = y - text_height - padding - baseline
            label_x2 = x + text_width + padding
            label_y2 = y + padding

            # Draw background
            cv2.rectangle(annotated_image, (label_x1, label_y1), (label_x2, label_y2), 
                         bg_color, -1)

            # Draw border if specified
            if border_color is not None:
                cv2.rectangle(annotated_image, (label_x1, label_y1), (label_x2, label_y2), 
                             border_color, border_thickness)

            # Draw text
            cv2.putText(annotated_image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, text_color, font_thickness, cv2.LINE_AA)

        return annotated_image
