
import cv2
import numpy as np

class PercentageBarAnnotator:
    """Annotator for drawing percentage bars/progress bars on images."""

    def __init__(self, bar_color=(0, 255, 0), bg_color=(128, 128, 128), 
                 text_color=(255, 255, 255), bar_height=20, show_text=True):
        """
        Args:
            bar_color (tuple): Color of the filled portion in BGR format
            bg_color (tuple): Color of the background bar in BGR format
            text_color (tuple): Color of the percentage text in BGR format
            bar_height (int): Height of the progress bar
            show_text (bool): Whether to show percentage text
        """
        self.bar_color = bar_color
        self.bg_color = bg_color
        self.text_color = text_color
        self.bar_height = bar_height
        self.show_text = show_text

    def annotate(self, image, percentage_bars):
        """
        Draw percentage bars on the image.

        Args:
            image (np.ndarray): Input image
            percentage_bars (list): List of percentage bars. Each should be dict with:
                                  - 'position': (x, y) top-left position
                                  - 'width': width of the bar
                                  - 'percentage': percentage value (0-100)
                                  - optional 'label': text label to show

        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = image.copy()

        for bar_info in percentage_bars:
            x, y = map(int, bar_info['position'])
            width = int(bar_info['width'])
            percentage = max(0, min(100, bar_info['percentage']))
            label = bar_info.get('label', f'{percentage:.1f}%')

            # Draw background bar
            cv2.rectangle(annotated_image, (x, y), (x + width, y + self.bar_height), 
                         self.bg_color, -1)

            # Calculate filled width
            filled_width = int(width * percentage / 100)

            # Draw filled portion
            if filled_width > 0:
                cv2.rectangle(annotated_image, (x, y), (x + filled_width, y + self.bar_height), 
                             self.bar_color, -1)

            # Draw border
            cv2.rectangle(annotated_image, (x, y), (x + width, y + self.bar_height), 
                         (0, 0, 0), 1)

            # Draw text if enabled
            if self.show_text:
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                text_x = x + (width - text_size[0]) // 2
                text_y = y + (self.bar_height + text_size[1]) // 2

                cv2.putText(annotated_image, label, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.text_color, 1, cv2.LINE_AA)

        return annotated_image
