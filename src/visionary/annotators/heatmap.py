
import cv2
import numpy as np

class HeatMapAnnotator:
    """Annotator for overlaying heatmaps on images."""

    def __init__(self, colormap=cv2.COLORMAP_JET, alpha=0.6):
        """
        Args:
            colormap: OpenCV colormap (e.g., cv2.COLORMAP_JET, cv2.COLORMAP_HOT)
            alpha (float): Opacity of the heatmap overlay
        """
        self.colormap = colormap
        self.alpha = alpha

    def annotate(self, image, heatmaps):
        """
        Overlay heatmaps on the image.

        Args:
            image (np.ndarray): Input image
            heatmaps (list): List of heatmap data. Each should be dict with:
                           - 'data': 2D numpy array with heat values (0-1 or 0-255)
                           - optional 'alpha': override alpha value
                           - optional 'colormap': override colormap

        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = image.copy()

        for heatmap_info in heatmaps:
            heat_data = heatmap_info['data']
            alpha = heatmap_info.get('alpha', self.alpha)
            colormap = heatmap_info.get('colormap', self.colormap)

            # Normalize heat data to 0-255 range
            if heat_data.max() <= 1.0:
                heat_data_normalized = (heat_data * 255).astype(np.uint8)
            else:
                heat_data_normalized = heat_data.astype(np.uint8)

            # Resize heatmap to match image size if necessary
            if heat_data_normalized.shape[:2] != annotated_image.shape[:2]:
                heat_data_normalized = cv2.resize(heat_data_normalized, 
                                                (annotated_image.shape[1], annotated_image.shape[0]))

            # Apply colormap
            heatmap_colored = cv2.applyColorMap(heat_data_normalized, colormap)

            # Blend with original image
            annotated_image = cv2.addWeighted(annotated_image, 1 - alpha, 
                                            heatmap_colored, alpha, 0)

        return annotated_image
