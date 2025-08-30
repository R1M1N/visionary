
import cv2
import numpy as np

class ColorAnnotator:
    """Annotator for applying color overlays to specific regions."""

    def __init__(self, opacity=0.3):
        """
        Args:
            opacity (float): Opacity of the color overlay (0.0 to 1.0)
        """
        self.opacity = opacity

    def annotate(self, image, color_regions):
        """
        Apply color overlays to regions.

        Args:
            image (np.ndarray): Input image
            color_regions (list): List of regions with colors. Each item should be dict with:
                                - 'region': bounding box [x1, y1, x2, y2] or mask array
                                - 'color': color tuple (B, G, R)
                                - 'opacity': optional override opacity

        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = image.copy()

        for item in color_regions:
            color = item['color']
            region = item['region']
            opacity = item.get('opacity', self.opacity)

            if isinstance(region, (list, tuple)) and len(region) == 4:
                # Bounding box region
                x1, y1, x2, y2 = map(int, region)
                # Create colored overlay
                overlay = annotated_image.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                # Blend with original
                annotated_image = cv2.addWeighted(annotated_image, 1 - opacity, 
                                                overlay, opacity, 0)

            elif isinstance(region, np.ndarray):
                # Mask region
                mask = region.astype(bool)
                # Create colored overlay
                overlay = annotated_image.copy()
                overlay[mask] = color
                # Blend with original
                annotated_image = cv2.addWeighted(annotated_image, 1 - opacity, 
                                                overlay, opacity, 0)

        return annotated_image
