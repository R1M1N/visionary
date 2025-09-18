
import cv2
import numpy as np

class BackgroundOverlayAnnotator:
    """Annotator for applying background overlays and effects."""

    def __init__(self, overlay_color=(0, 0, 0), opacity=0.5):
        """
        Args:
            overlay_color (tuple): Color of the overlay in BGR format
            opacity (float): Opacity of the overlay (0.0 to 1.0)
        """
        self.overlay_color = overlay_color
        self.opacity = opacity

    def annotate(self, image, overlays):
        """
        Apply background overlays.

        Args:
            image (np.ndarray): Input image
            overlays (list): List of overlay configurations. Each should be dict with:
                           - 'type': 'full', 'region', or 'mask'
                           - 'color': overlay color (optional, uses default)
                           - 'opacity': overlay opacity (optional, uses default)
                           - 'region': bounding box [x1,y1,x2,y2] (for 'region' type)
                           - 'mask': binary mask array (for 'mask' type)

        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = image.copy()

        for overlay_info in overlays:
            overlay_type = overlay_info.get('type', 'full')
            color = overlay_info.get('color', self.overlay_color)
            opacity = overlay_info.get('opacity', self.opacity)

            if overlay_type == 'full':
                # Full image overlay
                overlay = np.full_like(annotated_image, color, dtype=np.uint8)
                annotated_image = cv2.addWeighted(annotated_image, 1 - opacity, 
                                                overlay, opacity, 0)

            elif overlay_type == 'region':
                # Regional overlay
                region = overlay_info['region']
                x1, y1, x2, y2 = map(int, region)
                overlay = annotated_image.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                annotated_image = cv2.addWeighted(annotated_image, 1 - opacity, 
                                                overlay, opacity, 0)

            elif overlay_type == 'mask':
                # Mask-based overlay
                mask = overlay_info['mask'].astype(bool)
                overlay = annotated_image.copy()
                overlay[mask] = color
                annotated_image = cv2.addWeighted(annotated_image, 1 - opacity, 
                                                overlay, opacity, 0)

        return annotated_image
