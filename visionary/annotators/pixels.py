
import cv2
import numpy as np

class PixelateAnnotator:
    """Annotator for applying pixelation effects to specific regions."""

    def __init__(self, pixelation_factor=10):
        """
        Args:
            pixelation_factor (int): Factor by which to pixelate (higher = more pixelated)
        """
        self.pixelation_factor = pixelation_factor

    def annotate(self, image, pixelate_regions):
        """
        Apply pixelation effects to regions.

        Args:
            image (np.ndarray): Input image
            pixelate_regions (list): List of regions to pixelate. Each should be dict with:
                                   - 'region': bounding box [x1,y1,x2,y2] or 'mask': binary mask
                                   - 'factor': pixelation factor (optional, overrides default)

        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = image.copy()

        for pixelate_info in pixelate_regions:
            factor = pixelate_info.get('factor', self.pixelation_factor)

            if 'region' in pixelate_info:
                # Regional pixelation
                x1, y1, x2, y2 = map(int, pixelate_info['region'])
                roi = annotated_image[y1:y2, x1:x2]

                # Get original dimensions
                height, width = roi.shape[:2]

                # Calculate new dimensions
                new_width = max(1, width // factor)
                new_height = max(1, height // factor)

                # Resize down then back up for pixelation effect
                small_roi = cv2.resize(roi, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                pixelated_roi = cv2.resize(small_roi, (width, height), interpolation=cv2.INTER_NEAREST)

                annotated_image[y1:y2, x1:x2] = pixelated_roi

            elif 'mask' in pixelate_info:
                # Mask-based pixelation
                mask = pixelate_info['mask'].astype(bool)

                # Create pixelated version of entire image
                height, width = annotated_image.shape[:2]
                new_width = max(1, width // factor)
                new_height = max(1, height // factor)

                small_image = cv2.resize(annotated_image, (new_width, new_height), 
                                       interpolation=cv2.INTER_LINEAR)
                pixelated_image = cv2.resize(small_image, (width, height), 
                                           interpolation=cv2.INTER_NEAREST)

                # Apply only to masked regions
                annotated_image[mask] = pixelated_image[mask]

        return annotated_image
