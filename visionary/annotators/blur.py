
import cv2
import numpy as np

class BlurAnnotator:
    """Annotator for applying blur effects to specific regions."""

    def __init__(self, blur_strength=15):
        """
        Args:
            blur_strength (int): Strength of the blur effect (kernel size)
        """
        self.blur_strength = blur_strength

    def annotate(self, image, blur_regions):
        """
        Apply blur effects to regions.

        Args:
            image (np.ndarray): Input image
            blur_regions (list): List of regions to blur. Each should be dict with:
                               - 'region': bounding box [x1,y1,x2,y2] or 'mask': binary mask
                               - 'strength': blur strength (optional, overrides default)
                               - 'type': blur type ('gaussian', 'motion', 'median')

        Returns:
            np.ndarray: Annotated image
        """
        annotated_image = image.copy()

        for blur_info in blur_regions:
            strength = blur_info.get('strength', self.blur_strength)
            blur_type = blur_info.get('type', 'gaussian')

            # Ensure odd kernel size
            if strength % 2 == 0:
                strength += 1

            if 'region' in blur_info:
                # Regional blur
                x1, y1, x2, y2 = map(int, blur_info['region'])
                roi = annotated_image[y1:y2, x1:x2]

                if blur_type == 'gaussian':
                    blurred_roi = cv2.GaussianBlur(roi, (strength, strength), 0)
                elif blur_type == 'motion':
                    # Create motion blur kernel
                    kernel = np.zeros((strength, strength))
                    kernel[int((strength-1)/2), :] = np.ones(strength)
                    kernel = kernel / strength
                    blurred_roi = cv2.filter2D(roi, -1, kernel)
                elif blur_type == 'median':
                    blurred_roi = cv2.medianBlur(roi, strength)
                else:
                    blurred_roi = cv2.GaussianBlur(roi, (strength, strength), 0)

                annotated_image[y1:y2, x1:x2] = blurred_roi

            elif 'mask' in blur_info:
                # Mask-based blur
                mask = blur_info['mask'].astype(bool)

                if blur_type == 'gaussian':
                    blurred_image = cv2.GaussianBlur(annotated_image, (strength, strength), 0)
                elif blur_type == 'motion':
                    kernel = np.zeros((strength, strength))
                    kernel[int((strength-1)/2), :] = np.ones(strength)
                    kernel = kernel / strength
                    blurred_image = cv2.filter2D(annotated_image, -1, kernel)
                elif blur_type == 'median':
                    blurred_image = cv2.medianBlur(annotated_image, strength)
                else:
                    blurred_image = cv2.GaussianBlur(annotated_image, (strength, strength), 0)

                annotated_image[mask] = blurred_image[mask]

        return annotated_image
