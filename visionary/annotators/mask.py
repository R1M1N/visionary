
import numpy as np
import cv2

class MaskAnnotator:
    """Simple mask annotator to overlay binary masks on images."""

    def __init__(self, mask_color=(0, 255, 0), alpha=0.5):
        """Initialize the annotator with mask color and transparency."""
        self.mask_color = mask_color
        self.alpha = alpha

    def annotate(self, image, masks):
        """Overlay masks on the input image.
        Args:
            image (numpy.ndarray): The input image.
            masks (list of np.ndarray): List of boolean arrays representing masks.
        Returns:
            np.ndarray: Image with masks overlayed.
        """
        annotated_image = image.copy()
        for mask in masks:
            # Create a colored mask
            colored_mask = np.zeros_like(annotated_image, dtype=np.uint8)
            colored_mask[mask] = self.mask_color
            # Blend the colored mask with the image
            annotated_image = cv2.addWeighted(annotated_image, 1 - self.alpha, colored_mask, self.alpha, 0)
        return annotated_image
