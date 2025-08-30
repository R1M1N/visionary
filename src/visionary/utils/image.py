
"""
Image Utilities for Visionary

Provides image loading, preprocessing, batch processing,
 and image format conversion functionalities.
"""

import cv2
import numpy as np
from pathlib import Path  # Add this import
from typing import List, Callable, Optional  # Add List to this import


class ImageUtils:
    @staticmethod
    def load_image(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
        """
        Load an image from disk.
        Args:
            path: Image file path
            flags: OpenCV loading flags
        Returns:
            Loaded image as ndarray
        """
        image = cv2.imread(path, flags)
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return image

    @staticmethod
    def preprocess_image(image: np.ndarray, size: Optional[tuple] = None, 
                         normalize: bool = False, mean: tuple = (0, 0, 0), std: tuple = (1, 1, 1)) -> np.ndarray:
        """
        Preprocess image with optional resize and normalization.
        Args:
            image: Input image ndarray
            size: Target resize dimensions (width, height)
            normalize: Whether to normalize image
            mean: Mean for normalization
            std: Standard deviation for normalization
        Returns:
            Preprocessed image ndarray
        """
        processed = image.copy()
        if size is not None:
            processed = cv2.resize(processed, size, interpolation=cv2.INTER_AREA)
        if normalize:
            processed = processed.astype(np.float32) / 255.0
            for i in range(3):
                processed[..., i] = (processed[..., i] - mean[i]) / std[i]
        return processed

    @staticmethod
    def batch_process(images: List[np.ndarray], process_function: Callable[[np.ndarray], np.ndarray], batch_size: int = 8) -> List[np.ndarray]:
        """
        Process list of images in batches.
        Args:
            images: List of image ndarrays
            process_function: Processing function applied to batch
            batch_size: Batch size
        Returns:
            List of processed image ndarrays
        """
        processed_images = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            processed_batch = [process_function(img) for img in batch]
            processed_images.extend(processed_batch)
        return processed_images

    @staticmethod
    def convert_format(image: np.ndarray, conversion_code: int) -> np.ndarray:
        """
        Convert image format using OpenCV conversion codes.
        Args:
            image: Input image ndarray
            conversion_code: OpenCV color conversion code
        Returns:
            Converted image ndarray
        """
        return cv2.cvtColor(image, conversion_code)

class ImageSink:
    """
    Save image frames to disk with a sequential naming scheme.
    """
    def __init__(self, output_dir: str, prefix: str = 'frame_', 
                 extension: str = '.png', start_index: int = 0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.extension = extension
        self.index = start_index

    def write(self, image: np.ndarray):
        filename = f"{self.prefix}{self.index:06d}{self.extension}"
        path = self.output_dir / filename
        cv2.imwrite(str(path), image)
        self.index += 1

    def reset(self, start_index: int = 0):
        self.index = start_index

def list_files_with_extensions(directory: str, extensions: List[str]) -> List[Path]:
    """
    Convenience wrapper to list only files matching the given extensions.
    """
    from .file import FileUtils
    return FileUtils.list_files(directory, extensions)