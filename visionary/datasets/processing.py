"""
Dataset Processing for Visionary

Provides batch processing, polygon simplification, RLE encoding/decoding,
and image preprocessing pipelines.
"""

from typing import List, Tuple, Callable, Iterable
import cv2
import numpy as np
from shapely.geometry import Polygon
import pycocotools.mask as mask_util

class DatasetProcessor:
    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size

    def batch_process(self, items: Iterable, process_function: Callable) -> List:
        """
        Process dataset items in batches.
        Args:
            items: Iterable of dataset items
            process_function: Function to process each batch
        Returns:
            List of processed results
        """
        results = []
        batch = []
        for item in items:
            batch.append(item)
            if len(batch) == self.batch_size:
                results.extend(process_function(batch))
                batch.clear()
        if batch:
            results.extend(process_function(batch))
        return results

    @staticmethod
    def simplify_polygon(polygon_points: List[Tuple[float, float]], tolerance: float = 1.0) -> List[Tuple[float, float]]:
        """
        Simplify polygon using Ramer-Douglas-Peucker algorithm via Shapely.
        Args:
            polygon_points: List of (x, y) points
            tolerance: Simplification tolerance
        Returns:
            Simplified polygon points
        """
        poly = Polygon(polygon_points)
        simplified = poly.simplify(tolerance, preserve_topology=True)
        if simplified.is_empty:
            return polygon_points
        return list(simplified.exterior.coords)

    @staticmethod
    def rle_encode(mask: np.ndarray) -> dict:
        """
        Encode binary mask using COCO RLE format.
        Args:
            mask: Binary mask ndarray
        Returns:
            RLE encoded dict
        """
        rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('utf-8')  # Convert bytes to string
        return rle

    @staticmethod
    def rle_decode(rle: dict) -> np.ndarray:
        """
        Decode RLE to binary mask.
        Args:
            rle: RLE encoded dict
        Returns:
            Binary mask ndarray
        """
        if isinstance(rle['counts'], str):
            rle['counts'] = rle['counts'].encode('utf-8')
        return mask_util.decode(rle)

    @staticmethod
    def preprocess_image(image: np.ndarray, size: Tuple[int, int], mean: Tuple[float, float, float] = (0, 0, 0), std: Tuple[float, float, float] = (1, 1, 1)) -> np.ndarray:
        """
        Resize and normalize image with mean and std per channel.
        Args:
            image: Input image
            size: (width, height) tuple
            mean: Mean for normalization
            std: Std dev for normalization
        Returns:
            Normalized image
        """
        resized = cv2.resize(image, size)
        normalized = resized.astype(np.float32) / 255.0
        for i in range(3):
            normalized[:, :, i] = (normalized[:, :, i] - mean[i]) / std[i]
        return normalized
