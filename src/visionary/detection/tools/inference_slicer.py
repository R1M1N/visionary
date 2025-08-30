"""
Inference Slicer for Visionary

Implements SAHI (Slicing Adaptive Inference) functionality
with configurable slice dimensions, overlap, and parallel processing support.
"""

import math
from typing import List, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

class InferenceSlicer:
    def __init__(self, 
                 slice_width: int, 
                 slice_height: int, 
                 overlap_ratio: float = 0.2, 
                 max_workers: int = 4):
        """
        Initialize InferenceSlicer with slice size, overlap ratio, and parallel workers.
        Args:
            slice_width: Width of each slice in pixels
            slice_height: Height of each slice in pixels
            overlap_ratio: Overlap ratio between slices (0-1)
            max_workers: Number of parallel workers for processing
        """
        self.slice_width = slice_width
        self.slice_height = slice_height
        self.overlap_ratio = overlap_ratio
        self.max_workers = max_workers

    def get_slices(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Generate slice bounding boxes covering the entire image with overlaps.
        Returns list of slices as (x1, y1, x2, y2) tuples.
        """
        img_height, img_width = image.shape[:2]
        stride_x = int(self.slice_width * (1 - self.overlap_ratio))
        stride_y = int(self.slice_height * (1 - self.overlap_ratio))

        slices = []
        for y in range(0, img_height, stride_y):
            y1 = y
            y2 = min(y + self.slice_height, img_height)
            for x in range(0, img_width, stride_x):
                x1 = x
                x2 = min(x + self.slice_width, img_width)
                slices.append((x1, y1, x2, y2))
        return slices

    def slice_and_infer(self, 
                        image: np.ndarray, 
                        infer_function: Callable[[np.ndarray], List],
                        postprocess_function: Callable[[List], List] = None) -> List:
        """
        Perform sliced inference on the image.
        Args:
            image: Input image to slice and run inference on
            infer_function: Function that runs inference on a single slice
            postprocess_function: Optional function to postprocess results from slices
        Returns combined inference results.
        """
        slices = self.get_slices(image)

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for bbox in slices:
                x1, y1, x2, y2 = bbox
                img_slice = image[y1:y2, x1:x2]
                futures.append(executor.submit(infer_function, img_slice))

            for future in as_completed(futures):
                res = future.result()
                if postprocess_function:
                    res = postprocess_function(res)
                results.extend(res)

        return results
