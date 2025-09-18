
"""
Geometry Utilities for Visionary

Provides coordinate transformations,
geometric calculations, and perspective corrections.
"""

import numpy as np
import cv2
from typing import Tuple, List

class GeometryUtils:
    @staticmethod
    def xyxy_to_xywh(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """
        Convert bounding box from [x1, y1, x2, y2] to [x, y, w, h] format.
        """
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        return (x1, y1, w, h)

    @staticmethod
    def xywh_to_xyxy(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """
        Convert bounding box from [x, y, w, h] to [x1, y1, x2, y2] format.
        """
        x, y, w, h = box
        x2 = x + w
        y2 = y + h
        return (x, y, x2, y2)

    @staticmethod
    def point_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Apply a transformation matrix to points (Nx2 array).
        """
        assert points.shape[1] == 2
        num_points = points.shape[0]
        homog_points = np.hstack((points, np.ones((num_points, 1))))  # Nx3
        transformed = homog_points @ matrix.T  # Nx3
        # Normalize by last coordinate
        transformed = transformed[:, :2] / transformed[:, 2:3]
        return transformed

    @staticmethod
    def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
        """
        Compute intersection over union between two boxes (xyxy format).
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_area = max(0, xB - xA) * max(0, yB - yA)
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = inter_area / float(boxA_area + boxB_area - inter_area + 1e-16)
        return iou

    @staticmethod
    def perspective_correction(image: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
        """
        Apply perspective correction (homography) to an image.
        Args:
            image: Input image
            src_points: Four source points (Nx2)
            dst_points: Four destination points (Nx2)
            output_size: Output image size (width, height)
        Returns:
            Warped image with corrected perspective
        """
        matrix = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points.astype(np.float32))
        warped_image = cv2.warpPerspective(image, matrix, output_size)
        return warped_image
