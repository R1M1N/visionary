
"""
Oriented Bounding Box (OBB) Support for Visionary

Provides OBB-specific data structures, annotators,
OBB-aware NMS algorithms, and format converters.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional

class OrientedBoundingBox:
    def __init__(self, cx: float, cy: float, width: float, height: float, angle: float):
        """
        Initialize oriented bounding box.
        Args:
            cx: center x
            cy: center y
            width: width of box
            height: height of box
            angle: rotation angle in degrees (clockwise)
        """
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.angle = angle

    def to_cvbox(self) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
        """
        Convert to OpenCV rotated rectangle format.
        Returns:
            ((center_x, center_y), (width, height), angle)
        """
        return ((self.cx, self.cy), (self.width, self.height), self.angle)

    def to_points(self) -> np.ndarray:
        """
        Get the 4 points of the rotated rectangle.
        Returns:
            Nx2 numpy array of points
        """
        rect = self.to_cvbox()
        points = cv2.boxPoints(rect)
        return points.astype(np.float32)

    def draw(self, image: np.ndarray, color: Tuple[int, int, int] = (0,255,0), thickness: int = 2):
        """
        Draw oriented bounding box on image.
        """
        points = self.to_points()
        pts = points.reshape((-1,1,2)).astype(int)
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)

    def iou(self, other: 'OrientedBoundingBox') -> float:
        """
        Compute IoU between two oriented bounding boxes.
        """
        poly1 = self.to_points()
        poly2 = other.to_points()

        # Convert to cv2 contour format
        poly1 = poly1.reshape((-1, 1, 2)).astype(np.int32)
        poly2 = poly2.reshape((-1, 1, 2)).astype(np.int32)

        # Calculate intersection area
        ret, intersect = cv2.rotatedRectangleIntersection(self.to_cvbox(), other.to_cvbox())
        if ret == cv2.INTERSECT_NONE or intersect is None:
            return 0.0
        inter_area = cv2.contourArea(intersect)

        area1 = self.width * self.height
        area2 = other.width * other.height
        union = area1 + area2 - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

class OBBAnnotator:
    def __init__(self, color: Tuple[int,int,int] = (0,255,0)):
        self.color = color

    def annotate(self, image: np.ndarray, obb: OrientedBoundingBox) -> np.ndarray:
        obb.draw(image, color=self.color)
        return image

def obb_nms(boxes: List[OrientedBoundingBox], scores: List[float], iou_threshold: float) -> List[int]:
    """
    Perform non-maximum suppression (NMS) on oriented bounding boxes.
    Args:
        boxes: List of OrientedBoundingBox
        scores: Corresponding confidence scores
        iou_threshold: Threshold to suppress overlapping boxes
    Returns:
        List of indices of boxes to keep
    """
    indices = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep = []

    while indices:
        current = indices.pop(0)
        keep.append(current)
        to_delete = []
        for i in indices:
            iou_val = boxes[current].iou(boxes[i])
            if iou_val > iou_threshold:
                to_delete.append(i)
        indices = [i for i in indices if i not in to_delete]
    return keep

# Format conversion functions

def obb_to_xyxy(obb: OrientedBoundingBox) -> Tuple[float, float, float, float]:
    """
    Convert OBB to axis-aligned bounding box (xyxy format).
    """
    pts = obb.to_points()
    x_min = float(np.min(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    x_max = float(np.max(pts[:, 0]))
    y_max = float(np.max(pts[:, 1]))
    return (x_min, y_min, x_max, y_max)

def xyxy_to_obb(box: Tuple[float, float, float, float]) -> OrientedBoundingBox:
    """
    Convert axis-aligned bounding box to OBB with zero rotation.
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return OrientedBoundingBox(cx, cy, width, height, angle=0.0)
