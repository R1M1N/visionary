"""
Bounding Box Utilities for Visionary

This module provides comprehensive utilities for bounding box operations,
coordinate system conversions, and geometric calculations.
"""

import numpy as np
from typing import Union, Tuple, List, Optional
import warnings


def box_iou_batch(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculate IoU between two sets of boxes in batch mode.
    
    Args:
        boxes1: Array of shape (N, 4) in xyxy format
        boxes2: Array of shape (M, 4) in xyxy format
        
    Returns:
        IoU matrix of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Broadcasting for intersection calculation
    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])

    # Intersection area
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Union area
    union_area = area1[:, None] + area2 - inter_area
    
    # Avoid division by zero
    iou = inter_area / (union_area + 1e-6)
    return iou


def mask_to_xyxy(mask: np.ndarray) -> np.ndarray:
    """
    Convert segmentation mask to bounding box coordinates.
    
    Args:
        mask: 2D boolean or binary mask
        
    Returns:
        Bounding box coordinates [x1, y1, x2, y2]
    """
    if mask.ndim != 2:
        raise ValueError("Input mask must be a 2D array")
    
    if mask.dtype == bool:
        mask_indices = np.where(mask)
    else:
        mask_indices = np.where(mask > 0)
    
    y_indices, x_indices = mask_indices
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        return np.array([0, 0, 0, 0], dtype=np.float32)
    
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from (x1, y1, x2, y2) to (x, y, width, height) format.
    
    Args:
        boxes: Array of shape (..., 4) in xyxy format
        
    Returns:
        Array of same shape in xywh format
    """
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1
    return np.stack((x, y, w, h), axis=-1)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from (x, y, width, height) to (x1, y1, x2, y2) format.
    
    Args:
        boxes: Array of shape (..., 4) in xywh format
        
    Returns:
        Array of same shape in xyxy format
    """
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return np.stack((x1, y1, x2, y2), axis=-1)


def xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from (x1, y1, x2, y2) to (center_x, center_y, width, height).
    
    Args:
        boxes: Array of shape (..., 4) in xyxy format
        
    Returns:
        Array of same shape in cxcywh format
    """
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return np.stack((cx, cy, w, h), axis=-1)


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2).
    
    Args:
        boxes: Array of shape (..., 4) in cxcywh format
        
    Returns:
        Array of same shape in xyxy format
    """
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack((x1, y1, x2, y2), axis=-1)


def clip_boxes(boxes: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Clip bounding boxes to image boundaries.
    
    Args:
        boxes: Array of shape (..., 4) in xyxy format
        img_shape: Image dimensions (height, width)
        
    Returns:
        Clipped boxes
    """
    height, width = img_shape[:2]
    
    boxes = boxes.copy()
    boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, width)
    boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, height)
    
    return boxes


def scale_boxes(boxes: np.ndarray, scale_factor: Union[float, Tuple[float, float]]) -> np.ndarray:
    """
    Scale bounding boxes by given factor(s).
    
    Args:
        boxes: Array of shape (..., 4) in xyxy format
        scale_factor: Single scale factor or (scale_x, scale_y)
        
    Returns:
        Scaled boxes
    """
    if isinstance(scale_factor, (int, float)):
        scale_x = scale_y = float(scale_factor)
    else:
        scale_x, scale_y = float(scale_factor[0]), float(scale_factor[1])
    
    scaled_boxes = boxes.copy()
    scaled_boxes[..., [0, 2]] *= scale_x
    scaled_boxes[..., [1, 3]] *= scale_y
    
    return scaled_boxes


def expand_boxes(boxes: np.ndarray, expand_ratio: float = 0.1) -> np.ndarray:
    """
    Expand bounding boxes by given ratio.
    
    Args:
        boxes: Array of shape (..., 4) in xyxy format
        expand_ratio: Expansion ratio (0.1 = 10% larger)
        
    Returns:
        Expanded boxes
    """
    if expand_ratio <= 0:
        return boxes
    
    # Convert to center format for easier expansion
    cxcywh = xyxy_to_cxcywh(boxes)
    
    # Expand width and height
    cxcywh[..., 2] *= (1 + expand_ratio)
    cxcywh[..., 3] *= (1 + expand_ratio)
    
    # Convert back to xyxy
    return cxcywh_to_xyxy(cxcywh)


def box_area(boxes: np.ndarray) -> np.ndarray:
    """
    Calculate area of bounding boxes.
    
    Args:
        boxes: Array of shape (..., 4) in xyxy format
        
    Returns:
        Array of areas
    """
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def box_intersection(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculate intersection area between boxes.
    
    Args:
        boxes1: Array of shape (N, 4) in xyxy format
        boxes2: Array of shape (M, 4) in xyxy format
        
    Returns:
        Intersection areas of shape (N, M)
    """
    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    
    return inter_w * inter_h


def box_union(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculate union area between boxes.
    
    Args:
        boxes1: Array of shape (N, 4) in xyxy format  
        boxes2: Array of shape (M, 4) in xyxy format
        
    Returns:
        Union areas of shape (N, M)
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    intersection = box_intersection(boxes1, boxes2)
    
    return area1[:, None] + area2 - intersection


def normalize_boxes(boxes: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Normalize bounding boxes to [0, 1] range.
    
    Args:
        boxes: Array of shape (..., 4) in xyxy format
        img_shape: Image dimensions (height, width)
        
    Returns:
        Normalized boxes
    """
    height, width = img_shape[:2]
    
    # Ensure we work with float types to avoid formatting errors
    normalized_boxes = boxes.copy().astype(np.float32)
    normalized_boxes[..., [0, 2]] /= float(width)
    normalized_boxes[..., [1, 3]] /= float(height)
    
    return normalized_boxes


def denormalize_boxes(boxes: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Denormalize boxes from [0, 1] range to image coordinates.
    
    Args:
        boxes: Array of shape (..., 4) with normalized coordinates
        img_shape: Image dimensions (height, width)
        
    Returns:
        Denormalized boxes
    """
    height, width = img_shape[:2]
    
    denormalized_boxes = boxes.copy().astype(np.float32)
    denormalized_boxes[..., [0, 2]] *= float(width)
    denormalized_boxes[..., [1, 3]] *= float(height)
    
    return denormalized_boxes


def filter_small_boxes(boxes: np.ndarray, min_size: float = 1.0) -> np.ndarray:
    """
    Filter out boxes smaller than minimum size.
    
    Args:
        boxes: Array of shape (N, 4) in xyxy format
        min_size: Minimum box size (width or height)
        
    Returns:
        Boolean mask for valid boxes
    """
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    
    return (widths >= min_size) & (heights >= min_size)


def boxes_center_distance(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculate Euclidean distance between box centers.
    
    Args:
        boxes1: Array of shape (N, 4) in xyxy format
        boxes2: Array of shape (M, 4) in xyxy format
        
    Returns:
        Distance matrix of shape (N, M)
    """
    # Calculate centers
    centers1 = (boxes1[:, :2] + boxes1[:, 2:]) / 2
    centers2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2
    
    # Calculate pairwise distances
    diff = centers1[:, None, :] - centers2[None, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    
    return distances


def apply_box_transform(boxes: np.ndarray, transform_func, *args, **kwargs) -> np.ndarray:
    """
    Apply transformation function to boxes with error handling.
    
    Args:
        boxes: Input boxes
        transform_func: Transformation function
        *args, **kwargs: Arguments for transformation function
        
    Returns:
        Transformed boxes
    """
    try:
        return transform_func(boxes, *args, **kwargs)
    except Exception as e:
        warnings.warn(f"Box transformation failed: {str(e)}")
        return boxes


# Additional utility functions for comprehensive box operations
def boxes_overlap_ratio(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculate overlap ratio between boxes.
    
    Args:
        boxes1: Array of shape (N, 4) in xyxy format
        boxes2: Array of shape (M, 4) in xyxy format
        
    Returns:
        Overlap ratio matrix of shape (N, M)
    """
    intersection = box_intersection(boxes1, boxes2)
    area1 = box_area(boxes1)
    
    # Avoid division by zero
    overlap_ratio = intersection / (area1[:, None] + 1e-6)
    return overlap_ratio


def boxes_giou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculate Generalized IoU (GIoU) between boxes.
    
    Args:
        boxes1: Array of shape (N, 4) in xyxy format
        boxes2: Array of shape (M, 4) in xyxy format
        
    Returns:
        GIoU matrix of shape (N, M)
    """
    # Calculate IoU
    intersection = box_intersection(boxes1, boxes2)
    union = box_union(boxes1, boxes2)
    iou = intersection / (union + 1e-6)
    
    # Calculate enclosing box
    enclose_x1 = np.minimum(boxes1[:, None, 0], boxes2[:, 0])
    enclose_y1 = np.minimum(boxes1[:, None, 1], boxes2[:, 1])
    enclose_x2 = np.maximum(boxes1[:, None, 2], boxes2[:, 2])
    enclose_y2 = np.maximum(boxes1[:, None, 3], boxes2[:, 3])
    
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    
    # Calculate GIoU
    giou = iou - (enclose_area - union) / (enclose_area + 1e-6)
    return giou


def rotate_boxes(boxes: np.ndarray, angle: float, center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Rotate bounding boxes by given angle.
    
    Args:
        boxes: Array of shape (N, 4) in xyxy format
        angle: Rotation angle in degrees
        center: Rotation center (cx, cy). If None, uses image center
        
    Returns:
        Rotated boxes (may not be axis-aligned)
    """
    import math
    
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # Convert to corner points
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    
    if center is None:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
    else:
        cx, cy = center
    
    # Get all four corners
    corners = np.array([
        [x1, y1],  # top-left
        [x2, y1],  # top-right
        [x2, y2],  # bottom-right
        [x1, y2]   # bottom-left
    ])
    
    # Translate to origin, rotate, translate back
    corners_centered = corners - np.array([cx, cy])
    
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    rotated_corners = corners_centered @ rotation_matrix.T + np.array([cx, cy])
    
    # Find new bounding box
    x_coords = rotated_corners[:, :, 0]
    y_coords = rotated_corners[:, :, 1]
    
    new_x1 = np.min(x_coords, axis=1)
    new_y1 = np.min(y_coords, axis=1)
    new_x2 = np.max(x_coords, axis=1)
    new_y2 = np.max(y_coords, axis=1)
    
    return np.column_stack([new_x1, new_y1, new_x2, new_y2])
