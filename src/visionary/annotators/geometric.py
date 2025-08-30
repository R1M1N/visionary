"""
Geometric Annotators for Visionary

This module provides a comprehensive set of geometric annotation tools
for computer vision applications, including various box styles, shapes,
and minimalist corner displays.
"""

import numpy as np
import cv2
import math
from typing import Optional, Tuple, Any, Union, List
from enum import Enum

from .base import BaseAnnotator, ColorPalette, Position


class AnchorPoint(Enum):
    """Anchor point options for dot annotations."""
    CENTER = "center"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    TOP_CENTER = "top_center"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    BOTTOM_CENTER = "bottom_center"
    CENTER_LEFT = "center_left"
    CENTER_RIGHT = "center_right"


class BoxAnnotator(BaseAnnotator):
    """Standard rectangular bounding box annotator."""
    
    def annotate(self, scene: np.ndarray, detections: Any, **kwargs) -> np.ndarray:
        if len(detections) == 0:
            return scene
            
        for det_idx in range(len(detections)):
            box = detections.xyxy[det_idx].astype(int)
            
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                color = self.get_color_by_class(int(detections.class_id[det_idx]))
            else:
                color = self.color
                
            cv2.rectangle(scene, (box[0], box[1]), (box[2], box[3]), color, self.thickness)
            
        return scene


class RoundBoxAnnotator(BaseAnnotator):
    """Rounded rectangle box annotator for softer appearance."""
    
    def __init__(self, corner_radius: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.corner_radius = corner_radius
        
    def annotate(self, scene: np.ndarray, detections: Any, **kwargs) -> np.ndarray:
        if len(detections) == 0:
            return scene
            
        radius = kwargs.get('corner_radius', self.corner_radius)
        
        for det_idx in range(len(detections)):
            box = detections.xyxy[det_idx].astype(int)
            
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                color = self.get_color_by_class(int(detections.class_id[det_idx]))
            else:
                color = self.color
                
            self._draw_rounded_rectangle(scene, box, color, self.thickness, radius)
            
        return scene
    
    def _draw_rounded_rectangle(self, image: np.ndarray, box: np.ndarray, 
                              color: Tuple[int, int, int], thickness: int, radius: int):
        x1, y1, x2, y2 = box
        max_radius = min((x2 - x1) // 2, (y2 - y1) // 2)
        radius = min(radius, max_radius)
        
        if radius <= 0:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            return
            
        # Draw straight lines
        cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Draw corner arcs
        cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


class OrientedBoxAnnotator(BaseAnnotator):
    """Oriented bounding box annotator for rotated objects."""
    
    def annotate(self, scene: np.ndarray, detections: Any, **kwargs) -> np.ndarray:
        if len(detections) == 0:
            return scene
            
        for det_idx in range(len(detections)):
            if hasattr(detections, 'data') and detections.data and 'obb_coordinates' in detections.data:
                obb_coords = detections.data['obb_coordinates'][det_idx]
                
                if hasattr(detections, 'class_id') and detections.class_id is not None:
                    color = self.get_color_by_class(int(detections.class_id[det_idx]))
                else:
                    color = self.color
                
                cv2.drawContours(scene, [obb_coords.astype(int)], 0, color, self.thickness)
            
            else:
                # Fallback to regular box
                box = detections.xyxy[det_idx].astype(int)
                if hasattr(detections, 'class_id') and detections.class_id is not None:
                    color = self.get_color_by_class(int(detections.class_id[det_idx]))
                else:
                    color = self.color
                cv2.rectangle(scene, (box[0], box[1]), (box[2], box[3]), color, self.thickness)
                
        return scene


class CircleAnnotator(BaseAnnotator):
    """Circle annotator for marking detection centers."""
    
    def __init__(self, radius: int = 10, adaptive_radius: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius
        self.adaptive_radius = adaptive_radius
        
    def annotate(self, scene: np.ndarray, detections: Any, **kwargs) -> np.ndarray:
        if len(detections) == 0:
            return scene
            
        for det_idx in range(len(detections)):
            box = detections.xyxy[det_idx]
            
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            center = (center_x, center_y)
            
            if self.adaptive_radius:
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                radius = max(5, int(math.sqrt(box_area) / 10))
            else:
                radius = kwargs.get('radius', self.radius)
            
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                color = self.get_color_by_class(int(detections.class_id[det_idx]))
            else:
                color = self.color
                
            cv2.circle(scene, center, radius, color, self.thickness)
            
        return scene


class EllipseAnnotator(BaseAnnotator):
    """Ellipse annotator for oval-shaped annotations."""
    
    def __init__(self, axes_ratio: Tuple[float, float] = (1.0, 0.7), **kwargs):
        super().__init__(**kwargs)
        self.axes_ratio = axes_ratio
        
    def annotate(self, scene: np.ndarray, detections: Any, **kwargs) -> np.ndarray:
        if len(detections) == 0:
            return scene
            
        axes_ratio = kwargs.get('axes_ratio', self.axes_ratio)
        angle = kwargs.get('angle', 0)
        
        for det_idx in range(len(detections)):
            box = detections.xyxy[det_idx]
            
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            center = (center_x, center_y)
            
            width = int((box[2] - box[0]) / 2 * axes_ratio[0])
            height = int((box[3] - box[1]) / 2 * axes_ratio[1])
            axes = (width, height)
            
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                color = self.get_color_by_class(int(detections.class_id[det_idx]))
            else:
                color = self.color
                
            cv2.ellipse(scene, center, axes, angle, 0, 360, color, self.thickness)
            
        return scene


class TriangleAnnotator(BaseAnnotator):
    """Triangle annotator for directional or specialized marking."""
    
    def __init__(self, triangle_type: str = "equilateral", **kwargs):
        super().__init__(**kwargs)
        self.triangle_type = triangle_type
        
    def annotate(self, scene: np.ndarray, detections: Any, **kwargs) -> np.ndarray:
        if len(detections) == 0:
            return scene
            
        triangle_type = kwargs.get('triangle_type', self.triangle_type)
        
        for det_idx in range(len(detections)):
            box = detections.xyxy[det_idx].astype(int)
            
            if triangle_type == "equilateral":
                points = self._get_equilateral_triangle(box)
            elif triangle_type == "isosceles":
                points = self._get_isosceles_triangle(box)
            elif triangle_type == "right":
                points = self._get_right_triangle(box)
            else:
                points = self._get_equilateral_triangle(box)
            
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                color = self.get_color_by_class(int(detections.class_id[det_idx]))
            else:
                color = self.color
                
            cv2.drawContours(scene, [points], 0, color, self.thickness)
            
        return scene
    
    def _get_equilateral_triangle(self, box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        p1 = (center_x, y1)
        p2 = (x1, y2)
        p3 = (x2, y2)
        return np.array([p1, p2, p3], np.int32)
    
    def _get_isosceles_triangle(self, box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        quarter_width = (x2 - x1) // 4
        p1 = (center_x, y1)
        p2 = (x1 + quarter_width, y2)
        p3 = (x2 - quarter_width, y2)
        return np.array([p1, p2, p3], np.int32)
    
    def _get_right_triangle(self, box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = box
        p1 = (x1, y1)
        p2 = (x1, y2)
        p3 = (x2, y2)
        return np.array([p1, p2, p3], np.int32)


class DotAnnotator(BaseAnnotator):
    """Dot annotator for precise point marking."""
    
    def __init__(self, radius: int = 4, anchor: Union[AnchorPoint, str] = AnchorPoint.CENTER, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius
        self.anchor = anchor if isinstance(anchor, AnchorPoint) else AnchorPoint(anchor)
        
    def annotate(self, scene: np.ndarray, detections: Any, **kwargs) -> np.ndarray:
        if len(detections) == 0:
            return scene
            
        radius = kwargs.get('radius', self.radius)
        anchor = kwargs.get('anchor', self.anchor)
        if isinstance(anchor, str):
            anchor = AnchorPoint(anchor)
            
        for det_idx in range(len(detections)):
            box = detections.xyxy[det_idx].astype(int)
            point = self._calculate_anchor_point(box, anchor)
            
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                color = self.get_color_by_class(int(detections.class_id[det_idx]))
            else:
                color = self.color
                
            cv2.circle(scene, point, radius, color, -1)
            
        return scene
    
    def _calculate_anchor_point(self, box: np.ndarray, anchor: AnchorPoint) -> Tuple[int, int]:
        x1, y1, x2, y2 = box
        
        anchor_map = {
            AnchorPoint.CENTER: ((x1 + x2) // 2, (y1 + y2) // 2),
            AnchorPoint.TOP_LEFT: (x1, y1),
            AnchorPoint.TOP_RIGHT: (x2, y1),
            AnchorPoint.TOP_CENTER: ((x1 + x2) // 2, y1),
            AnchorPoint.BOTTOM_LEFT: (x1, y2),
            AnchorPoint.BOTTOM_RIGHT: (x2, y2),
            AnchorPoint.BOTTOM_CENTER: ((x1 + x2) // 2, y2),
            AnchorPoint.CENTER_LEFT: (x1, (y1 + y2) // 2),
            AnchorPoint.CENTER_RIGHT: (x2, (y1 + y2) // 2),
        }
        
        return anchor_map.get(anchor, ((x1 + x2) // 2, (y1 + y2) // 2))


class BoxCornerAnnotator(BaseAnnotator):
    """Minimalist corner-only box annotator."""
    
    def __init__(self, corner_length: int = 15, **kwargs):
        super().__init__(**kwargs)
        self.corner_length = corner_length
        
    def annotate(self, scene: np.ndarray, detections: Any, **kwargs) -> np.ndarray:
        if len(detections) == 0:
            return scene
            
        corner_length = kwargs.get('corner_length', self.corner_length)
        
        for det_idx in range(len(detections)):
            box = detections.xyxy[det_idx].astype(int)
            
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                color = self.get_color_by_class(int(detections.class_id[det_idx]))
            else:
                color = self.color
                
            self._draw_corner_brackets(scene, box, color, self.thickness, corner_length)
            
        return scene
    
    def _draw_corner_brackets(self, image: np.ndarray, box: np.ndarray, 
                            color: Tuple[int, int, int], thickness: int, length: int):
        x1, y1, x2, y2 = box
        max_length = min((x2 - x1) // 3, (y2 - y1) // 3)
        length = min(length, max_length)
        
        # Draw corner brackets
        cv2.line(image, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(image, (x1, y1), (x1, y1 + length), color, thickness)
        cv2.line(image, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(image, (x2, y1), (x2, y1 + length), color, thickness)
        cv2.line(image, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(image, (x1, y2), (x1, y2 - length), color, thickness)
        cv2.line(image, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(image, (x2, y2), (x2, y2 - length), color, thickness)


class PolygonAnnotator(BaseAnnotator):
    """Polygon annotator for irregular shapes and segmentation masks."""
    
    def annotate(self, scene: np.ndarray, detections: Any, **kwargs) -> np.ndarray:
        if len(detections) == 0:
            return scene
            
        if hasattr(detections, 'mask') and detections.mask is not None:
            for det_idx in range(len(detections)):
                mask = detections.mask[det_idx]
                
                if hasattr(detections, 'class_id') and detections.class_id is not None:
                    color = self.get_color_by_class(int(detections.class_id[det_idx]))
                else:
                    color = self.color
                
                if mask.dtype != np.uint8:
                    mask = (mask * 255).astype(np.uint8)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    cv2.drawContours(scene, [contour], -1, color, self.thickness)
        
        return scene


class HexagonAnnotator(BaseAnnotator):
    """Hexagon annotator for specialized geometric marking."""
    
    def annotate(self, scene: np.ndarray, detections: Any, **kwargs) -> np.ndarray:
        if len(detections) == 0:
            return scene
            
        for det_idx in range(len(detections)):
            box = detections.xyxy[det_idx].astype(int)
            
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                color = self.get_color_by_class(int(detections.class_id[det_idx]))
            else:
                color = self.color
            
            points = self._get_hexagon_points(box)
            cv2.drawContours(scene, [points], 0, color, self.thickness)
            
        return scene
    
    def _get_hexagon_points(self, box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius_x = (x2 - x1) // 3
        radius_y = (y2 - y1) // 3
        
        points = []
        for i in range(6):
            angle = i * np.pi / 3
            x = center_x + radius_x * np.cos(angle)
            y = center_y + radius_y * np.sin(angle)
            points.append([int(x), int(y)])
        
        return np.array(points, np.int32)


class CrossAnnotator(BaseAnnotator):
    """Cross/plus annotator for center marking."""
    
    def __init__(self, cross_size: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.cross_size = cross_size
        
    def annotate(self, scene: np.ndarray, detections: Any, **kwargs) -> np.ndarray:
        if len(detections) == 0:
            return scene
            
        cross_size = kwargs.get('cross_size', self.cross_size)
        
        for det_idx in range(len(detections)):
            box = detections.xyxy[det_idx].astype(int)
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2
            
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                color = self.get_color_by_class(int(detections.class_id[det_idx]))
            else:
                color = self.color
            
            cv2.line(scene, (center_x - cross_size, center_y), 
                    (center_x + cross_size, center_y), color, self.thickness)
            cv2.line(scene, (center_x, center_y - cross_size), 
                    (center_x, center_y + cross_size), color, self.thickness)
            
        return scene


class CompositeAnnotator(BaseAnnotator):
    """Composite annotator that combines multiple annotation styles."""
    
    def __init__(self, annotator_configs: List[Tuple[str, dict]], **kwargs):
        super().__init__(**kwargs)
        self.annotators = []
        
        annotator_map = {
            'box': BoxAnnotator,
            'round_box': RoundBoxAnnotator,
            'oriented_box': OrientedBoxAnnotator,
            'circle': CircleAnnotator,
            'ellipse': EllipseAnnotator,
            'triangle': TriangleAnnotator,
            'dot': DotAnnotator,
            'corner': BoxCornerAnnotator,
            'polygon': PolygonAnnotator,
            'hexagon': HexagonAnnotator,
            'cross': CrossAnnotator,
        }
        
        for annotator_type, config in annotator_configs:
            if annotator_type in annotator_map:
                annotator_class = annotator_map[annotator_type]
                self.annotators.append(annotator_class(**config))
    
    def annotate(self, scene: np.ndarray, detections: Any, **kwargs) -> np.ndarray:
        for annotator in self.annotators:
            scene = annotator.annotate(scene, detections, **kwargs)
        return scene


# Utility functions
def create_composite_annotator(annotator_configs: List[Tuple[str, dict]]) -> CompositeAnnotator:
    """Create a composite annotator from configuration list."""
    return CompositeAnnotator(annotator_configs)


def get_annotator_by_name(name: str, **kwargs) -> BaseAnnotator:
    """Factory function to create annotators by name."""
    annotators = {
        'box': BoxAnnotator,
        'round_box': RoundBoxAnnotator,
        'oriented_box': OrientedBoxAnnotator,
        'circle': CircleAnnotator,
        'ellipse': EllipseAnnotator,
        'triangle': TriangleAnnotator,
        'dot': DotAnnotator,
        'corner': BoxCornerAnnotator,
        'polygon': PolygonAnnotator,
        'hexagon': HexagonAnnotator,
        'cross': CrossAnnotator,
        'composite': CompositeAnnotator,
    }
    
    if name.lower() not in annotators:
        raise ValueError(f"Unknown annotator: {name}. Available: {list(annotators.keys())}")
    
    return annotators[name.lower()](**kwargs)


def create_detection_style_preset(style_name: str) -> List[Tuple[str, dict]]:
    """Create preset annotation style configurations."""
    presets = {
        'minimal': [('dot', {'radius': 3, 'anchor': 'center'})],
        'classic': [('box', {'thickness': 2})],
        'modern': [('round_box', {'corner_radius': 8, 'thickness': 2})],
        'detailed': [('box', {'thickness': 2}), ('dot', {'radius': 4, 'anchor': 'center'})],
        'corners_only': [('corner', {'corner_length': 12, 'thickness': 2})],
        'artistic': [('ellipse', {'axes_ratio': (0.8, 1.2), 'thickness': 3}), ('cross', {'cross_size': 8, 'thickness': 2})],
        'professional': [('box', {'thickness': 2}), ('corner', {'corner_length': 8, 'thickness': 1})]
    }
    
    if style_name.lower() not in presets:
        raise ValueError(f"Unknown preset: {style_name}. Available: {list(presets.keys())}")
    
    return presets[style_name.lower()]
