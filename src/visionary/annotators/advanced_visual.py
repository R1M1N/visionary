
"""
Advanced Visual Annotators for Visionary

This module provides sophisticated visual annotation tools including
mask overlays, polygon shapes, halo effects, solid fills, and
confidence visualization bars.
"""

import numpy as np
import cv2
import math
from typing import Optional, Tuple, Any, Union, List
from enum import Enum

from .base import BaseAnnotator, ColorPalette, Position, blend_colors, adjust_color_brightness


class BlendMode(Enum):
    """Blend modes for mask overlays."""
    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    SOFT_LIGHT = "soft_light"


class MaskAnnotator(BaseAnnotator):
    """
    Advanced mask annotator with opacity control and blend modes.

    Provides sophisticated mask overlay capabilities with multiple
    blend modes, opacity control, and edge enhancement options.
    """

    def __init__(
        self, 
        opacity: float = 0.3,
        blend_mode: BlendMode = BlendMode.NORMAL,
        show_edges: bool = True,
        edge_thickness: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.opacity = opacity
        self.blend_mode = blend_mode if isinstance(blend_mode, BlendMode) else BlendMode(blend_mode)
        self.show_edges = show_edges
        self.edge_thickness = edge_thickness

    def annotate(
        self, 
        scene: np.ndarray, 
        detections: Any,
        **kwargs
    ) -> np.ndarray:
        """Draw mask overlays with advanced blending."""
        if len(detections) == 0 or not hasattr(detections, 'mask') or detections.mask is None:
            return scene

        opacity = kwargs.get('opacity', self.opacity)
        blend_mode = kwargs.get('blend_mode', self.blend_mode)
        show_edges = kwargs.get('show_edges', self.show_edges)

        # Create overlay for all masks
        overlay = scene.copy()

        for det_idx in range(len(detections)):
            mask = detections.mask[det_idx]

            # Get color for this detection
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                color = self.get_color_by_class(int(detections.class_id[det_idx]))
            else:
                color = self.color

            # Apply confidence-based opacity
            current_opacity = opacity
            if hasattr(detections, 'confidence') and detections.confidence is not None:
                confidence = float(detections.confidence[det_idx])
                current_opacity = opacity * confidence

            # Ensure mask is binary
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)

            # Apply mask overlay
            overlay = self._apply_mask_overlay(
                overlay, mask, color, current_opacity, blend_mode
            )

            # Draw mask edges if enabled
            if show_edges:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    cv2.drawContours(scene, [contour], -1, color, self.edge_thickness)

        # Final blend with original scene
        scene = cv2.addWeighted(scene, 1 - opacity, overlay, opacity, 0)

        return scene

    def _apply_mask_overlay(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        color: Tuple[int, int, int],
        opacity: float,
        blend_mode: BlendMode
    ) -> np.ndarray:
        """Apply mask overlay with specified blend mode."""
        # Create colored mask
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = color

        # Apply blend mode
        if blend_mode == BlendMode.NORMAL:
            result = self._blend_normal(image, mask_colored, mask, opacity)
        elif blend_mode == BlendMode.MULTIPLY:
            result = self._blend_multiply(image, mask_colored, mask, opacity)
        elif blend_mode == BlendMode.SCREEN:
            result = self._blend_screen(image, mask_colored, mask, opacity)
        elif blend_mode == BlendMode.OVERLAY:
            result = self._blend_overlay(image, mask_colored, mask, opacity)
        elif blend_mode == BlendMode.SOFT_LIGHT:
            result = self._blend_soft_light(image, mask_colored, mask, opacity)
        else:
            result = self._blend_normal(image, mask_colored, mask, opacity)

        return result

    def _blend_normal(self, base: np.ndarray, overlay: np.ndarray, mask: np.ndarray, opacity: float) -> np.ndarray:
        """Normal blend mode."""
        result = base.copy()
        mask_area = mask > 0
        result[mask_area] = cv2.addWeighted(
            base[mask_area], 1 - opacity, overlay[mask_area], opacity, 0
        )
        return result

    def _blend_multiply(self, base: np.ndarray, overlay: np.ndarray, mask: np.ndarray, opacity: float) -> np.ndarray:
        """Multiply blend mode."""
        result = base.copy()
        mask_area = mask > 0
        blended = (base.astype(np.float32) * overlay.astype(np.float32) / 255.0).astype(np.uint8)
        result[mask_area] = cv2.addWeighted(
            base[mask_area], 1 - opacity, blended[mask_area], opacity, 0
        )
        return result

    def _blend_screen(self, base: np.ndarray, overlay: np.ndarray, mask: np.ndarray, opacity: float) -> np.ndarray:
        """Screen blend mode."""
        result = base.copy()
        mask_area = mask > 0
        blended = (255 - (255 - base.astype(np.float32)) * (255 - overlay.astype(np.float32)) / 255.0).astype(np.uint8)
        result[mask_area] = cv2.addWeighted(
            base[mask_area], 1 - opacity, blended[mask_area], opacity, 0
        )
        return result

    def _blend_overlay(self, base: np.ndarray, overlay: np.ndarray, mask: np.ndarray, opacity: float) -> np.ndarray:
        """Overlay blend mode."""
        result = base.copy()
        mask_area = mask > 0

        base_norm = base.astype(np.float32) / 255.0
        overlay_norm = overlay.astype(np.float32) / 255.0

        # Overlay formula
        blended = np.where(
            base_norm < 0.5,
            2 * base_norm * overlay_norm,
            1 - 2 * (1 - base_norm) * (1 - overlay_norm)
        )

        blended = (blended * 255).astype(np.uint8)
        result[mask_area] = cv2.addWeighted(
            base[mask_area], 1 - opacity, blended[mask_area], opacity, 0
        )
        return result

    def _blend_soft_light(self, base: np.ndarray, overlay: np.ndarray, mask: np.ndarray, opacity: float) -> np.ndarray:
        """Soft light blend mode."""
        result = base.copy()
        mask_area = mask > 0

        base_norm = base.astype(np.float32) / 255.0
        overlay_norm = overlay.astype(np.float32) / 255.0

        # Soft light formula (simplified)
        blended = np.where(
            overlay_norm < 0.5,
            base_norm - (1 - 2 * overlay_norm) * base_norm * (1 - base_norm),
            base_norm + (2 * overlay_norm - 1) * (np.sqrt(base_norm) - base_norm)
        )

        blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
        result[mask_area] = cv2.addWeighted(
            base[mask_area], 1 - opacity, blended[mask_area], opacity, 0
        )
        return result


class PolygonAnnotator(BaseAnnotator):
    """
    Advanced polygon annotator for custom shapes.

    Supports custom polygon shapes, filled polygons, gradient fills,
    and complex multi-polygon annotations.
    """

    def __init__(
        self, 
        fill: bool = False,
        gradient_fill: bool = False,
        line_type: int = cv2.LINE_AA,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fill = fill
        self.gradient_fill = gradient_fill
        self.line_type = line_type

    def annotate(
        self, 
        scene: np.ndarray, 
        detections: Any,
        **kwargs
    ) -> np.ndarray:
        """Draw polygon annotations with various fill options."""
        if len(detections) == 0:
            return scene

        fill = kwargs.get('fill', self.fill)
        gradient_fill = kwargs.get('gradient_fill', self.gradient_fill)

        # Handle mask-based polygons
        if hasattr(detections, 'mask') and detections.mask is not None:
            return self._annotate_from_masks(scene, detections, fill, gradient_fill, **kwargs)

        # Handle custom polygon data
        if hasattr(detections, 'data') and 'polygons' in detections.data:
            return self._annotate_from_polygons(scene, detections, fill, gradient_fill, **kwargs)

        # Fallback to bounding box polygons
        return self._annotate_from_boxes(scene, detections, fill, gradient_fill, **kwargs)

    def _annotate_from_masks(
        self, 
        scene: np.ndarray, 
        detections: Any, 
        fill: bool, 
        gradient_fill: bool,
        **kwargs
    ) -> np.ndarray:
        """Create polygon annotations from segmentation masks."""
        for det_idx in range(len(detections)):
            mask = detections.mask[det_idx]

            if hasattr(detections, 'class_id') and detections.class_id is not None:
                color = self.get_color_by_class(int(detections.class_id[det_idx]))
            else:
                color = self.color

            # Find contours from mask
            if mask.dtype != np.uint8:
                mask = (mask * 255).astype(np.uint8)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) >= 3:  # Valid polygon
                    if fill:
                        if gradient_fill:
                            self._draw_gradient_polygon(scene, contour, color)
                        else:
                            cv2.fillPoly(scene, [contour], color, self.line_type)
                    else:
                        cv2.polylines(scene, [contour], True, color, self.thickness, self.line_type)

        return scene

    def _annotate_from_polygons(
        self, 
        scene: np.ndarray, 
        detections: Any, 
        fill: bool, 
        gradient_fill: bool,
        **kwargs
    ) -> np.ndarray:
        """Create annotations from custom polygon data."""
        polygons = detections.data['polygons']

        for det_idx in range(len(detections)):
            polygon = polygons[det_idx]

            if hasattr(detections, 'class_id') and detections.class_id is not None:
                color = self.get_color_by_class(int(detections.class_id[det_idx]))
            else:
                color = self.color

            # Convert polygon to numpy array
            if isinstance(polygon, list):
                polygon = np.array(polygon, dtype=np.int32)

            if len(polygon) >= 3:
                if fill:
                    if gradient_fill:
                        self._draw_gradient_polygon(scene, polygon, color)
                    else:
                        cv2.fillPoly(scene, [polygon], color, self.line_type)
                else:
                    cv2.polylines(scene, [polygon], True, color, self.thickness, self.line_type)

        return scene

    def _annotate_from_boxes(
        self, 
        scene: np.ndarray, 
        detections: Any, 
        fill: bool, 
        gradient_fill: bool,
        **kwargs
    ) -> np.ndarray:
        """Create polygon annotations from bounding boxes."""
        polygon_type = kwargs.get('polygon_type', 'octagon')

        for det_idx in range(len(detections)):
            box = detections.xyxy[det_idx].astype(int)

            if hasattr(detections, 'class_id') and detections.class_id is not None:
                color = self.get_color_by_class(int(detections.class_id[det_idx]))
            else:
                color = self.color

            # Generate polygon from bounding box
            polygon = self._generate_polygon_from_box(box, polygon_type)

            if fill:
                if gradient_fill:
                    self._draw_gradient_polygon(scene, polygon, color)
                else:
                    cv2.fillPoly(scene, [polygon], color, self.line_type)
            else:
                cv2.polylines(scene, [polygon], True, color, self.thickness, self.line_type)

        return scene

    def _generate_polygon_from_box(self, box: np.ndarray, polygon_type: str) -> np.ndarray:
        """Generate polygon points from bounding box."""
        x1, y1, x2, y2 = box

        if polygon_type == 'octagon':
            # Create octagon
            corner_cut = min((x2 - x1), (y2 - y1)) // 6
            points = [
                [x1 + corner_cut, y1],
                [x2 - corner_cut, y1],
                [x2, y1 + corner_cut],
                [x2, y2 - corner_cut],
                [x2 - corner_cut, y2],
                [x1 + corner_cut, y2],
                [x1, y2 - corner_cut],
                [x1, y1 + corner_cut]
            ]
        elif polygon_type == 'hexagon':
            # Create hexagon
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
        else:
            # Default rectangle
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        return np.array(points, dtype=np.int32)

    def _draw_gradient_polygon(self, image: np.ndarray, polygon: np.ndarray, base_color: Tuple[int, int, int]):
        """Draw polygon with gradient fill."""
        # Get bounding rectangle
        rect = cv2.boundingRect(polygon)
        x, y, w, h = rect

        # Create mask for polygon
        mask = np.zeros((h, w), dtype=np.uint8)
        poly_shifted = polygon - [x, y]
        cv2.fillPoly(mask, [poly_shifted], 255)

        # Create gradient
        gradient = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            ratio = i / max(1, h - 1)
            color = blend_colors(base_color, adjust_color_brightness(base_color, 0.5), ratio)
            gradient[i, :] = color

        # Apply mask to gradient
        gradient_masked = cv2.bitwise_and(gradient, gradient, mask=mask)

        # Blend with original image
        roi = image[y:y+h, x:x+w]
        blended = cv2.addWeighted(roi, 0.7, gradient_masked, 0.3, 0)
        image[y:y+h, x:x+w] = blended


class HaloAnnotator(BaseAnnotator):
    """
    Halo annotator with glowing effects.

    Creates glowing halo effects around detections with customizable
    colors, intensity, and blur radius for dramatic visual impact.
    """

    def __init__(
        self, 
        halo_radius: int = 20,
        intensity: float = 0.6,
        blur_iterations: int = 3,
        inner_fade: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.halo_radius = halo_radius
        self.intensity = intensity
        self.blur_iterations = blur_iterations
        self.inner_fade = inner_fade

    def annotate(
        self, 
        scene: np.ndarray, 
        detections: Any,
        **kwargs
    ) -> np.ndarray:
        """Draw halo effects around detections."""
        if len(detections) == 0:
            return scene

        halo_radius = kwargs.get('halo_radius', self.halo_radius)
        intensity = kwargs.get('intensity', self.intensity)
        blur_iterations = kwargs.get('blur_iterations', self.blur_iterations)

        # Create halo overlay
        halo_overlay = np.zeros_like(scene, dtype=np.float32)

        for det_idx in range(len(detections)):
            box = detections.xyxy[det_idx].astype(int)

            # Get color for this detection
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                color = self.get_color_by_class(int(detections.class_id[det_idx]))
            else:
                color = self.color

            # Apply confidence-based intensity
            current_intensity = intensity
            if hasattr(detections, 'confidence') and detections.confidence is not None:
                confidence = float(detections.confidence[det_idx])
                current_intensity = intensity * confidence

            # Create halo for this detection
            self._create_halo(halo_overlay, box, color, halo_radius, current_intensity, blur_iterations)

        # Blend halo with original scene
        halo_overlay = np.clip(halo_overlay, 0, 255).astype(np.uint8)
        scene = cv2.addWeighted(scene, 1.0, halo_overlay, 1.0, 0)

        return scene

    def _create_halo(
        self, 
        overlay: np.ndarray, 
        box: np.ndarray, 
        color: Tuple[int, int, int],
        radius: int,
        intensity: float,
        blur_iterations: int
    ):
        """Create halo effect for a single detection."""
        x1, y1, x2, y2 = box

        # Expand box for halo
        halo_x1 = max(0, x1 - radius)
        halo_y1 = max(0, y1 - radius)
        halo_x2 = min(overlay.shape[1], x2 + radius)
        halo_y2 = min(overlay.shape[0], y2 + radius)

        # Create initial halo shape
        if hasattr(self, 'mask') and self.mask is not None:
            # Use mask if available
            mask = cv2.resize(self.mask, (halo_x2 - halo_x1, halo_y2 - halo_y1))
        else:
            # Create elliptical halo
            center_x = (halo_x2 + halo_x1) // 2 - halo_x1
            center_y = (halo_y2 + halo_y1) // 2 - halo_y1
            axes = ((halo_x2 - halo_x1) // 2, (halo_y2 - halo_y1) // 2)

            mask = np.zeros((halo_y2 - halo_y1, halo_x2 - halo_x1), dtype=np.uint8)
            cv2.ellipse(mask, (center_x, center_y), axes, 0, 0, 360, 255, -1)

        # Create colored halo
        halo_region = np.zeros((halo_y2 - halo_y1, halo_x2 - halo_x1, 3), dtype=np.float32)
        halo_region[mask > 0] = color

        # Apply multiple blur iterations for smoother glow
        for _ in range(blur_iterations):
            halo_region = cv2.GaussianBlur(halo_region, (radius * 2 + 1, radius * 2 + 1), radius / 3)

        # Apply intensity
        halo_region *= intensity

        # Blend with overlay
        overlay[halo_y1:halo_y2, halo_x1:halo_x2] += halo_region


class ColorAnnotator(BaseAnnotator):
    """
    Color annotator for solid fills and color overlays.

    Provides solid color fills with various patterns, textures,
    and color application modes for enhanced visual distinction.
    """

    def __init__(
        self, 
        fill_opacity: float = 0.4,
        pattern: Optional[str] = None,
        border: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fill_opacity = fill_opacity
        self.pattern = pattern
        self.border = border

    def annotate(
        self, 
        scene: np.ndarray, 
        detections: Any,
        **kwargs
    ) -> np.ndarray:
        """Apply solid color fills to detections."""
        if len(detections) == 0:
            return scene

        fill_opacity = kwargs.get('fill_opacity', self.fill_opacity)
        pattern = kwargs.get('pattern', self.pattern)
        border = kwargs.get('border', self.border)

        # Create overlay for fills
        overlay = scene.copy()

        for det_idx in range(len(detections)):
            box = detections.xyxy[det_idx].astype(int)

            # Get color for this detection
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                color = self.get_color_by_class(int(detections.class_id[det_idx]))
            else:
                color = self.color

            # Apply confidence-based opacity
            current_opacity = fill_opacity
            if hasattr(detections, 'confidence') and detections.confidence is not None:
                confidence = float(detections.confidence[det_idx])
                current_opacity = fill_opacity * confidence

            # Apply color fill
            if hasattr(detections, 'mask') and detections.mask is not None:
                self._fill_mask(overlay, detections.mask[det_idx], color, pattern)
            else:
                self._fill_box(overlay, box, color, pattern)

            # Draw border if enabled
            if border:
                if hasattr(detections, 'mask') and detections.mask is not None:
                    mask = detections.mask[det_idx]
                    if mask.dtype != np.uint8:
                        mask = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        cv2.drawContours(scene, [contour], -1, color, self.thickness)
                else:
                    cv2.rectangle(scene, (box[0], box[1]), (box[2], box[3]), color, self.thickness)

        # Blend overlay with scene
        scene = cv2.addWeighted(scene, 1 - fill_opacity, overlay, fill_opacity, 0)

        return scene

    def _fill_mask(self, overlay: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], pattern: Optional[str]):
        """Fill mask area with color and pattern."""
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        if pattern is None:
            # Solid fill
            overlay[mask > 0] = color
        elif pattern == 'stripes':
            self._apply_stripe_pattern(overlay, mask, color)
        elif pattern == 'dots':
            self._apply_dot_pattern(overlay, mask, color)
        elif pattern == 'checkerboard':
            self._apply_checkerboard_pattern(overlay, mask, color)
        else:
            overlay[mask > 0] = color

    def _fill_box(self, overlay: np.ndarray, box: np.ndarray, color: Tuple[int, int, int], pattern: Optional[str]):
        """Fill box area with color and pattern."""
        x1, y1, x2, y2 = box

        if pattern is None:
            # Solid fill
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        elif pattern == 'stripes':
            self._apply_stripe_pattern_box(overlay, box, color)
        elif pattern == 'dots':
            self._apply_dot_pattern_box(overlay, box, color)
        elif pattern == 'checkerboard':
            self._apply_checkerboard_pattern_box(overlay, box, color)
        else:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    def _apply_stripe_pattern(self, overlay: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]):
        """Apply stripe pattern to mask."""
        h, w = mask.shape
        for y in range(0, h, 8):  # Every 8 pixels
            line_mask = mask[y:y+4]  # 4 pixel wide stripes
            if line_mask.size > 0:
                overlay[y:y+4][line_mask > 0] = color

    def _apply_dot_pattern(self, overlay: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]):
        """Apply dot pattern to mask."""
        h, w = mask.shape
        for y in range(4, h, 8):
            for x in range(4, w, 8):
                if mask[y, x] > 0:
                    cv2.circle(overlay, (x, y), 2, color, -1)

    def _apply_checkerboard_pattern(self, overlay: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]):
        """Apply checkerboard pattern to mask."""
        h, w = mask.shape
        for y in range(0, h, 8):
            for x in range(0, w, 8):
                if (x // 8 + y // 8) % 2 == 0:
                    region = mask[y:y+8, x:x+8]
                    if region.size > 0:
                        overlay[y:y+8, x:x+8][region > 0] = color

    def _apply_stripe_pattern_box(self, overlay: np.ndarray, box: np.ndarray, color: Tuple[int, int, int]):
        """Apply stripe pattern to box."""
        x1, y1, x2, y2 = box
        for y in range(y1, y2, 8):
            cv2.rectangle(overlay, (x1, y), (x2, min(y + 4, y2)), color, -1)

    def _apply_dot_pattern_box(self, overlay: np.ndarray, box: np.ndarray, color: Tuple[int, int, int]):
        """Apply dot pattern to box."""
        x1, y1, x2, y2 = box
        for y in range(y1 + 4, y2, 8):
            for x in range(x1 + 4, x2, 8):
                cv2.circle(overlay, (x, y), 2, color, -1)

    def _apply_checkerboard_pattern_box(self, overlay: np.ndarray, box: np.ndarray, color: Tuple[int, int, int]):
        """Apply checkerboard pattern to box."""
        x1, y1, x2, y2 = box
        for y in range(y1, y2, 8):
            for x in range(x1, x2, 8):
                if ((x - x1) // 8 + (y - y1) // 8) % 2 == 0:
                    cv2.rectangle(overlay, (x, y), (min(x + 8, x2), min(y + 8, y2)), color, -1)


class PercentageBarAnnotator(BaseAnnotator):
    """
    Percentage bar annotator for confidence visualization.

    Displays confidence levels as horizontal or vertical bars with
    customizable styling, gradients, and threshold indicators.
    """

    def __init__(
        self, 
        bar_height: int = 8,
        bar_width: int = 60,
        orientation: str = 'horizontal',
        show_text: bool = True,
        gradient_bar: bool = True,
        position_offset: Tuple[int, int] = (0, -15),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bar_height = bar_height
        self.bar_width = bar_width
        self.orientation = orientation
        self.show_text = show_text
        self.gradient_bar = gradient_bar
        self.position_offset = position_offset

    def annotate(
        self, 
        scene: np.ndarray, 
        detections: Any,
        **kwargs
    ) -> np.ndarray:
        """Draw confidence percentage bars."""
        if len(detections) == 0:
            return scene

        if not hasattr(detections, 'confidence') or detections.confidence is None:
            return scene

        bar_height = kwargs.get('bar_height', self.bar_height)
        bar_width = kwargs.get('bar_width', self.bar_width)
        orientation = kwargs.get('orientation', self.orientation)
        show_text = kwargs.get('show_text', self.show_text)
        gradient_bar = kwargs.get('gradient_bar', self.gradient_bar)

        for det_idx in range(len(detections)):
            box = detections.xyxy[det_idx].astype(int)
            confidence = float(detections.confidence[det_idx])

            # Get color for this detection
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                color = self.get_color_by_class(int(detections.class_id[det_idx]))
            else:
                color = self.color

            # Calculate bar position
            if orientation == 'horizontal':
                bar_x = box[0] + self.position_offset[0]
                bar_y = box[1] + self.position_offset[1]
                bar_rect = (bar_x, bar_y, bar_x + bar_width, bar_y + bar_height)
            else:  # vertical
                bar_x = box[2] + self.position_offset[0]
                bar_y = box[1] + self.position_offset[1]
                bar_rect = (bar_x, bar_y, bar_x + bar_height, bar_y + bar_width)

            # Draw percentage bar
            self._draw_percentage_bar(
                scene, bar_rect, confidence, color, orientation, gradient_bar, show_text
            )

        return scene

    def _draw_percentage_bar(
        self,
        image: np.ndarray,
        bar_rect: Tuple[int, int, int, int],
        confidence: float,
        color: Tuple[int, int, int],
        orientation: str,
        gradient_bar: bool,
        show_text: bool
    ):
        """Draw individual percentage bar."""
        x1, y1, x2, y2 = bar_rect

        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, image.shape[1] - 1))
        x2 = max(0, min(x2, image.shape[1] - 1))
        y1 = max(0, min(y1, image.shape[0] - 1))
        y2 = max(0, min(y2, image.shape[0] - 1))

        if x2 <= x1 or y2 <= y1:
            return

        # Background bar (empty)
        background_color = (60, 60, 60)  # Dark gray
        cv2.rectangle(image, (x1, y1), (x2, y2), background_color, -1)

        # Calculate filled portion
        if orientation == 'horizontal':
            fill_width = int((x2 - x1) * confidence)
            fill_rect = (x1, y1, x1 + fill_width, y2)
        else:  # vertical
            fill_height = int((y2 - y1) * confidence)
            fill_rect = (x1, y2 - fill_height, x2, y2)

        # Draw filled portion
        if gradient_bar:
            self._draw_gradient_bar(image, fill_rect, confidence, color, orientation)
        else:
            if fill_rect[2] > fill_rect[0] and fill_rect[3] > fill_rect[1]:
                cv2.rectangle(image, (fill_rect[0], fill_rect[1]), (fill_rect[2], fill_rect[3]), color, -1)

        # Draw border
        border_color = adjust_color_brightness(color, 0.7)
        cv2.rectangle(image, (x1, y1), (x2, y2), border_color, 1)

        # Draw text
        if show_text:
            text = f"{confidence * 100:.0f}%"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]

            if orientation == 'horizontal':
                text_x = x1 + (x2 - x1 - text_size[0]) // 2
                text_y = y1 + (y2 - y1 + text_size[1]) // 2
            else:
                text_x = x1 + (x2 - x1 - text_size[0]) // 2
                text_y = y1 - 5

            # Ensure text is within image bounds
            if text_x >= 0 and text_y >= 0 and text_x + text_size[0] < image.shape[1] and text_y < image.shape[0]:
                cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _draw_gradient_bar(
        self,
        image: np.ndarray,
        rect: Tuple[int, int, int, int],
        confidence: float,
        base_color: Tuple[int, int, int],
        orientation: str
    ):
        """Draw gradient-filled bar."""
        x1, y1, x2, y2 = rect

        if x2 <= x1 or y2 <= y1:
            return

        # Create gradient colors based on confidence
        if confidence < 0.5:
            # Red to yellow gradient for low confidence
            start_color = (0, 0, 255)  # Red
            end_color = (0, 255, 255)  # Yellow
        else:
            # Yellow to green gradient for high confidence
            start_color = (0, 255, 255)  # Yellow
            end_color = (0, 255, 0)    # Green

        # Draw gradient
        if orientation == 'horizontal':
            for x in range(x1, x2):
                ratio = (x - x1) / max(1, x2 - x1)
                color = blend_colors(start_color, end_color, ratio)
                cv2.line(image, (x, y1), (x, y2), color, 1)
        else:  # vertical
            for y in range(y1, y2):
                ratio = (y2 - y) / max(1, y2 - y1)  # Inverted for vertical
                color = blend_colors(start_color, end_color, ratio)
                cv2.line(image, (x1, y), (x2, y), color, 1)


# Utility functions for advanced visual effects
def create_glow_effect(
    image: np.ndarray, 
    mask: np.ndarray, 
    color: Tuple[int, int, int], 
    intensity: float = 0.5,
    radius: int = 15
) -> np.ndarray:
    """Create a glow effect around a mask."""
    # Create glow overlay
    glow = np.zeros_like(image, dtype=np.float32)
    glow[mask > 0] = color

    # Apply multiple blur passes for smooth glow
    for _ in range(3):
        glow = cv2.GaussianBlur(glow, (radius * 2 + 1, radius * 2 + 1), radius / 3)

    glow *= intensity
    glow = np.clip(glow, 0, 255).astype(np.uint8)

    return cv2.addWeighted(image, 1.0, glow, 1.0, 0)


def apply_color_temperature(
    color: Tuple[int, int, int], 
    temperature: float
) -> Tuple[int, int, int]:
    """Apply color temperature adjustment."""
    b, g, r = color

    if temperature > 0:  # Warmer (more red/yellow)
        r = min(255, int(r * (1 + temperature * 0.3)))
        g = min(255, int(g * (1 + temperature * 0.1)))
    else:  # Cooler (more blue)
        b = min(255, int(b * (1 + abs(temperature) * 0.3)))
        g = max(0, int(g * (1 - abs(temperature) * 0.1)))

    return (b, g, r)


def create_neon_effect(
    image: np.ndarray,
    contour: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int = 3
) -> np.ndarray:
    """Create neon-like effect on contours."""
    # Draw multiple layers for neon effect
    glow_color = adjust_color_brightness(color, 1.5)

    # Outer glow
    cv2.drawContours(image, [contour], -1, glow_color, thickness + 4, cv2.LINE_AA)
    # Inner bright line
    cv2.drawContours(image, [contour], -1, (255, 255, 255), thickness, cv2.LINE_AA)
    # Core color
    cv2.drawContours(image, [contour], -1, color, thickness - 1, cv2.LINE_AA)

    return image
