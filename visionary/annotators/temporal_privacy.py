
"""
Temporal and Privacy Annotators for Visionary

Includes TraceAnnotator, HeatMapAnnotator, BackgroundOverlayAnnotator,
BlurAnnotator, PixelateAnnotator, and IconAnnotator.
"""
import cv2
import numpy as np
from typing import Any, Tuple, List, Dict, Union
from .base import BaseAnnotator, ColorPalette, Position

# At the end of temporal_privacy.py, add:
__all__ = [
    "TraceAnnotator",
    "HeatMapAnnotator", 
    "BackgroundOverlayAnnotator",
    "BlurAnnotator",
    "PixelateAnnotator",
    "IconAnnotator"
]



class TraceAnnotator(BaseAnnotator):
    """
    Visualize object trajectories over time.
    Draws fading lines showing movement paths across frames.
    """
    def __init__(self, max_length: int = 50, color: Tuple[int,int,int]=None, thickness: int=2, **kwargs):
        super().__init__(color=color, thickness=thickness, **kwargs)
        self.max_length = max_length
        # store history of positions per tracker_id
        self.history: Dict[int, List[Tuple[int,int]]] = {}

    def annotate(self, scene: np.ndarray, detections: Any, **kwargs) -> np.ndarray:
        if not hasattr(detections, 'tracker_id') or detections.tracker_id is None:
            return scene
        for idx, tid in enumerate(detections.tracker_id):
            tid = int(tid)
            box = detections.xyxy[idx].astype(int)
            center = ((box[0]+box[2])//2, (box[1]+box[3])//2)
            self.history.setdefault(tid, []).append(center)
            pts = self.history[tid][-self.max_length:]
            # draw fading polyline
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                col = tuple(int(c * alpha) for c in self.color)
                cv2.line(scene, pts[i-1], pts[i], col, self.thickness)
        return scene


class HeatMapAnnotator(BaseAnnotator):
    """
    Create heatmap of activity density over multiple frames.
    """
    def __init__(self, alpha: float=0.6, colormap=cv2.COLORMAP_JET, **kwargs):
        super().__init__(**kwargs)
        self.accumulator = None
        self.alpha = alpha
        self.colormap = colormap

    def annotate(self, scene: np.ndarray, detections: Any, **kwargs) -> np.ndarray:
        h, w = scene.shape[:2]
        if self.accumulator is None:
            self.accumulator = np.zeros((h, w), dtype=np.float32)
        # for each detection add to accumulator mask
        mask = np.zeros((h, w), dtype=np.uint8)
        for box in detections.xyxy.astype(int):
            cv2.rectangle(mask, (box[0],box[1]), (box[2],box[3]), 255, -1)
        self.accumulator += mask.astype(np.float32)
        # normalize
        norm = cv2.normalize(self.accumulator, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.applyColorMap(norm, self.colormap)
        return cv2.addWeighted(scene, 1-self.alpha, heat, self.alpha, 0)


class BackgroundOverlayAnnotator(BaseAnnotator):
    """
    Overlay a translucent background for improved contrast.
    """
    def __init__(self, color: Tuple[int,int,int]=(0,0,0), alpha: float=0.3, **kwargs):
        super().__init__(**kwargs)
        self.bg_color = color
        self.alpha = alpha

    def annotate(self, scene: np.ndarray, detections: Any, **kwargs) -> np.ndarray:
        overlay = scene.copy()
        scene[:] = cv2.addWeighted(overlay, 1-self.alpha, np.full_like(scene, self.bg_color), self.alpha, 0)
        return scene


class BlurAnnotator(BaseAnnotator):
    """
    Blur sensitive regions for privacy.
    """
    def __init__(self, ksize: Tuple[int,int]=(25,25), **kwargs):
        super().__init__(**kwargs)
        self.ksize = ksize

    def annotate(self, scene: np.ndarray, detections: Any, **kwargs) -> np.ndarray:
        for box in detections.xyxy.astype(int):
            x1,y1,x2,y3 = box[0],box[1],box[2],box[3]
            region = scene[y1:y3,x1:x2]
            scene[y1:y3,x1:x2] = cv2.GaussianBlur(region, self.ksize, 0)
        return scene


class PixelateAnnotator(BaseAnnotator):
    """
    Pixelate sensitive regions for privacy.
    """
    def __init__(self, pixel_size: int=10, **kwargs):
        super().__init__(**kwargs)
        self.pixel_size = pixel_size

    def annotate(self, scene: np.ndarray, detections: Any, **kwargs) -> np.ndarray:
        for box in detections.xyxy.astype(int):
            x1,y1,x2,y3 = box[0],box[1],box[2],box[3]
            region = scene[y1:y3,x1:x2]
            h,w = region.shape[:2]
            temp = cv2.resize(region,(w//self.pixel_size,h//self.pixel_size), interpolation=cv2.INTER_LINEAR)
            scene[y1:y3,x1:x2] = cv2.resize(temp,(w,h), interpolation=cv2.INTER_NEAREST)
        return scene


class IconAnnotator(BaseAnnotator):
    """
    Place custom icons at detection positions for privacy or labels.
    """
    def __init__(self, icon: np.ndarray, position: Union[Position,str]=Position.TOP_LEFT, **kwargs):
        super().__init__(**kwargs)
        self.icon = icon
        self.position = Position(position) if isinstance(position,str) else position

    def annotate(self, scene: np.ndarray, detections: Any, **kwargs) -> np.ndarray:
        for box in detections.xyxy.astype(int):
            x1,y1,x2,y3 = box[0],box[1],box[2],box[3]
            h_icon, w_icon = self.icon.shape[:2]
            if self.position == Position.TOP_LEFT:
                x,y = x1,y1
            elif self.position == Position.TOP_RIGHT:
                x,y = x2-w_icon,y1
            elif self.position == Position.BOTTOM_LEFT:
                x,y = x1,y3-h_icon
            else:
                x,y = x2-w_icon,y3-h_icon
            overlay = scene.copy()
            alpha_s = self.icon[:,:,3]/255.0 if self.icon.shape[2]==4 else 1
            for c in range(0,3):
                scene[y:y+h_icon,x:x+w_icon,c] = (alpha_s*self.icon[:,:,c] + (1-alpha_s)*overlay[y:y+h_icon,x:x+w_icon,c])
        return scene

