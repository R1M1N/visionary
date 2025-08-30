"""
Detectron2 adapter for Visionary.
"""

import numpy as np
from typing import Any, Optional
from .base import BaseAdapter
from ..core import Detections
import warnings


class Detectron2Adapter(BaseAdapter):
    """
    Adapter for Detectron2 models.
    
    Supports:
    - Faster R-CNN
    - Mask R-CNN  
    - RetinaNet
    - FCOS
    """
    
    def __init__(self):
        super().__init__("Detectron2")
        self.supported_tasks = ["detection", "segmentation", "keypoint"]
    
    def validate_input(self, results: Any) -> bool:
        """Validate Detectron2 Instances object."""
        try:
            return hasattr(results, 'pred_boxes') and hasattr(results, 'scores')
        except Exception:
            return False
    
    def extract_detections(self, results: Any) -> Detections:
        """Extract detections from Detectron2 Instances."""
        try:
            if not self.validate_input(results):
                return Detections(xyxy=np.empty((0, 4)))
            
            # Get boxes and convert to numpy
            boxes = results.pred_boxes.tensor.cpu().numpy().astype(np.float32)
            scores = results.scores.cpu().numpy().astype(np.float32)
            classes = results.pred_classes.cpu().numpy().astype(int)
            
            return Detections(
                xyxy=boxes,
                confidence=scores,
                class_id=classes
            )
            
        except Exception as e:
            warnings.warn(f"Error extracting Detectron2 detections: {e}")
            return Detections(xyxy=np.empty((0, 4)))
    
    def extract_segmentation(self, results: Any) -> Detections:
        """Extract segmentation from Detectron2 Instances."""
        try:
            detections = self.extract_detections(results)
            
            if hasattr(results, 'pred_masks') and results.pred_masks is not None:
                masks = results.pred_masks.cpu().numpy()
                detections.mask = masks
            
            return detections
            
        except Exception as e:
            warnings.warn(f"Error extracting Detectron2 segmentation: {e}")
            return Detections(xyxy=np.empty((0, 4)))
