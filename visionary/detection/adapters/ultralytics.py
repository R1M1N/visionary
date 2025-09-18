"""
Enhanced Ultralytics YOLO adapter for Visionary.
"""

import numpy as np
from typing import Any, Optional, Union, List
from .base import BaseAdapter
from ..core import Detections
import warnings


class UltralyticsAdapter(BaseAdapter):
    """
    Advanced adapter for Ultralytics YOLO models (v8, v9, v10, v11).
    
    Supports:
    - Object detection
    - Instance segmentation  
    - Oriented bounding boxes (OBB)
    - Pose estimation
    - Classification
    """
    
    def __init__(self, weights=None, device='cpu', **kwargs):
        super().__init__("Ultralytics YOLO")
        self.supported_tasks = ["detection", "segmentation", "obb", "pose", "classification"]
        if weights:
            from ultralytics import YOLO
            self.model = YOLO(weights)
            if device != 'cpu':
                self.model.to(device)
            self.device = device
        else:
            self.model = None
            self.device = device

    def __call__(self, source, **kwargs):
        """Allow calling the adapter directly like model(source)"""
        return self.predict(source, **kwargs)    
    
    
    def predict(self, source, **kwargs):
        """Run prediction using the underlying YOLO model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Use the underlying YOLO model's prediction
        results = self.model(source, **kwargs)
        return results
    
    def __call__(self, source, **kwargs):
        """Allow calling the adapter directly like model(source)"""
        return self.predict(source, **kwargs)    

    def load_model(self, weights, device='cpu'):
        """Load a YOLO model with specified weights and device"""
        from ultralytics import YOLO
        self.model = YOLO(weights)
        if device != 'cpu':
            self.model.to(device)
        self.device = device
        return self.model
    
    def validate_input(self, results: Any) -> bool:
        """Validate Ultralytics Results object."""
        try:
            # Check if it's an Ultralytics Results object
            if not hasattr(results, 'boxes') and not hasattr(results, 'masks'):
                return False
            return True
        except Exception:
            return False
    
    def extract_detections(self, results: Any) -> Detections:
        """Extract standard object detections from Ultralytics results."""
        try:
            if not hasattr(results, 'boxes') or results.boxes is None:
                return Detections(xyxy=np.empty((0, 4)))
            
            boxes = results.boxes.xyxy.cpu().numpy().astype(np.float32)
            confidences = None
            class_ids = None
            
            if hasattr(results.boxes, 'conf') and results.boxes.conf is not None:
                confidences = results.boxes.conf.cpu().numpy().astype(np.float32)
            
            if hasattr(results.boxes, 'cls') and results.boxes.cls is not None:
                class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            return Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=class_ids
            )
            
        except Exception as e:
            warnings.warn(f"Error extracting Ultralytics detections: {e}")
            return Detections(xyxy=np.empty((0, 4)))
    
    def extract_segmentation(self, results: Any) -> Detections:
        """Extract instance segmentation from Ultralytics results."""
        try:
            detections = self.extract_detections(results)
            
            # Add masks if available
            if hasattr(results, 'masks') and results.masks is not None:
                masks = results.masks.data.cpu().numpy()
                detections.mask = masks
            
            return detections
            
        except Exception as e:
            warnings.warn(f"Error extracting Ultralytics segmentation: {e}")
            return Detections(xyxy=np.empty((0, 4)))
    
    def extract_obb(self, results: Any) -> Detections:
        """Extract oriented bounding boxes."""
        try:
            if not hasattr(results, 'obb') or results.obb is None:
                return Detections(xyxy=np.empty((0, 4)))
            
            # Convert OBB to standard xyxy format for compatibility
            obb_boxes = results.obb.xyxyxyxy.cpu().numpy()
            
            # Convert to axis-aligned bounding boxes
            x_coords = obb_boxes[:, [0, 2, 4, 6]]
            y_coords = obb_boxes[:, [1, 3, 5, 7]]
            
            x_min = np.min(x_coords, axis=1)
            y_min = np.min(y_coords, axis=1)
            x_max = np.max(x_coords, axis=1)
            y_max = np.max(y_coords, axis=1)
            
            boxes = np.column_stack([x_min, y_min, x_max, y_max]).astype(np.float32)
            
            confidences = None
            class_ids = None
            
            if hasattr(results.obb, 'conf'):
                confidences = results.obb.conf.cpu().numpy().astype(np.float32)
            if hasattr(results.obb, 'cls'):
                class_ids = results.obb.cls.cpu().numpy().astype(int)
            
            # Store original OBB data
            data = {"obb_coordinates": obb_boxes}
            
            return Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=class_ids,
                data=data
            )
            
        except Exception as e:
            warnings.warn(f"Error extracting Ultralytics OBB: {e}")
            return Detections(xyxy=np.empty((0, 4)))
    
    def process(self, results: Any, task: str = "detection") -> Detections:
        """Enhanced processing with task routing."""
        if not self.validate_input(results):
            raise ValueError("Input not compatible with Ultralytics adapter")
        
        if task == "detection":
            return self.extract_detections(results)
        elif task == "segmentation":
            return self.extract_segmentation(results)
        elif task == "obb":
            return self.extract_obb(results)
        else:
            raise ValueError(f"Unsupported task for Ultralytics: {task}")
