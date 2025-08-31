"""
Enhanced Hugging Face Transformers adapter for Visionary.
"""

import numpy as np
from typing import Any, List, Dict, Union, Optional
from .base import BaseAdapter
from ..core import Detections
import warnings


class TransformersAdapter(BaseAdapter):
    """
    Enhanced adapter for Hugging Face Transformers models.
    
    Supports:
    - DETR (Detection Transformer)
    - RT-DETR
    - YOLOS
    - Vision Language Models (VLMs)
    """
    
    def __init__(self, model_type: str = "detr"):
        super().__init__(f"Transformers ({model_type})")
        self.model_type = model_type.lower()
        self.supported_tasks = ["detection", "segmentation", "vlm"]
    
    def validate_input(self, results: Any) -> bool:
        """Validate Transformers model output."""
        try:
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], dict) and 'bbox' in results[0]:
                    return True
            
            if isinstance(results, dict) and 'predictions' in results:
                return True
                
            return False
        except Exception:
            return False
    
    def extract_detections(self, results: Any) -> Detections:
        """Extract detections from various Transformers model formats."""
        try:
            # Handle list of detection dictionaries (standard format)
            if isinstance(results, list):
                return self._process_detection_list(results)
            
            # Handle model output dictionary
            elif isinstance(results, dict):
                return self._process_model_output(results)
            
            else:
                warnings.warn("Unknown Transformers output format")
                return Detections(xyxy=np.empty((0, 4)))
                
        except Exception as e:
            warnings.warn(f"Error extracting Transformers detections: {e}")
            return Detections(xyxy=np.empty((0, 4)))
    
    def _process_detection_list(self, detections: List[Dict]) -> Detections:
        """Process list of detection dictionaries."""
        if not detections:
            return Detections(xyxy=np.empty((0, 4)))
        
        boxes = []
        confidences = []
        class_ids = []
        
        for det in detections:
            # Handle different bbox formats
            bbox = det.get('bbox', det.get('box', None))
            if bbox is None:
                continue
                
            # Convert bbox format if needed
            if len(bbox) == 4:
                # Check if it's xywh or xyxy format
                if 'format' in det and det['format'] == 'xywh':
                    x, y, w, h = bbox
                    bbox = [x, y, x + w, y + h]
                
                boxes.append(bbox)
                confidences.append(det.get('score', det.get('confidence', 1.0)))
                class_ids.append(det.get('label', det.get('class_id', 0)))
        
        if not boxes:
            return Detections(xyxy=np.empty((0, 4)))
        
        return Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            confidence=np.array(confidences, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int)
        )
    
    def _process_model_output(self, output: Dict) -> Detections:
        """Process raw model output dictionary."""
        if 'predictions' in output:
            predictions = output['predictions']
            if isinstance(predictions, list):
                return self._process_detection_list(predictions)
        
        # Handle DETR-style output
        if 'pred_boxes' in output and 'scores' in output:
            boxes = output['pred_boxes']
            scores = output['scores']
            labels = output.get('pred_classes', output.get('labels', None))
            
            # Convert tensors to numpy if needed
            if hasattr(boxes, 'cpu'):
                boxes = boxes.cpu().numpy()
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()
            if labels is not None and hasattr(labels, 'cpu'):
                labels = labels.cpu().numpy()
            
            # Filter by confidence threshold
            if len(scores.shape) > 1:
                # Multi-class scores - take max
                confidences = np.max(scores, axis=1)
                class_ids = np.argmax(scores, axis=1)
            else:
                confidences = scores
                class_ids = labels if labels is not None else np.zeros(len(scores))
            
            return Detections(
                xyxy=boxes.astype(np.float32),
                confidence=confidences.astype(np.float32),
                class_id=class_ids.astype(int)
            )
        
        return Detections(xyxy=np.empty((0, 4)))
    
    def extract_segmentation(self, results: Any) -> Detections:
        """Extract segmentation masks (if available)."""
        detections = self.extract_detections(results)
        
        # Add mask extraction logic for segmentation models
        if isinstance(results, dict) and 'masks' in results:
            masks = results['masks']
            if hasattr(masks, 'cpu'):
                masks = masks.cpu().numpy()
            detections.mask = masks
        
        return detections
