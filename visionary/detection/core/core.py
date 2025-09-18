
import numpy as np
from typing import Optional, List, Dict, Any, Union, Tuple
import warnings

class Detections:
    """
    Central data structure for standardizing object detection results.

    This class provides a unified interface for handling detection results
    from various computer vision models, ensuring consistency across different
    frameworks like Ultralytics, Transformers, MMDetection, etc.
    """

    def __init__(
        self,
        xyxy: np.ndarray,
        mask: Optional[np.ndarray] = None,
        confidence: Optional[np.ndarray] = None,
        class_id: Optional[np.ndarray] = None,
        tracker_id: Optional[np.ndarray] = None,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Detections object.

        Args:
            xyxy: np.ndarray of shape (N, 4) - bounding boxes in (x1, y1, x2, y2) format
            mask: Optional masks of shape (N, H, W) for instance segmentation
            confidence: Confidence scores for each detection (N,)
            class_id: Class IDs for each detection (N,)
            tracker_id: Tracker IDs for consistent object identities (N,)
            data: Dictionary for additional metadata per detection
        """
        self.xyxy = xyxy.astype(np.float32) if xyxy is not None else np.empty((0, 4), dtype=np.float32)
        self.mask = mask

        n_detections = self.xyxy.shape[0]
        self.confidence = confidence if confidence is not None else np.ones(n_detections, dtype=np.float32)
        self.class_id = class_id if class_id is not None else np.zeros(n_detections, dtype=int)
        self.tracker_id = tracker_id
        self.data = data if data is not None else {}

        # Validate dimensions
        self._validate_dimensions()

    def _validate_dimensions(self) -> None:
        """Validate that all arrays have consistent dimensions."""
        n = len(self.xyxy)

        if len(self.confidence) != n:
            raise ValueError(f"Confidence array length {len(self.confidence)} doesn't match boxes {n}")
        if len(self.class_id) != n:
            raise ValueError(f"Class ID array length {len(self.class_id)} doesn't match boxes {n}")
        if self.tracker_id is not None and len(self.tracker_id) != n:
            raise ValueError(f"Tracker ID array length {len(self.tracker_id)} doesn't match boxes {n}")
        if self.mask is not None and self.mask.shape[0] != n:
            raise ValueError(f"Mask array length {self.mask.shape[0]} doesn't match boxes {n}")

    @classmethod
    def from_ultralytics(cls, results: Any) -> 'Detections':
        """
        Create Detections from Ultralytics YOLO results.

        Args:
            results: Ultralytics Results object

        Returns:
            Detections object with standardized format
        """
        if not hasattr(results, 'boxes') or results.boxes is None:
            return cls(xyxy=np.empty((0, 4)))

        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy() if hasattr(results.boxes, "conf") else None
        class_ids = results.boxes.cls.cpu().numpy().astype(int) if hasattr(results.boxes, "cls") else None

        # Handle masks for segmentation
        masks = None
        if hasattr(results, 'masks') and results.masks is not None:
            masks = results.masks.data.cpu().numpy()

        return cls(
            xyxy=boxes,
            confidence=confidences,
            class_id=class_ids,
            mask=masks
        )

    @classmethod
    def from_transformers(cls, detections: List[Dict[str, Any]]) -> 'Detections':
        """
        Create Detections from Hugging Face Transformers results.

        Args:
            detections: List of detection dictionaries with keys: bbox, score, label

        Returns:
            Detections object with standardized format
        """
        if not detections:
            return cls(xyxy=np.empty((0, 4)))

        boxes = np.array([d["bbox"] for d in detections])
        confidences = np.array([d["score"] for d in detections])
        class_ids = np.array([d["label"] for d in detections])

        return cls(xyxy=boxes, confidence=confidences, class_id=class_ids)

    @classmethod
    def from_mmdet(cls, detections_bbox: np.ndarray, detections_label: np.ndarray) -> 'Detections':
        """
        Create Detections from MMDetection results.

        Args:
            detections_bbox: Array of shape (N, 5) containing [x1, y1, x2, y2, score]
            detections_label: Array of shape (N,) containing class labels

        Returns:
            Detections object with standardized format
        """
        if detections_bbox.size == 0:
            return cls(xyxy=np.empty((0, 4)))

        boxes = detections_bbox[:, :4]
        confidences = detections_bbox[:, 4]
        class_ids = detections_label.astype(int)

        return cls(xyxy=boxes, confidence=confidences, class_id=class_ids)

    @classmethod
    def from_inference(cls, raw_results: Any) -> 'Detections':
        """
        Create Detections from general inference results.

        This is a flexible method that can be customized for different
        inference engines and formats.

        Args:
            raw_results: Raw inference results in any format

        Returns:
            Detections object with standardized format
        """
        # Placeholder implementation - customize based on your inference format
        if hasattr(raw_results, 'predictions'):
            # Example for a common inference format
            predictions = raw_results.predictions
            boxes = np.array([[p.x1, p.y1, p.x2, p.y2] for p in predictions])
            confidences = np.array([p.confidence for p in predictions])
            class_ids = np.array([p.class_id for p in predictions])
            return cls(xyxy=boxes, confidence=confidences, class_id=class_ids)

        return cls(xyxy=np.empty((0, 4)))

    # Add to Detections class
    @classmethod
    def from_model(cls, results: Any, model_type: Optional[str] = None) -> 'Detections':
        """
        Create Detections from any supported model using adapters.
        
        Args:
            results: Model results
            model_type: Optional model type hint
            
        Returns:
            Detections object
        """
        from .adapters import AdapterFactory
        
        if model_type:
            adapter = AdapterFactory.create_adapter(model_type)
        else:
            adapter = AdapterFactory.auto_detect_adapter(results)
            
        if adapter is None:
            raise ValueError("Could not detect compatible adapter for results")
        
        return adapter.extract_detections(results)


    def filter(self, mask: Union[np.ndarray, List[bool]]) -> 'Detections':
        """
        Filter detections using a boolean mask.

        Args:
            mask: Boolean array or list indicating which detections to keep

        Returns:
            Filtered Detections object
        """
        mask = np.array(mask, dtype=bool)

        return Detections(
            xyxy=self.xyxy[mask],
            confidence=self.confidence[mask],
            class_id=self.class_id[mask],
            mask=self.mask[mask] if self.mask is not None else None,
            tracker_id=self.tracker_id[mask] if self.tracker_id is not None else None,
            data={k: np.array(v)[mask] if isinstance(v, (list, np.ndarray)) else v 
                  for k, v in self.data.items()}
        )

    def with_threshold(self, threshold: float) -> 'Detections':
        """
        Filter detections by confidence threshold.

        Args:
            threshold: Minimum confidence score to keep

        Returns:
            Filtered Detections object
        """
        mask = self.confidence >= threshold
        return self.filter(mask)

    @staticmethod
    def _calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union (IoU) between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def with_nms(self, iou_threshold: float = 0.5) -> 'Detections':
        """
        Apply Non-Maximum Suppression to filter overlapping detections.

        Args:
            iou_threshold: IoU threshold for considering boxes as overlapping

        Returns:
            Filtered Detections object with NMS applied
        """
        if len(self) == 0:
            return self

        # Sort by confidence (descending)
        indices = np.argsort(-self.confidence)
        keep = []

        while len(indices) > 0:
            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            # Calculate IoU with remaining boxes
            ious = np.array([
                self._calculate_iou(self.xyxy[current], self.xyxy[i]) 
                for i in indices[1:]
            ])

            # Keep only boxes with IoU below threshold
            indices = indices[1:][ious < iou_threshold]

        keep = np.array(keep)
        return Detections(
            xyxy=self.xyxy[keep],
            confidence=self.confidence[keep],
            class_id=self.class_id[keep],
            mask=self.mask[keep] if self.mask is not None else None,
            tracker_id=self.tracker_id[keep] if self.tracker_id is not None else None,
            data={k: np.array(v)[keep] if isinstance(v, (list, np.ndarray)) else v 
                  for k, v in self.data.items()}
        )

    def area(self, index: Optional[int] = None) -> Union[np.ndarray, float]:
        """
        Calculate area of bounding boxes.

        Args:
            index: If provided, calculate area for specific detection

        Returns:
            Area(s) of the bounding box(es)
        """
        if index is not None:
            box = self.xyxy[index]
            width = max(0, box[2] - box[0])
            height = max(0, box[3] - box[1])
            return float(width * height)
        else:
            widths = np.maximum(0, self.xyxy[:, 2] - self.xyxy[:, 0])
            heights = np.maximum(0, self.xyxy[:, 3] - self.xyxy[:, 1])
            return widths * heights

    def get_anchors(self, anchor_strategy: str = "center") -> np.ndarray:
        """
        Get anchor points for each detection.

        Args:
            anchor_strategy: Strategy for anchor point calculation
                - "center": Center of bounding box
                - "bottom_center": Bottom center of bounding box
                - "top_left": Top left corner

        Returns:
            Array of anchor points with shape (N, 2)
        """
        if anchor_strategy == "center":
            centers_x = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2
            centers_y = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2
            return np.column_stack([centers_x, centers_y])

        elif anchor_strategy == "bottom_center":
            centers_x = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2
            bottom_y = self.xyxy[:, 3]
            return np.column_stack([centers_x, bottom_y])

        elif anchor_strategy == "top_left":
            return self.xyxy[:, :2]

        else:
            raise ValueError(f"Unknown anchor strategy: {anchor_strategy}")

    def crop_image(self, image: np.ndarray, index: int, pad: int = 0) -> np.ndarray:
        """
        Crop image according to bounding box.

        Args:
            image: Input image as numpy array
            index: Index of detection to crop
            pad: Padding around the bounding box

        Returns:
            Cropped image region
        """
        if index >= len(self):
            raise IndexError(f"Index {index} out of range for {len(self)} detections")

        box = self.xyxy[index].astype(int)
        h, w = image.shape[:2]

        # Apply padding and clamp to image boundaries
        x1 = max(0, box[0] - pad)
        y1 = max(0, box[1] - pad)
        x2 = min(w, box[2] + pad)
        y2 = min(h, box[3] + pad)

        return image[y1:y2, x1:x2]

    def to_dict(self) -> Dict[str, Any]:
        """Convert Detections to dictionary format."""
        result = {
            'xyxy': self.xyxy.tolist(),
            'confidence': self.confidence.tolist(),
            'class_id': self.class_id.tolist(),
        }

        if self.mask is not None:
            result['mask'] = self.mask.tolist()
        if self.tracker_id is not None:
            result['tracker_id'] = self.tracker_id.tolist()
        if self.data:
            result['data'] = self.data

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Detections':
        """Create Detections from dictionary format."""
        return cls(
            xyxy=np.array(data['xyxy']),
            confidence=np.array(data.get('confidence', [])),
            class_id=np.array(data.get('class_id', [])),
            mask=np.array(data['mask']) if 'mask' in data else None,
            tracker_id=np.array(data['tracker_id']) if 'tracker_id' in data else None,
            data=data.get('data', {})
        )

    def __len__(self) -> int:
        """Return number of detections."""
        return self.xyxy.shape[0]

    def __getitem__(self, index: Union[int, slice, np.ndarray]) -> 'Detections':
        """Support indexing and slicing."""
        if isinstance(index, int):
            if index < 0:
                index = len(self) + index
            if index >= len(self) or index < 0:
                raise IndexError(f"Index {index} out of range")
            index = slice(index, index + 1)

        return Detections(
            xyxy=self.xyxy[index],
            confidence=self.confidence[index],
            class_id=self.class_id[index],
            mask=self.mask[index] if self.mask is not None else None,
            tracker_id=self.tracker_id[index] if self.tracker_id is not None else None,
            data={k: np.array(v)[index] if isinstance(v, (list, np.ndarray)) else v 
                  for k, v in self.data.items()}
        )

    def __repr__(self) -> str:
        """String representation of Detections."""
        return f"<Detections(n={len(self)})>"
    def __str__(self) -> str:
        """Human-readable string representation."""
        if len(self) == 0:
            return "Detections: empty"
        
        conf_stats = f"conf: {self.confidence.min():.3f}-{self.confidence.max():.3f}"
        class_info = f"classes: {len(np.unique(self.class_id))}"
        
        return f"Detections: {len(self)} boxes, {conf_stats}, {class_info}"


# Utility functions (outside the class)
def box_iou_batch(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculate IoU between two sets of boxes in batch.
    
    Args:
        boxes1: Array of shape (N, 4) in xyxy format
        boxes2: Array of shape (M, 4) in xyxy format
        
    Returns:
        IoU matrix of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Intersection coordinates
    inter_x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0])
    inter_y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1])
    inter_x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2])
    inter_y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3])

    # Intersection area
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Union area
    union_area = area1[:, None] + area2 - inter_area

    return inter_area / union_area


def merge_detections(detections_list: List[Detections]) -> Detections:
    """
    Merge multiple Detections objects into one.
    
    Args:
        detections_list: List of Detections objects to merge
        
    Returns:
        Merged Detections object
    """
    if not detections_list:
        return Detections(xyxy=np.empty((0, 4)))
    
    if len(detections_list) == 1:
        return detections_list[0]
    
    # Concatenate all arrays
    xyxy = np.vstack([det.xyxy for det in detections_list])
    confidence = np.hstack([det.confidence for det in detections_list])
    class_id = np.hstack([det.class_id for det in detections_list])
    
    # Handle optional attributes
    masks = None
    if all(det.mask is not None for det in detections_list):
        masks = np.vstack([det.mask for det in detections_list])
    
    tracker_ids = None
    if all(det.tracker_id is not None for det in detections_list):
        tracker_ids = np.hstack([det.tracker_id for det in detections_list])
    
    # Merge data dictionaries
    merged_data = {}
    for det in detections_list:
        for key, value in det.data.items():
            if key not in merged_data:
                merged_data[key] = []
            merged_data[key].extend(value if isinstance(value, list) else [value])
    
    return Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        mask=masks,
        tracker_id=tracker_ids,
        data=merged_data
    )