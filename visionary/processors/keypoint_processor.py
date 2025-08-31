from typing import Dict, Any, Union, List
from pathlib import Path
import numpy as np
from PIL import Image
from .base import BaseProcessor

class KeypointProcessor(BaseProcessor):
    """Processor for keypoint detection and pose estimation tasks."""
    
    def __init__(self, config):
        """
        Initialize the KeypointProcessor.
        
        Args:
            config: Can be either TaskConfig (for factory pattern) or model_config (for direct instantiation)
        """
        super().__init__(config)
        
        # Handle both factory pattern (TaskConfig) and direct instantiation (model_config)
        if hasattr(config, 'confidence_threshold'):
            # New factory pattern approach
            self.confidence_threshold = config.confidence_threshold
            self.keypoint_names = config.classes or []
            self.task_type = config.task
        else:
            # Backward compatibility with your existing approach
            self.confidence_threshold = 0.5
            self.keypoint_names = []
            self.task_type = "keypoint_detection"
        
    def _load_model(self):  # ← Changed from load_model to _load_model
        """Load keypoint detection model."""
        print(f"Loading keypoint model: {self.model_config.model_type}")
        self.model = "mock_keypoint_model"
        
    
    def process(self, input_data: Union[str, Path, np.ndarray, Image.Image], 
                task_config=None) -> Dict[str, Any]:
        """
        Process keypoint detection on image data.
        
        Args:
            input_data: Image data as numpy array or path to image file
            task_config: Optional task configuration (for backward compatibility)
            
        Returns:
            Dictionary containing detected keypoints and metadata
        """
        if not self.validate_input(input_data):
            raise ValueError("Invalid input data for keypoint detection")
            
        image = self.preprocess(input_data)
        
        # Use configured keypoint names or fall back to COCO-style defaults
        keypoint_names = self.keypoint_names if self.keypoint_names else [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # TODO: Replace this with actual model inference
        # keypoints_data, confidence_scores = self.model.predict(image)
        
        # Initialize empty results structure - to be populated by actual model
        results = {
            'task_type': str(self.task_type).replace('TaskType.', '').lower() if hasattr(self.task_type, 'value') else str(self.task_type),
            'keypoints': [],  # Will be populated with detected keypoint coordinates
            'confidence_scores': [],  # Will be populated with confidence values
            'keypoint_names': keypoint_names,
            'num_persons': 0,  # Will be set based on actual detections
            'image_size': image.size if hasattr(image, 'size') else (224, 224),
            'metadata': {
                'confidence_threshold': self.confidence_threshold,
                'num_keypoints_detected': 0  # Will be updated with actual count
            }
        }
        
        return results
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data for keypoint detection."""
        if isinstance(data, np.ndarray):
            return len(data.shape) >= 2  # At least 2D array (image)
        elif isinstance(data, (str, Path)):
            path_str = str(data).lower()
            return path_str.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))
        elif isinstance(data, Image.Image):
            return True
        return False
    
    def postprocess(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process raw model results into standardized format.
        
        Args:
            raw_results: Raw output from the keypoint detection model
            
        Returns:
            Processed results in standard format
        """
        # TODO: Implement when you have actual model output
        # This is where you'd convert raw model output to the standard format
        return raw_results
    
    def get_skeleton_connections(self) -> List[tuple]:
        """
        Get skeleton connections for COCO-style keypoints.
        
        Returns:
            List of tuples representing connections between keypoints
        """
        # COCO-style skeleton connections (17 keypoints)
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head connections
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (11, 12), (5, 11), (6, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        return skeleton
    
    def filter_keypoints_by_confidence(self, keypoints_data: List[Dict], 
                                     min_confidence: float = None) -> List[Dict]:
        """
        Filter keypoints based on confidence threshold.
        
        Args:
            keypoints_data: List of keypoint dictionaries
            min_confidence: Minimum confidence threshold (uses self.confidence_threshold if None)
            
        Returns:
            Filtered keypoints data
        """
        threshold = min_confidence if min_confidence is not None else self.confidence_threshold
        
        filtered_data = []
        for person in keypoints_data:
            if person.get('confidence', 0) >= threshold:
                filtered_person = person.copy()
                # Filter individual keypoints by confidence
                if 'keypoints' in filtered_person:
                    filtered_person['keypoints'] = [
                        kp for kp in person['keypoints'] 
                        if len(kp) > 2 and kp[2] >= threshold
                    ]
                filtered_data.append(filtered_person)
        
        return filtered_data
