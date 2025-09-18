from typing import List, Optional, Dict, Any
from .task_types import TaskType


class TaskConfig:
    """Configuration for specific computer vision tasks."""
    
    def __init__(self, 
                 task: TaskType,
                 confidence_threshold: float = 0.5,
                 classes: Optional[List[str]] = None,
                 **kwargs):
        self.task = task
        self.confidence_threshold = confidence_threshold
        self.classes = classes
        self.extra_params = kwargs
        
        # Task-specific parameters
        if task == TaskType.KEYPOINT_DETECTION:
            self.keypoint_threshold = kwargs.get('keypoint_threshold', 0.5)
            self.num_keypoints = kwargs.get('num_keypoints', len(classes) if classes else 17)
            self.skeleton_connections = kwargs.get('skeleton_connections', None)
            
        elif task == TaskType.TRACKING:
            self.max_distance = kwargs.get('max_distance', 100)
            self.max_disappeared = kwargs.get('max_disappeared', 30)
            self.tracker_type = kwargs.get('tracker_type', 'centroid')
            
        elif task in [TaskType.SEGMENTATION, TaskType.INSTANCE_SEGMENTATION]:
            self.mask_threshold = kwargs.get('mask_threshold', 0.5)
            self.min_area = kwargs.get('min_area', 100)
