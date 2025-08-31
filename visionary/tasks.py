from enum import Enum
from typing import List, Optional
import warnings
from visionary.model_files.task_types import TaskType  # Changed from ....model_files.task_types
from visionary.model_files.task_config import TaskConfig  # Changed from ...model_files.task_config

warnings.warn(
    "Importing from visionary.tasks is deprecated. "
    "Use 'from visionary.models import TaskType, TaskConfig' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["TaskType", "TaskConfig"]


# class TaskType(Enum):
#     """Computer vision task types supported by Visionary."""
#     DETECTION = "detection"
#     SEGMENTATION = "segmentation"
#     INSTANCE_SEGMENTATION = "instance_segmentation"
#     KEYPOINT_DETECTION = "keypoint_detection"
#     TRACKING = "tracking"
#     VIDEO_SEGMENTATION = "video_segmentation"
#     VIDEO_TRACKING = "video_tracking"
#     CLASSIFICATION = "classification"
#     DEPTH_ESTIMATION = "depth_estimation"
#     OPTICAL_FLOW = "optical_flow"
#     # KEYPOINT = "keypoint"

# class TaskConfig:
#     """Configuration for specific tasks."""
#     def __init__(self, 
#                  task: TaskType,
#                  confidence_threshold: float = 0.5,
#                  classes: Optional[List[str]] = None,
#                  **kwargs):
#         self.task = task
#         self.confidence_threshold = confidence_threshold
#         self.classes = classes
#         self.extra_params = kwargs
