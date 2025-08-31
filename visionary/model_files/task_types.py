from enum import Enum

class TaskType(Enum):
    """Computer vision task types supported by your supervision clone."""
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    KEYPOINT_DETECTION = "keypoint_detection"  # Keep only this one
    TRACKING = "tracking"
    VIDEO_SEGMENTATION = "video_segmentation"
    VIDEO_TRACKING = "video_tracking"
    CLASSIFICATION = "classification"
    DEPTH_ESTIMATION = "depth_estimation"
    OPTICAL_FLOW = "optical_flow"
    # Remove the duplicate KEYPOINT = "keypoint"
