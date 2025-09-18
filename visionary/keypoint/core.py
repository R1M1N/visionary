"""
KeyPoint Core System for Visionary

Provides KeyPoints class with confidence scores,
multi-person pose support, and pose-to-detection conversion.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np

class KeyPoints:
    def __init__(self, keypoints: List[Tuple[float, float, float]], person_id: Optional[int] = None):
        """
        Initialize keypoints object.
        Args:
            keypoints: List of (x, y, confidence) tuples for keypoints
            person_id: Optional ID for multi-person tracking
        """
        self.keypoints = keypoints
        self.person_id = person_id

    def get_confidence(self) -> float:
        """Average confidence over all keypoints."""
        confidences = [kp[2] for kp in self.keypoints]
        return float(np.mean(confidences)) if confidences else 0.0

    def to_detection(self) -> Dict[str, any]:
        """
        Convert keypoints to bounding box detection.
        Returns:
            dict with bbox and score
        """
        xs = [kp[0] for kp in self.keypoints if kp[2] > 0]
        ys = [kp[1] for kp in self.keypoints if kp[2] > 0]
        if not xs or not ys:
            return None
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        score = self.get_confidence()
        bbox = [x_min, y_min, x_max, y_max]
        return {"bbox": bbox, "score": score, "person_id": self.person_id}

class PoseEstimator:
    def __init__(self):
        self.poses: List[KeyPoints] = []

    def add_pose(self, keypoints: List[Tuple[float, float, float]], person_id: Optional[int] = None):
        self.poses.append(KeyPoints(keypoints, person_id))

    def get_detections(self) -> List[Dict[str, any]]:
        detections = []
        for pose in self.poses:
            detection = pose.to_detection()
            if detection is not None:
                detections.append(detection)
        return detections

    def clear(self):
        self.poses.clear()
