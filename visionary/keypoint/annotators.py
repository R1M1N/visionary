
"""
Pose Annotators for Visionary

Provides keypoint visualization annotators,
skeletal structure drawing, pose confidence visualization,
and temporal pose consistency tracking.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

class KeypointVisualizer:
    def __init__(self, skeleton: List[Tuple[int, int]], keypoint_colors: Optional[Dict[int, Tuple[int,int,int]]] = None):
        """
        Initialize keypoint visualizer.
        Args:
            skeleton: List of keypoint connections (pairs of keypoint indices)
            keypoint_colors: Optional dict mapping keypoint index to BGR color
        """
        self.skeleton = skeleton
        self.keypoint_colors = keypoint_colors or {}

    def draw_keypoints(self, image: np.ndarray, keypoints: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Draw keypoints on image.
        Args:
            image: Input BGR image to draw on
            keypoints: List of (x, y, confidence)
        Returns:
            Image with keypoints drawn
        """
        for idx, (x, y, conf) in enumerate(keypoints):
            if conf > 0.1:
                color = self.keypoint_colors.get(idx, (0, 255, 0))  # default green
                cv2.circle(image, (int(x), int(y)), 5, color, -1)
        return image

    def draw_skeleton(self, image: np.ndarray, keypoints: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Draw skeletal structure by connecting keypoints.
        Args:
            image: Input BGR image
            keypoints: List of (x, y, confidence)
        Returns:
            Image with skeleton drawn
        """
        for connection in self.skeleton:
            idx1, idx2 = connection
            if keypoints[idx1][2] > 0.1 and keypoints[idx2][2] > 0.1:
                pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
                pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
                cv2.line(image, pt1, pt2, (255, 0, 0), 2)  # blue lines
        return image

class PoseConfidenceVisualizer:
    def __init__(self, color_map: Tuple[Tuple[int,int,int], Tuple[int,int,int]] = ((0,0,255), (0,255,0))):
        """
        Visualize pose confidence by overlay color.
        Args:
            color_map: Tuple of colors (low confidence color, high confidence color) in BGR
        """
        self.color_map = color_map

    def visualize_confidence(self, image: np.ndarray, confidence: float) -> np.ndarray:
        """
        Overlay confidence visualization on image.
        Args:
            image: BGR image
            confidence: Confidence value between 0 and 1
        Returns:
            Image with overlay
        """
        alpha = confidence  # opacity based on confidence
        overlay = image.copy()
        color = tuple([int(a*alpha + b*(1-alpha)) for a,b in zip(self.color_map[1], self.color_map[0])])
        cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), color, -1)
        return cv2.addWeighted(overlay, 0.3, image, 0.7, 0)

class TemporalPoseConsistencyTracker:
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.pose_histories: Dict[int, List[List[Tuple[float, float, float]]]] = {}

    def update(self, poses: List[Keypoints]) -> Dict[int, float]:
        """
        Update pose history and compute consistency scores
        Args:
            poses: List of Keypoints objects
        Returns:
            Dict mapping person_id to consistency score (0-1)
        """
        consistency_scores = {}
        for pose in poses:
            pid = pose.person_id if pose.person_id is not None else 0
            history = self.pose_histories.setdefault(pid, [])
            if len(history) >= self.max_history:
                history.pop(0)
            history.append(pose.keypoints)
            # Compute consistency as average +/- std dev of keypoint distances over history
            if len(history) > 1:
                avg_consistency = self._compute_pose_consistency(history)
                consistency_scores[pid] = avg_consistency
            else:
                consistency_scores[pid] = 1.0
        return consistency_scores

    def _compute_pose_consistency(self, history: List[List[Tuple[float, float, float]]]) -> float:
        n = len(history)
        distances = []
        # Compare consecutive poses
        for i in range(1, n):
            d = self._pose_distance(history[i], history[i-1])
            distances.append(d)
        if not distances:
            return 1.0
        mean_dist = np.mean(distances)
        max_dist = 50.0  # heuristic max distance for normalization
        score = max(0.0, 1.0 - mean_dist / max_dist)
        return score

    @staticmethod
    def _pose_distance(pose1: List[Tuple[float, float, float]], pose2: List[Tuple[float, float, float]]) -> float:
        distances = []
        for kp1, kp2 in zip(pose1, pose2):
            if kp1[2] > 0.1 and kp2[2] > 0.1:
                dist = np.linalg.norm(np.array(kp1[:2]) - np.array(kp2[:2]))
                distances.append(dist)
        return np.mean(distances) if distances else 0.0
