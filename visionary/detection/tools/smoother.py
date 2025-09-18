"""
Detection Smoothing for Visionary

Implements temporal smoothing across frames,
confidence-based filtering, and bounding box stabilization.
"""

from typing import List, Dict, Tuple
import numpy as np

class DetectionSmoother:
    def __init__(self, max_history: int = 5, confidence_threshold: float = 0.5):
        """
        Args:
            max_history: Number of prior frames to consider for smoothing
            confidence_threshold: Minimum confidence to accept detection
        """
        self.max_history = max_history
        self.confidence_threshold = confidence_threshold
        self.history: Dict[int, List[np.ndarray]] = {}

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update smoother with new detections for the current frame.
        Args:
            detections: List of detections, each dict containing keys:
                'track_id', 'bbox' (x1,y1,x2,y2), 'confidence'
        Returns:
            List of smoothed detections
        """
        smoothed_detections = []
        for det in detections:
            track_id = det['track_id']
            bbox = np.array(det['bbox'], dtype=np.float32)
            confidence = det.get('confidence', 1.0)

            # Confidence filtering
            if confidence < self.confidence_threshold:
                continue

            # Update history for track
            if track_id not in self.history:
                self.history[track_id] = []
            self.history[track_id].append(bbox)
            if len(self.history[track_id]) > self.max_history:
                self.history[track_id].pop(0)

            # Compute smoothed bounding box by averaging history
            smoothed_bbox = np.mean(self.history[track_id], axis=0)

            smoothed_detections.append({
                'track_id': track_id,
                'bbox': smoothed_bbox.tolist(),
                'confidence': confidence
            })

        return smoothed_detections

    def reset(self):
        self.history.clear()
