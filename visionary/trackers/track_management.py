
from typing import List, Optional
import numpy as np

from .byte_track import ByteTracker, Track, TrackState, KalmanFilter

class TrackManager:
    def __init__(self, 
                 max_occlusion_frames: int = 30, 
                 min_activation_confidence: float = 0.6,
                 max_time_lost: int = 30):
        self.byte_tracker = ByteTracker(max_time_lost=max_time_lost)
        self.max_occlusion_frames = max_occlusion_frames
        self.min_activation_confidence = min_activation_confidence

        # Maintain an internal mapping of track IDs keyed by class id
        self.active_tracks = []

    def update(self, detections: List[np.ndarray], scores: List[float], features: List[np.ndarray], class_ids: List[int]):
        # Run ByteTracker update with detections
        self.byte_tracker.update(detections, scores, features)

        # Get predicted tracks and update track states with occlusion handling
        for track in self.byte_tracker.tracks:
            # Occlusion handling: tracks lost longer than max occlusion frames marked as removed
            if track.state == TrackState.Lost and track.time_since_update > self.max_occlusion_frames:
                track.state = TrackState.Removed

        # Confidence-based activation
        for i, track in enumerate(self.byte_tracker.tracks):
            # Activate tracks meeting confidence threshold
            if track.state == TrackState.Lost and i < len(scores) and scores[i] > self.min_activation_confidence:
                track.state = TrackState.Active
                track.time_since_update = 0

        # Save current active tracks
        self.active_tracks = [t for t in self.byte_tracker.tracks if t.state == TrackState.Active]

    def get_active_tracks(self) -> List[Track]:
        """Get list of active tracks."""
        return self.active_tracks

    def get_tracks_by_class(self, class_id: int) -> List[Track]:
        """Get list of active tracks by class ID."""
        filtered_tracks = []
        for track in self.active_tracks:
            # Assuming Track class supports class_id, else extend Track class
            if hasattr(track, 'class_id') and track.class_id == class_id:
                filtered_tracks.append(track)
        return filtered_tracks

    # Additional methods as needed...
