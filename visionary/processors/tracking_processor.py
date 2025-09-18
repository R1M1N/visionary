from typing import Dict, Any, Union, Optional
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics import YOLO
from .base import BaseProcessor


class TrackingProcessor(BaseProcessor):
    """Processor for real YOLO object tracking."""
    
    def __init__(self, model_config, input_type):
        """Initialize the TrackingProcessor with model configuration and input type."""
        super().__init__(model_config, input_type)
        self.track_history = {}
        self.next_track_id = 1
        self._load_model()
    
    def _load_model(self):
        """Load the actual YOLO tracking model."""
        model_type = getattr(self.model_config, 'model_type', 'unknown')
        print(f"Loading tracking model: {model_type}")
        
        # Load the actual YOLO model for tracking
        model_path = f"{model_type.value}.pt"
        self.model = YOLO(model_path)
        print(f"✅ YOLO model loaded: {model_path}")
    
    def process(self, input_data: Union[str, Path, np.ndarray, Image.Image], 
                task_config: Optional[object] = None) -> Dict[str, Any]:
        """Process input with real YOLO tracking."""
        try:
            # Run YOLO tracking on the input video
            results = self.model.track(str(input_data), persist=True)
            
            tracks = []
            frame_tracks = {}
            
            for frame_idx, result in enumerate(results):
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    track_ids = result.boxes.id.cpu().numpy().astype(np.int32)
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(np.int32)
                    
                    # Initialize frame tracking
                    if frame_idx not in frame_tracks:
                        frame_tracks[frame_idx] = []
                    
                    # Convert to structured format
                    for box, track_id, score, cls_id in zip(boxes, track_ids, scores, classes):
                        track_data = {
                            'track_id': int(track_id),
                            'bbox': box.tolist(),
                            'confidence': float(score),
                            'class': self.model.names[cls_id],
                            'frame': frame_idx,
                            'state': 'confirmed'
                        }
                        tracks.append(track_data)
                        frame_tracks[frame_idx].append(track_data)
            
            # Update track history with real data
            self._update_track_history_real(tracks)
            
            # Add trajectories for established tracks
            self._add_trajectories_real(tracks)
            
            return {
                'tracks': tracks,
                'frame_tracks': frame_tracks,
                'total_tracks': len(set(t['track_id'] for t in tracks)),
                'active_tracks': len(set(t['track_id'] for t in tracks)),
                'total_frames': len(frame_tracks),
                'tracking_metrics': {
                    'avg_confidence': np.mean([t['confidence'] for t in tracks]) if tracks else 0.0,
                    'tracks_per_frame': len(tracks) / max(1, len(frame_tracks))
                }
            }
            
        except Exception as e:
            print(f"Error in tracking process: {e}")
            return {
                'error': str(e),
                'tracks': [],
                'frame_tracks': {},
                'total_tracks': 0,
                'active_tracks': 0
            }
    
    def _update_track_history_real(self, tracks: list):
        """Update tracking history for real tracks."""
        for track in tracks:
            track_id = track['track_id']
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            self.track_history[track_id].append({
                'frame_id': track['frame'],
                'bbox': track['bbox'],
                'confidence': track['confidence']
            })
            
            # Keep only last 50 frames to manage memory
            if len(self.track_history[track_id]) > 50:
                self.track_history[track_id] = self.track_history[track_id][-50:]
    
    def _add_trajectories_real(self, tracks: list):
        """Add trajectory information for long-running tracks."""
        for track in tracks:
            track_id = track['track_id']
            if track_id in self.track_history and len(self.track_history[track_id]) > 5:
                # Get recent positions for trajectory
                recent_positions = self.track_history[track_id][-10:]
                track['trajectory'] = [
                    [int((pos['bbox'][0] + pos['bbox'][2]) / 2),  # center_x
                     int((pos['bbox'][1] + pos['bbox'][3]) / 2)]  # center_y
                    for pos in recent_positions
                ]
    
    def reset_tracks(self):
        """Reset all tracking state."""
        self.track_history.clear()
        self.next_track_id = 1
    
    def get_track_count(self) -> int:
        """Get the current number of active tracks."""
        return len(self.track_history)
    
    def get_track_history(self, track_id: int) -> list:
        """Get history for a specific track."""
        return self.track_history.get(track_id, [])
