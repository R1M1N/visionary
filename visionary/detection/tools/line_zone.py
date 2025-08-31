"""
Line Zone Analytics for Visionary

This module implements LineZone class for bidirectional crossing detection,
multi-class counting, and lane-specific analytics suitable for traffic management.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional

class LineZone:
    """
    Represents a line zone for counting objects crossing a given line.
    Supports bidirectional crossing detection and multi-class capabilities.
    """
    def __init__(self, p1: Tuple[int, int], p2: Tuple[int, int], zone_id: int = 0):
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.zone_id = zone_id
        self.crossed_counts = {"forward": 0, "backward": 0}
        self.last_positions: Dict[int, np.ndarray] = {}
        self.multi_class_counts: Dict[int, Dict[str, int]] = {}

    def _vector(self):
        return self.p2 - self.p1

    def _line_side(self, point: np.ndarray) -> float:
        """Calculate which side of the line the point lies on."""
        return np.cross(self._vector(), point - self.p1)

    def update(self, detections_xyxy: List[np.ndarray], tracker_ids: List[int], class_ids: Optional[List[int]] = None):
        """
        Update the line zone with new detections.
        Args:
            detections_xyxy: List of bounding boxes in [x1,y1,x2,y2] format
            tracker_ids: List of tracker IDs for each detection
            class_ids: Optional list of class IDs for each detection
        """
        for i, (box, tid) in enumerate(zip(detections_xyxy, tracker_ids)):
            center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            current_side = self._line_side(center)
            last_center = self.last_positions.get(tid)
            if last_center is None:
                self.last_positions[tid] = center
                continue
            last_side = self._line_side(last_center)

            # Detect crossing
            if last_side * current_side < 0:
                direction = "forward" if current_side > 0 else "backward"
                self.crossed_counts[direction] += 1

                if class_ids is not None:
                    class_id = class_ids[i]
                    if class_id not in self.multi_class_counts:
                        self.multi_class_counts[class_id] = {"forward": 0, "backward": 0}
                    self.multi_class_counts[class_id][direction] += 1

            self.last_positions[tid] = center

    def get_counts(self) -> Dict[str, int]:
        return self.crossed_counts

    def get_class_counts(self) -> Dict[int, Dict[str, int]]:
        return self.multi_class_counts


class LaneManager:
    """
    Manage multiple line zones for lane-specific analytics.
    """
    def __init__(self):
        self.zones: Dict[int, LineZone] = {}

    def add_zone(self, p1: Tuple[int, int], p2: Tuple[int, int], zone_id: int):
        zone = LineZone(p1, p2, zone_id)
        self.zones[zone_id] = zone

    def remove_zone(self, zone_id: int):
        if zone_id in self.zones:
            del self.zones[zone_id]

    def update_all(self, detections_xyxy: List[np.ndarray], tracker_ids: List[int], class_ids: Optional[List[int]] = None) -> Dict[int, Dict[str, int]]:
        all_counts = {}
        for zone_id, zone in self.zones.items():
            zone.update(detections_xyxy, tracker_ids, class_ids)
            all_counts[zone_id] = zone.get_counts()
        return all_counts

    def get_counts_for_zone(self, zone_id: int) -> Optional[Dict[str, int]]:
        if zone_id in self.zones:
            return self.zones[zone_id].get_counts()
        return None
