"""
Polygon Zone Analysis for Visionary

This module implements PolygonZone class to define custom polygonal zones
and perform real-time analytics like object counting within multiple zones.
"""

import numpy as np
from typing import List, Tuple, Dict

class PolygonZone:
    """
    Represents a polygon zone for spatial analytics.
    Supports multi-zone management and real-time object counting.
    """
    def __init__(self, polygon_points: List[Tuple[int, int]], zone_id: int = 0):
        self.polygon = np.array(polygon_points, dtype=np.int32)
        self.zone_id = zone_id
        self.object_counts = 0

    def contains_point(self, point: Tuple[int, int]) -> bool:
        """
        Check if a point is inside the polygon.
        Uses cv2 pointPolygonTest for robust point-in-polygon check.
        """
        import cv2
        return cv2.pointPolygonTest(self.polygon, point, False) >= 0

    def count_objects_in_zone(self, detections_xyxy: List[np.ndarray]) -> int:
        """
        Count objects whose centers are inside the polygon zone.
        Args:
            detections_xyxy: List of bounding boxes in [x1,y1,x2,y2] format
        Returns:
            Number of objects inside this zone
        """
        count = 0
        for box in detections_xyxy:
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            if self.contains_point((center_x, center_y)):
                count += 1
        self.object_counts = count
        return count


class MultiPolygonZoneManager:
    """
    Manage multiple polygon zones with independent analytics.
    """
    def __init__(self):
        self.zones: Dict[int, PolygonZone] = {}

    def add_zone(self, polygon_points: List[Tuple[int, int]], zone_id: int):
        zone = PolygonZone(polygon_points, zone_id)
        self.zones[zone_id] = zone

    def remove_zone(self, zone_id: int):
        if zone_id in self.zones:
            del self.zones[zone_id]

    def update_counts(self, detections_xyxy: List[np.ndarray]) -> Dict[int, int]:
        counts = {}
        for zone_id, zone in self.zones.items():
            counts[zone_id] = zone.count_objects_in_zone(detections_xyxy)
        return counts

    def get_zone_count(self, zone_id: int) -> int:
        if zone_id in self.zones:
            return self.zones[zone_id].object_counts
        return 0
