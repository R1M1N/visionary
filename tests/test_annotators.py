
import cv2
import numpy as np
import unittest
from visionary.annotators.base import BaseAnnotator
from visionary.annotators.geometric import *
from visionary.annotators.advanced_visual import *
from visionary.annotators.temporal_privacy import *

class DummyDetections:
    """Dummy detections class for testing."""
    def __init__(self):
        self.xyxy = np.array([[20, 20, 80, 80], [100, 100, 160, 160]], dtype=np.float32)
        self.class_id = np.array([0, 1])
        self.confidence = np.array([0.8, 0.95])
        self.tracker_id = np.array([1, 2])
        self.mask = np.array([
            np.zeros((200, 200), dtype=np.uint8),
            np.zeros((200, 200), dtype=np.uint8)
        ])
        self.mask[0][20:81, 20:81] = 1
        self.mask[1][100:161, 100:161] = 1

    def __len__(self):
        return len(self.xyxy)

class AnnotatorsTestCase(unittest.TestCase):
    def setUp(self):
        # Create a blank image
        self.image = np.zeros((200, 200, 3), dtype=np.uint8)
        self.detections = DummyDetections()

    def test_box_annotator(self):
        annotator = BoxAnnotator()
        result = annotator.annotate(self.image.copy(), self.detections)
        self.assertIsNotNone(result)

    def test_round_box_annotator(self):
        annotator = RoundBoxAnnotator(corner_radius=15)
        result = annotator.annotate(self.image.copy(), self.detections)
        self.assertIsNotNone(result)

    def test_oriented_box_annotator(self):
        # Add oriented_boxes attribute
        self.detections.oriented_boxes = np.array([
            [50, 50, 40, 60, 30],
            [130, 130, 50, 70, -15]
        ], dtype=np.float32)
        annotator = OrientedBoxAnnotator()
        result = annotator.annotate(self.image.copy(), self.detections)
        self.assertIsNotNone(result)

    def test_circle_annotator(self):
        annotator = CircleAnnotator(radius=10)
        result = annotator.annotate(self.image.copy(), self.detections)
        self.assertIsNotNone(result)

    def test_ellipse_annotator(self):
        annotator = EllipseAnnotator(axes_ratio=(1, 0.5))
        result = annotator.annotate(self.image.copy(), self.detections)
        self.assertIsNotNone(result)

    def test_triangle_annotator(self):
        annotator = TriangleAnnotator(triangle_type='right')
        result = annotator.annotate(self.image.copy(), self.detections)
        self.assertIsNotNone(result)

    def test_dot_annotator(self):
        annotator = DotAnnotator(radius=5)
        result = annotator.annotate(self.image.copy(), self.detections)
        self.assertIsNotNone(result)

    def test_box_corner_annotator(self):
        annotator = BoxCornerAnnotator(corner_length=10)
        result = annotator.annotate(self.image.copy(), self.detections)
        self.assertIsNotNone(result)

    def test_mask_annotator(self):
        annotator = MaskAnnotator(opacity=0.5, show_edges=True)
        result = annotator.annotate(self.image.copy(), self.detections)
        self.assertIsNotNone(result)

    def test_polygon_annotator(self):
        annotator = PolygonAnnotator(fill=True, gradient_fill=True)
        result = annotator.annotate(self.image.copy(), self.detections)
        self.assertIsNotNone(result)

    def test_halo_annotator(self):
        annotator = HaloAnnotator(halo_radius=20, intensity=0.7)
        result = annotator.annotate(self.image.copy(), self.detections)
        self.assertIsNotNone(result)

    def test_color_annotator(self):
        annotator = ColorAnnotator(fill_opacity=0.3, pattern='stripes')
        result = annotator.annotate(self.image.copy(), self.detections)
        self.assertIsNotNone(result)

    def test_percentage_bar_annotator(self):
        annotator = PercentageBarAnnotator(bar_height=10, show_text=True)
        result = annotator.annotate(self.image.copy(), self.detections)
        self.assertIsNotNone(result)

    def test_trace_annotator(self):
        annotator = TraceAnnotator(max_length=10)
        result = annotator.annotate(self.image.copy(), self.detections)
        self.assertIsNotNone(result)

    def test_heatmap_annotator(self):
        annotator = HeatMapAnnotator(alpha=0.5)
        result = annotator.annotate(self.image.copy(), self.detections)
        self.assertIsNotNone(result)

    def test_background_overlay_annotator(self):
        annotator = BackgroundOverlayAnnotator(alpha=0.3)
        result = annotator.annotate(self.image.copy(), self.detections)
        self.assertIsNotNone(result)

    def test_blur_annotator(self):
        annotator = BlurAnnotator(ksize=(15,15))
        result = annotator.annotate(self.image.copy(), self.detections)
        self.assertIsNotNone(result)

    def test_pixelate_annotator(self):
        annotator = PixelateAnnotator(pixel_size=8)
        result = annotator.annotate(self.image.copy(), self.detections)
        self.assertIsNotNone(result)

    def test_icon_annotator(self):
        icon_img = np.ones((20, 20, 4), dtype=np.uint8) * 255
        annotator = IconAnnotator(icon=icon_img, position=Position.TOP_LEFT)
        result = annotator.annotate(self.image.copy(), self.detections)
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
