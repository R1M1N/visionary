"""
Comprehensive test suite for Visionary Detections system

This tests the complete Detections functionality including:
- Core detection operations
- Model integration adapters
- Bounding box utilities
- Advanced features and edge cases
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from visionary.detection.core import Detections, box_iou_batch, merge_detections
from visionary.detection.utils.boxes import (
    xyxy_to_xywh, xywh_to_xyxy, xyxy_to_cxcywh, cxcywh_to_xyxy,
    clip_boxes, scale_boxes, expand_boxes, normalize_boxes, mask_to_xyxy
)


class ComprehensiveDetectionsTest:
    """Complete test suite for Detections system"""
    
    def __init__(self):
        self.test_count = 0
        self.passed_count = 0
        self.failed_tests = []
    
    def run_test(self, test_name, test_func):
        """Run individual test with error handling"""
        self.test_count += 1
        try:
            test_func()
            print(f"✅ {test_name}")
            self.passed_count += 1
        except Exception as e:
            print(f"❌ {test_name}: {e}")
            self.failed_tests.append((test_name, str(e)))
    
    def test_basic_creation_and_properties(self):
        """Test basic Detections creation and properties"""
        # Standard creation
        boxes = np.array([[0, 0, 100, 100], [50, 50, 150, 150]], dtype=np.float32)
        conf = np.array([0.9, 0.8])
        classes = np.array([0, 1])
        
        det = Detections(xyxy=boxes, confidence=conf, class_id=classes)
        
        assert len(det) == 2, "Length mismatch"
        assert det.xyxy.shape == (2, 4), "Box shape mismatch"
        assert np.array_equal(det.confidence, conf), "Confidence mismatch"
        assert np.array_equal(det.class_id, classes), "Class ID mismatch"
        
        # Test string representations
        repr_str = repr(det)
        str_repr = str(det)
        assert "Detections(n=2)" in repr_str, "Repr format wrong"
        assert "2 boxes" in str_repr, "String format wrong"
    
    def test_confidence_filtering(self):
        """Test confidence threshold filtering"""
        boxes = np.array([[0, 0, 100, 100], [50, 50, 150, 150], [100, 100, 200, 200]], dtype=np.float32)
        conf = np.array([0.9, 0.5, 0.7])
        
        det = Detections(xyxy=boxes, confidence=conf)
        
        # Filter with threshold 0.6
        filtered = det.with_threshold(0.6)
        assert len(filtered) == 2, "Threshold filtering failed"
        assert np.min(filtered.confidence) >= 0.6, "Threshold not applied correctly"
        
        # Filter with high threshold
        high_filtered = det.with_threshold(0.95)
        assert len(high_filtered) == 0, "High threshold should filter all"
    
    def test_nms_functionality(self):
        """Test Non-Maximum Suppression"""
        # Overlapping boxes with different confidences
        boxes = np.array([
            [0, 0, 100, 100],    # High confidence
            [10, 10, 110, 110],  # Medium confidence, overlaps with first
            [200, 200, 300, 300] # Low confidence, no overlap
        ], dtype=np.float32)
        
        conf = np.array([0.9, 0.7, 0.5])
        
        det = Detections(xyxy=boxes, confidence=conf)
        
        # Apply NMS with low IoU threshold
        nms_result = det.with_nms(iou_threshold=0.3)
        assert len(nms_result) <= len(det), "NMS should not increase detections"
        
        # Apply NMS with high IoU threshold
        nms_high = det.with_nms(iou_threshold=0.9)
        assert len(nms_high) == len(det), "High IoU threshold should keep all"
    
    def test_area_calculations(self):
        """Test area calculation methods"""
        boxes = np.array([
            [0, 0, 10, 10],    # Area = 100
            [0, 0, 20, 30],    # Area = 600
            [10, 10, 15, 15]   # Area = 25
        ], dtype=np.float32)
        
        det = Detections(xyxy=boxes)
        
        # Test batch area calculation
        areas = det.area()
        expected_areas = np.array([100, 600, 25])
        np.testing.assert_array_equal(areas, expected_areas, "Batch area calculation failed")
        
        # Test single area calculation
        single_area = det.area(index=1)
        assert single_area == 600, "Single area calculation failed"
    
    def test_anchor_strategies(self):
        """Test different anchor point strategies"""
        boxes = np.array([[0, 0, 100, 100]], dtype=np.float32)
        det = Detections(xyxy=boxes)
        
        # Test center anchors
        center_anchors = det.get_anchors("center")
        expected_center = np.array([[50, 50]])
        np.testing.assert_array_equal(center_anchors, expected_center, "Center anchors wrong")
        
        # Test bottom center
        bottom_anchors = det.get_anchors("bottom_center")
        expected_bottom = np.array([[50, 100]])
        np.testing.assert_array_equal(bottom_anchors, expected_bottom, "Bottom anchors wrong")
        
        # Test top left
        top_left_anchors = det.get_anchors("top_left")
        expected_top_left = np.array([[0, 0]])
        np.testing.assert_array_equal(top_left_anchors, expected_top_left, "Top-left anchors wrong")
    
    def test_image_cropping(self):
        """Test image cropping functionality"""
        # Create test image
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        det = Detections(xyxy=boxes)
        
        # Test basic cropping
        cropped = det.crop_image(image, 0)
        assert cropped.shape == (40, 40, 3), "Cropped dimensions wrong"
        
        # Test cropping with padding
        cropped_padded = det.crop_image(image, 0, pad=5)
        assert cropped_padded.shape == (50, 50, 3), "Padded cropping wrong"
        
        # Test out of bounds handling
        try:
            det.crop_image(image, 5)  # Index out of bounds
            assert False, "Should raise IndexError"
        except IndexError:
            pass  # Expected
    
    def test_serialization(self):
        """Test dictionary serialization and deserialization"""
        boxes = np.array([[0, 0, 100, 100], [50, 50, 150, 150]], dtype=np.float32)
        conf = np.array([0.9, 0.8])
        classes = np.array([0, 1])
        tracker_ids = np.array([10, 20])
        
        original = Detections(
            xyxy=boxes, 
            confidence=conf, 
            class_id=classes,
            tracker_id=tracker_ids
        )
        
        # Serialize to dict
        data_dict = original.to_dict()
        assert 'xyxy' in data_dict, "Missing xyxy in serialization"
        assert 'confidence' in data_dict, "Missing confidence in serialization"
        
        # Deserialize from dict
        reconstructed = Detections.from_dict(data_dict)
        assert len(reconstructed) == len(original), "Deserialization length mismatch"
        np.testing.assert_array_equal(reconstructed.xyxy, original.xyxy, "Deserialized boxes wrong")
        np.testing.assert_array_equal(reconstructed.confidence, original.confidence, "Deserialized conf wrong")
    
    def test_indexing_and_slicing(self):
        """Test array-like indexing and slicing operations"""
        boxes = np.array([
            [0, 0, 100, 100],
            [50, 50, 150, 150],
            [100, 100, 200, 200],
            [150, 150, 250, 250]
        ], dtype=np.float32)
        conf = np.array([0.9, 0.8, 0.7, 0.6])
        classes = np.array([0, 1, 2, 3])
        
        det = Detections(xyxy=boxes, confidence=conf, class_id=classes)
        
        # Test single indexing
        single = det[1]
        assert len(single) == 1, "Single indexing failed"
        assert single.confidence[0] == 0.8, "Wrong detection indexed"
        
        # Test negative indexing
        last = det[-1]
        assert last.confidence[0] == 0.6, "Negative indexing failed"
        
        # Test slicing
        slice_result = det[1:3]
        assert len(slice_result) == 2, "Slicing failed"
        np.testing.assert_array_equal(slice_result.confidence, conf[1:3], "Sliced conf wrong")
        
        # Test out of bounds
        try:
            det[10]
            assert False, "Should raise IndexError"
        except IndexError:
            pass  # Expected
    
    def test_bounding_box_utilities(self):
        """Test bounding box utility functions"""
        # Test coordinate conversions
        xyxy_boxes = np.array([[10, 20, 30, 40]], dtype=np.float32)
        
        # xyxy to xywh conversion
        xywh_boxes = xyxy_to_xywh(xyxy_boxes)
        expected_xywh = np.array([[10, 20, 20, 20]])
        np.testing.assert_array_equal(xywh_boxes, expected_xywh, "xyxy to xywh conversion wrong")
        
        # xywh back to xyxy
        xyxy_back = xywh_to_xyxy(xywh_boxes)
        np.testing.assert_array_equal(xyxy_back, xyxy_boxes, "xywh to xyxy conversion wrong")
        
        # Test center format conversion
        cxcywh_boxes = xyxy_to_cxcywh(xyxy_boxes)
        expected_cxcywh = np.array([[20, 30, 20, 20]])  # center_x, center_y, width, height
        np.testing.assert_array_equal(cxcywh_boxes, expected_cxcywh, "xyxy to cxcywh conversion wrong")
        
        # Test mask to bbox
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:30, 15:35] = True
        bbox = mask_to_xyxy(mask)
        expected_bbox = np.array([15, 10, 34, 29])
        np.testing.assert_array_equal(bbox, expected_bbox, "Mask to bbox conversion wrong")
    
    def test_advanced_box_operations(self):
        """Test advanced box manipulation operations"""
        boxes = np.array([[10, 10, 90, 90]], dtype=np.float32)
        
        # Test box clipping
        img_shape = (100, 100)
        clipped = clip_boxes(boxes, img_shape)
        np.testing.assert_array_equal(clipped, boxes, "Clipping changed valid boxes")
        
        # Test box scaling - FIXED VERSION
        scaled = scale_boxes(boxes, 2.0)
        expected_scaled = np.array([[20, 20, 180, 180]], dtype=np.float32)  # Ensure float32
        np.testing.assert_allclose(scaled, expected_scaled, rtol=1e-5, err_msg="Box scaling wrong")
        
        # Test box expansion
        expanded = expand_boxes(boxes, 0.1)  # 10% expansion
        assert expanded[0, 2] > boxes[0, 2], "Box not expanded"
        assert expanded[0, 3] > boxes[0, 3], "Box not expanded"
        
        # Test normalization - FIXED VERSION
        normalized = normalize_boxes(boxes, (100, 100))
        expected_norm = np.array([[0.1, 0.1, 0.9, 0.9]], dtype=np.float32)  # Ensure float32
        np.testing.assert_allclose(normalized, expected_norm, rtol=1e-5, err_msg="Normalization wrong")

    
    def test_batch_iou_calculation(self):
        """Test batch IoU calculation"""
        boxes1 = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        boxes2 = np.array([[5, 5, 15, 15], [25, 25, 35, 35]], dtype=np.float32)
        
        iou_matrix = box_iou_batch(boxes1, boxes2)
        assert iou_matrix.shape == (2, 2), "IoU matrix shape wrong"
        
        # First box should have some overlap with first target
        assert iou_matrix[0, 0] > 0, "Expected overlap not detected"
        
        # Non-overlapping boxes should have IoU = 0
        assert iou_matrix[0, 1] == 0, "Non-overlapping boxes have IoU > 0"
    
    def test_detection_merging(self):
        """Test merging multiple Detection objects"""
        det1 = Detections(xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32))
        det2 = Detections(xyxy=np.array([[20, 20, 30, 30]], dtype=np.float32))
        
        merged = merge_detections([det1, det2])
        assert len(merged) == 2, "Merge failed"
        
        # Test merging with empty list
        empty_merged = merge_detections([])
        assert len(empty_merged) == 0, "Empty merge should return empty detections"
    
    def test_model_integration_formats(self):
        """Test model integration with different formats"""
        # Test Transformers format
        transformers_data = [
            {"bbox": [0, 0, 100, 100], "score": 0.9, "label": 0},
            {"bbox": [50, 50, 150, 150], "score": 0.8, "label": 1}
        ]
        
        det_transformers = Detections.from_transformers(transformers_data)
        assert len(det_transformers) == 2, "Transformers integration failed"
        
        # Test MMDetection format
        mmdet_bbox = np.array([[0, 0, 100, 100, 0.9], [50, 50, 150, 150, 0.8]])
        mmdet_labels = np.array([0, 1])
        
        det_mmdet = Detections.from_mmdet(mmdet_bbox, mmdet_labels)
        assert len(det_mmdet) == 2, "MMDet integration failed"
        
        # Test empty results
        empty_det = Detections.from_transformers([])
        assert len(empty_det) == 0, "Empty transformers result should return empty detections"
    
    def test_error_handling(self):
        """Test error handling and edge cases"""
        boxes = np.array([[0, 0, 100, 100]], dtype=np.float32)
        det = Detections(xyxy=boxes)
        
        # Test invalid anchor strategy
        try:
            det.get_anchors("invalid_strategy")
            assert False, "Should raise ValueError"
        except ValueError:
            pass  # Expected
        
        # Test dimension mismatch
        try:
            Detections(
                xyxy=np.array([[0, 0, 100, 100]]),
                confidence=np.array([0.9, 0.8])  # Wrong length
            )
            assert False, "Should raise ValueError for dimension mismatch"
        except ValueError:
            pass  # Expected
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Starting Comprehensive Detections Test Suite")
        print("=" * 60)
        
        # Define all test methods
        tests = [
            ("Basic Creation and Properties", self.test_basic_creation_and_properties),
            ("Confidence Filtering", self.test_confidence_filtering),
            ("Non-Maximum Suppression", self.test_nms_functionality),
            ("Area Calculations", self.test_area_calculations),
            ("Anchor Strategies", self.test_anchor_strategies),
            ("Image Cropping", self.test_image_cropping),
            ("Serialization/Deserialization", self.test_serialization),
            ("Indexing and Slicing", self.test_indexing_and_slicing),
            ("Bounding Box Utilities", self.test_bounding_box_utilities),
            ("Advanced Box Operations", self.test_advanced_box_operations),
            ("Batch IoU Calculation", self.test_batch_iou_calculation),
            ("Detection Merging", self.test_detection_merging),
            ("Model Integration Formats", self.test_model_integration_formats),
            ("Error Handling", self.test_error_handling)
        ]
        
        # Run all tests
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Print results
        print("\n" + "=" * 60)
        print(f"🎯 Test Results: {self.passed_count}/{self.test_count} passed")
        
        if self.failed_tests:
            print(f"\n❌ Failed Tests ({len(self.failed_tests)}):")
            for name, error in self.failed_tests:
                print(f"   • {name}: {error}")
        else:
            print("\n🏆 ALL TESTS PASSED! Your Detections system is fully functional!")
        
        return len(self.failed_tests) == 0


if __name__ == "__main__":
    tester = ComprehensiveDetectionsTest()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
