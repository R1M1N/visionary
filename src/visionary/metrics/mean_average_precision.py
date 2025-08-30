"""
Mean Average Precision (mAP) Evaluation Framework for Visionary

Provides COCO-style mAP with multiple IoU thresholds,
class-specific and class-agnostic evaluation modes,
and size-based performance analysis.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

class MeanAveragePrecision:
    def __init__(self, iou_thresholds: List[float] = None, class_agnostic: bool = False):
        """
        Initialize mAP evaluator.
        Args:
            iou_thresholds: List of IoU thresholds to evaluate at
            class_agnostic: Whether to use class-agnostic evaluation
        """
        self.iou_thresholds = iou_thresholds or np.linspace(0.5, 0.95, 10).tolist()
        self.class_agnostic = class_agnostic
        self.reset()

    def reset(self):
        self.detections = []  # List of dicts {'image_id', 'bbox', 'score', 'category_id'}
        self.groundtruths = []  # List of dicts {'image_id', 'bbox', 'category_id'}

    def add_detections(self, detections: List[Dict]):
        self.detections.extend(detections)

    def add_groundtruths(self, groundtruths: List[Dict]):
        self.groundtruths.extend(groundtruths)

    @staticmethod
    def _compute_iou(box1: List[float], box2: List[float]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area
        if union_area == 0:
            return 0
        return inter_area / union_area

    def evaluate(self) -> Dict[str, float]:
        """
        Compute mAP for all IoU thresholds and classes.
        Returns:
            Dict with overall mAP and per-class AP
        """
        if not self.detections or not self.groundtruths:
            return {"mAP": 0.0, "AP_per_class": {}}

        # Prepare data by class or agnostic
        if self.class_agnostic:
            all_classes = [None]
        else:
            all_classes = list({gt['category_id'] for gt in self.groundtruths})

        AP_per_class = {}

        for cls in all_classes:
            gt_for_class = [gt for gt in self.groundtruths if (cls is None or gt['category_id'] == cls)]
            det_for_class = [det for det in self.detections if (cls is None or det['category_id'] == cls)]

            AP_per_iou = []
            for iou_thresh in self.iou_thresholds:
                ap = self._average_precision(gt_for_class, det_for_class, iou_thresh)
                AP_per_iou.append(ap)

            AP_per_class[cls if cls is not None else 'all'] = np.mean(AP_per_iou)

        mAP = np.mean(list(AP_per_class.values())) if AP_per_class else 0.0

        return {"mAP": mAP, "AP_per_class": AP_per_class}

    def _average_precision(self, gts: List[Dict], dets: List[Dict], iou_threshold: float) -> float:
        """
        Calculate average precision for one class and IoU threshold.
        """
        if not gts:
            return 0.0
        if not dets:
            return 0.0

        gts_matched = set()
        dets_sorted = sorted(dets, key=lambda x: x['score'], reverse=True)

        tp = []
        fp = []
        for det in dets_sorted:
            ious = [self._compute_iou(det['bbox'], gt['bbox']) for gt in gts]
            max_iou = 0.0
            max_iou_idx = -1
            for idx, iou in enumerate(ious):
                if iou > max_iou and idx not in gts_matched:
                    max_iou = iou
                    max_iou_idx = idx

            if max_iou >= iou_threshold:
                tp.append(1)
                fp.append(0)
                gts_matched.add(max_iou_idx)
            else:
                tp.append(0)
                fp.append(1)

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recalls = tp_cum / len(gts) if len(gts) > 0 else np.zeros_like(tp_cum)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-16)

        # Calculate AP using numerical integration
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))

        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
        ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
        return ap

    def size_based_analysis(self, size_ranges: List[Tuple[int, int]]) -> Dict[str, Dict[str, float]]:
        """
        Perform mAP analysis for different size ranges.
        Args:
            size_ranges: List of (min_area, max_area) tuples
        Returns:
            Dict mapping size range str to mAP and AP per class
        """
        results = {}
        for min_area, max_area in size_ranges:
            filtered_gts = [gt for gt in self.groundtruths if self._bbox_area(gt['bbox']) >= min_area and self._bbox_area(gt['bbox']) < max_area]
            filtered_dets = [det for det in self.detections if self._bbox_area(det['bbox']) >= min_area and self._bbox_area(det['bbox']) < max_area]

            if not filtered_gts:
                results[f"{min_area}-{max_area}"] = {"mAP": 0.0, "AP_per_class": {}}
                continue

            self.groundtruths = filtered_gts
            self.detections = filtered_dets

            eval_result = self.evaluate()
            results[f"{min_area}-{max_area}"] = eval_result

        return results

    @staticmethod
    def _bbox_area(bbox: List[float]) -> float:
        return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
