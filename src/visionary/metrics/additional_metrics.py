
"""
Additional Metrics for Visionary

Includes Precision, Recall, F1Score, MeanAverageRecall,
confusion matrix generation, and performance visualization tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

class Precision:
    def __init__(self):
        self.tp = 0
        self.fp = 0

    def update(self, true_positives: int, false_positives: int):
        self.tp += true_positives
        self.fp += false_positives

    def compute(self) -> float:
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)

class Recall:
    def __init__(self):
        self.tp = 0
        self.fn = 0

    def update(self, true_positives: int, false_negatives: int):
        self.tp += true_positives
        self.fn += false_negatives

    def compute(self) -> float:
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)

class F1Score:
    def __init__(self, precision: Precision, recall: Recall):
        self.precision = precision
        self.recall = recall

    def compute(self) -> float:
        p = self.precision.compute()
        r = self.recall.compute()
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

class MeanAverageRecall:
    def __init__(self):
        self.recalls = []

    def update(self, recall: float):
        self.recalls.append(recall)

    def compute(self) -> float:
        if not self.recalls:
            return 0.0
        return sum(self.recalls) / len(self.recalls)

class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=int)

    def update(self, true_labels: List[int], pred_labels: List[int]):
        for t, p in zip(true_labels, pred_labels):
            self.matrix[t, p] += 1

    def plot(self, class_names: Optional[List[str]] = None):
        import seaborn as sns
        import matplotlib.pyplot as plt

        labels = class_names if class_names else [str(i) for i in range(self.num_classes)]
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.matrix, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

class PerformanceVisualizer:
    @staticmethod
    def plot_precision_recall_curve(precisions: List[float], recalls: List[float]):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(recalls, precisions, marker='.', label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_f1_vs_threshold(thresholds: List[float], f1_scores: List[float]):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(thresholds, f1_scores, marker='.', label='F1 Score vs Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Detection Threshold')
        plt.legend()
        plt.grid(True)
        plt.show()

class BatchEvaluationFramework:
    def __init__(self, iou_thresholds: List[float] = [0.5], batch_size: int = 10):
        self.iou_thresholds = iou_thresholds
        self.batch_size = batch_size
        self.results = []

    def batch_evaluate(self, model, dataset):
        """
        Batch evaluation of model on dataset.
        Args:
            model: Model with predict method
            dataset: Dataset iterable
        Returns:
            List of evaluation metrics per batch
        """
        batch_results = []
        batch = []
        for idx, item in enumerate(dataset):
            batch.append(item)
            if len(batch) == self.batch_size or idx == len(dataset)-1:
                predictions = model.predict_batch(batch)
                metrics = self.evaluate_batch(predictions, batch)
                batch_results.append(metrics)
                batch.clear()
        self.results.extend(batch_results)
        return batch_results

    def evaluate_batch(self, predictions, groundtruths) -> Dict:
        """
        Evaluate a batch of predictions and groundtruths.
        Args:
            predictions: List of prediction dicts
            groundtruths: Corresponding list of groundtruth dicts
        Returns:
            Evaluation metrics for the batch
        """
        # Placeholder for real evaluation logic
        return {'batch_size': len(predictions), 'metrics': {}}

    def get_aggregated_results(self) -> Dict:
        """
        Aggregate all batch results.
        Returns:
            Aggregated metrics
        """
        # Placeholder for aggregation
        return {}

class TemporalConsistencyMetric:
    def __init__(self):
        self.previous_detections = None
        self.consistency_scores = []

    def update(self, current_detections: List[Dict]):
        """
        Calculate temporal consistency between previous and current detections.
        """
        if self.previous_detections is None:
            self.previous_detections = current_detections
            return

        # Simple IoU matching for temporal consistency
        matches = []
        for cur_det in current_detections:
            max_iou = 0
            for prev_det in self.previous_detections:
                iou = self._iou(cur_det['bbox'], prev_det['bbox'])
                if iou > max_iou:
                    max_iou = iou
            matches.append(max_iou)
        avg_consistency = np.mean(matches) if matches else 0
        self.consistency_scores.append(avg_consistency)
        self.previous_detections = current_detections

    def compute(self) -> float:
        if not self.consistency_scores:
            return 1.0
        return np.mean(self.consistency_scores)

    @staticmethod
    def _iou(box1: List[float], box2: List[float]) -> float:
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

class RealTimePerformanceMonitor:
    def __init__(self):
        import time
        self.timings = []
        self.time = time

    def start(self):
        self.start_time = self.time.time()

    def stop(self):
        duration = self.time.time() - self.start_time
        self.timings.append(duration)

    def average_fps(self) -> float:
        if not self.timings:
            return 0
        avg_time = sum(self.timings) / len(self.timings)
        return 1.0 / avg_time if avg_time > 0 else 0
