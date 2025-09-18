
"""
Classification Core System for Visionary

Provides Classifications class, classification annotators,
and classification evaluation metrics.
"""

from typing import List, Dict, Optional
import numpy as np
import cv2

class Classification:
    def __init__(self, class_id: int, class_name: str, confidence: float):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence

class Classifications:
    def __init__(self):
        self.classifications: List[Classification] = []

    def add(self, class_id: int, class_name: str, confidence: float):
        self.classifications.append(Classification(class_id, class_name, confidence))

    def top_k(self, k: int = 5) -> List[Classification]:
        return sorted(self.classifications, key=lambda c: c.confidence, reverse=True)[:k]

class ClassificationAnnotator:
    def __init__(self, font_scale: float = 1.0, color: tuple = (0, 255, 0)):
        self.font_scale = font_scale
        self.color = color

    def annotate(self, image: np.ndarray, classifications: Classifications, org: tuple = (10, 30)) -> np.ndarray:
        y = org[1]
        for c in classifications.top_k():
            text = f"{c.class_name}: {c.confidence:.2f}"
            cv2.putText(image, text, (org[0], y), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.color, 2)
            y += int(30 * self.font_scale)
        return image

class ClassificationMetrics:
    def __init__(self):
        self.true_labels: List[int] = []
        self.pred_labels: List[int] = []

    def update(self, true_label: int, pred_label: int):
        self.true_labels.append(true_label)
        self.pred_labels.append(pred_label)

    def accuracy(self) -> float:
        if not self.true_labels:
            return 0.0
        correct = sum(t == p for t, p in zip(self.true_labels, self.pred_labels))
        return correct / len(self.true_labels)

    def precision(self) -> float:
        from sklearn.metrics import precision_score
        return precision_score(self.true_labels, self.pred_labels, average='weighted', zero_division=0)

    def recall(self) -> float:
        from sklearn.metrics import recall_score
        return recall_score(self.true_labels, self.pred_labels, average='weighted', zero_division=0)

    def f1_score(self) -> float:
        from sklearn.metrics import f1_score
        return f1_score(self.true_labels, self.pred_labels, average='weighted', zero_division=0)
