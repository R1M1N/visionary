
"""
Dataset Core Classes for Visionary

Provides DetectionDataset with lazy loading,
auto format detection, and support for COCO, YOLO, Pascal VOC.
"""

import os
from pathlib import Path
from typing import List, Optional, Union, Dict
import json
import xml.etree.ElementTree as ET
import numpy as np

class DetectionDataset:
    def __init__(self, dataset_dir: Union[str, Path], format: Optional[str] = None):
        """
        Initialize the dataset loader.
        Args:
            dataset_dir: Path to dataset root directory
            format: Optional dataset format override ("coco", "yolo", "pascal_voc")
        """
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        self.format = format or self._detect_format()
        self.images = []  # List of image file paths
        self.annotations = []  # Corresponding annotation objects

        self._load_dataset()

    def _detect_format(self) -> str:
        """Detect dataset format based on files in directory."""
        files = list(self.dataset_dir.rglob('*'))
        filenames = [f.name.lower() for f in files]

        # COCO
        if any(f.endswith('.json') for f in filenames):
            return 'coco'
        # YOLO
        if any(f.endswith('.txt') for f in filenames) and any(f.endswith(('.jpg','.png','.jpeg')) for f in filenames):
            return 'yolo'
        # Pascal VOC
        if any(f.endswith('.xml') for f in filenames):
            return 'pascal_voc'

        raise RuntimeError("Unable to determine dataset format")

    def _load_dataset(self):
        if self.format == 'coco':
            self._load_coco()
        elif self.format == 'yolo':
            self._load_yolo()
        elif self.format == 'pascal_voc':
            self._load_pascal_voc()
        else:
            raise RuntimeError(f"Unsupported dataset format: {self.format}")

    def _load_coco(self):
        json_file = next(self.dataset_dir.glob('*.json'), None)
        if json_file is None:
            raise FileNotFoundError("COCO JSON annotation file not found")

        with open(json_file, 'r') as f:
            data = json.load(f)

        self.images = [img['file_name'] for img in data.get('images', [])]
        self.annotations = data.get('annotations', [])

    def _load_yolo(self):
        # YOLO assumed directory structure:
        # images/ and labels/ directories
        images_dir = self.dataset_dir / 'images'
        labels_dir = self.dataset_dir / 'labels'
        if not images_dir.exists() or not labels_dir.exists():
            raise FileNotFoundError("YOLO dataset missing 'images' or 'labels' directories")

        self.images = sorted(images_dir.glob('*'))
        self.annotations = sorted(labels_dir.glob('*.txt'))

    def _load_pascal_voc(self):
        # Pascal VOC assumed directory structure:
        # JPEGImages/ and Annotations/ directories
        images_dir = self.dataset_dir / 'JPEGImages'
        annotations_dir = self.dataset_dir / 'Annotations'
        if not images_dir.exists() or not annotations_dir.exists():
            raise FileNotFoundError("Pascal VOC dataset missing 'JPEGImages' or 'Annotations' directories")

        self.images = sorted(images_dir.glob('*'))
        self.annotations = sorted(annotations_dir.glob('*.xml'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        """
        Lazy loading of image and annotations at index.
        Returns:
            image_path: Path to image file
            annotations: Annotation data in format depending on dataset
        """
        image_path = self.images[idx]
        annotations = None

        if self.format == 'coco':
            image_id = idx + 1  # Assuming 1-based ids
            annotations = [ann for ann in self.annotations if ann['image_id'] == image_id]

        elif self.format == 'yolo':
            ann_path = self.annotations[idx]
            with open(ann_path, 'r') as f:
                annotations = f.readlines()

        elif self.format == 'pascal_voc':
            ann_path = self.annotations[idx]
            tree = ET.parse(ann_path)
            root = tree.getroot()
            annotations = root

        return image_path, annotations
