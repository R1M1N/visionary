
"""
Dataset Utilities for Visionary

Provides format conversion, dataset splitting, merging with class mapping,
and annotation validation tools.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import shutil
import random
import json
import xml.etree.ElementTree as ET

class DatasetUtils:
    @staticmethod
    def convert_coco_to_yolo(coco_json_path: Path, images_dir: Path, output_dir: Path):
        """
        Convert COCO format dataset to YOLO format.
        Args:
            coco_json_path: Path to COCO JSON annotation file
            images_dir: Directory containing images
            output_dir: Directory to save YOLO annotations
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        img_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        category_to_id = {name: idx for idx, name in enumerate(categories.values())}

        annotations_by_img = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            annotations_by_img.setdefault(img_id, []).append(ann)

        for img_id, anns in annotations_by_img.items():
            filename = img_id_to_filename[img_id]
            img_path = images_dir / filename
            label_path = output_dir / f"{img_path.stem}.txt"

            with open(label_path, 'w') as f:
                for ann in anns:
                    category_name = categories[ann['category_id']]
                    class_id = category_to_id[category_name]
                    bbox = ann['bbox']  # x,y,width,height

                    # COCO bbox to YOLO bbox
                    x_center = (bbox[0] + bbox[2] / 2) / img_path.stat().st_size
                    y_center = (bbox[1] + bbox[3] / 2) / img_path.stat().st_size
                    width = bbox[2] / img_path.stat().st_size
                    height = bbox[3] / img_path.stat().st_size
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}")

    @staticmethod
    def split_dataset(dataset_dir: Path, train_ratio: float=0.7, val_ratio: float=0.15, test_ratio: float=0.15, seed: int=42):
        """
        Split dataset into train, val, and test folders
        Assumes images/ and labels/ folders exist
        """
        random.seed(seed)
        images_dir = dataset_dir / 'images'
        labels_dir = dataset_dir / 'labels'

        image_files = list(images_dir.glob('*'))
        random.shuffle(image_files)

        train_cutoff = int(len(image_files) * train_ratio)
        val_cutoff = train_cutoff + int(len(image_files) * val_ratio)

        sets = {
            'train': image_files[:train_cutoff],
            'val': image_files[train_cutoff:val_cutoff],
            'test': image_files[val_cutoff:]
        }

        for split_name, files in sets.items():
            split_img_dir = dataset_dir / split_name / 'images'
            split_label_dir = dataset_dir / split_name / 'labels'
            split_img_dir.mkdir(parents=True, exist_ok=True)
            split_label_dir.mkdir(parents=True, exist_ok=True)

            for img_file in files:
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.copy(img_file, split_img_dir / img_file.name)
                    shutil.copy(label_file, split_label_dir / label_file.name)

    @staticmethod
    def merge_datasets(source_dirs: List[Path], target_dir: Path, class_map: Optional[Dict[int, int]] = None):
        """
        Merge multiple YOLO datasets with optional class mapping.
        Args:
            source_dirs: List of dataset directories to merge
            target_dir: Directory to save merged dataset
            class_map: Dict mapping source class IDs to target class IDs
        """
        target_img_dir = target_dir / 'images'
        target_label_dir = target_dir / 'labels'
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_label_dir.mkdir(parents=True, exist_ok=True)

        for src_dir in source_dirs:
            src_img_dir = src_dir / 'images'
            src_label_dir = src_dir / 'labels'

            for label_file in src_label_dir.glob('*.txt'):
                src_label_path = label_file
                src_img_path = src_img_dir / f"{label_file.stem}.jpg"

                # Copy image
                if src_img_path.exists():
                    shutil.copy(src_img_path, target_img_dir / src_img_path.name)

                # Read and remap labels
                with open(src_label_path, 'r') as f:
                    lines = f.readlines()

                remapped_lines = []
                for line in lines:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    remapped_class_id = class_map.get(class_id, class_id) if class_map else class_id
                    remapped_line = f"{remapped_class_id} {' '.join(parts[1:])}"
                    remapped_lines.append(remapped_line)

                # Write remapped labels
                target_label_file = target_label_dir / label_file.name
                with open(target_label_file, 'w') as f:
                    f.writelines(remapped_lines)

    @staticmethod
    def validate_annotations(dataset_dir: Path) -> bool:
        """
        Basic validation for YOLO annotations.
        Checks if label file exists for each image and label format correctness.
        """
        images_dir = dataset_dir / 'images'
        labels_dir = dataset_dir / 'labels'

        for img_file in images_dir.glob('*'):
            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                print(f"Missing label file for image: {img_file.name}")
                return False

            with open(label_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"Incorrect label format in {label_file.name}: {line}")
                    return False
                try:
                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:5]))
                except Exception as e:
                    print(f"Invalid number in label file {label_file.name}: {line}")
                    return False

        print("Annotations validation passed.")
        return True
