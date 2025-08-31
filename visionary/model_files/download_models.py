# download_models.py
from ultralytics import YOLO
import os

# Create model_files directory if it doesn't exist
os.makedirs('model_files', exist_ok=True)

# Essential models list
essential_models = [
    # Detection models
    'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
    
    # Segmentation models
    'yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt', 
    'yolov8l-seg.pt', 'yolov8x-seg.pt',
    
    # Classification models
    'yolov8n-cls.pt', 'yolov8s-cls.pt', 'yolov8m-cls.pt',
    'yolov8l-cls.pt', 'yolov8x-cls.pt',
    
    # YOLOv11 models
    'yolo11n.pt', 'yolo11m.pt', 'yolo11n-seg.pt', 'yolo11m-seg.pt'
]

print("Downloading essential YOLOv8/YOLOv11 models...")

for model_name in essential_models:
    try:
        print(f"Downloading {model_name}...")
        model = YOLO(model_name)
        print(f"✓ {model_name} downloaded successfully")
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")

print("Download complete!")
