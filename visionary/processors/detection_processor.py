# visionary/processors/detection_processor.py
from ultralytics import YOLO
import numpy as np
from pathlib import Path
from .base import BaseProcessor

class DetectionProcessor(BaseProcessor):
    
    def _load_model(self):
        """Load the actual YOLO detection model."""
        model_type = getattr(self.model_config, 'model_type', 'yolov8m')
        print(f"Loading detection model: {model_type}")
        
        # Load the actual YOLO model
        model_path = f"{model_type.value}.pt"
        self.model = YOLO(model_path)
    
    def process(self, input_data, task_config):
        """Process input with real YOLO detection."""
        try:
            # Run YOLO inference on the input
            results = self.model(input_data)
            
            predictions = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    scores = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(np.int32)
                    
                    # Convert to structured format
                    for box, score, cls_id in zip(boxes, scores, classes):
                        predictions.append({
                            'bbox': box.tolist(),
                            'confidence': float(score),
                            'class': self.model.names[cls_id]
                        })
            
            return {
                'predictions': predictions,
                'total_detections': len(predictions),
                'image_size': getattr(results[0], 'orig_shape', (640, 480))
            }
            
        except Exception as e:
            print(f"Detection error: {e}")
            return {
                'predictions': [],
                'error': str(e),
                'total_detections': 0
            }
