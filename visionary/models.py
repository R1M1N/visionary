"""
Model configurations and types
"""
import os
from enum import Enum
from typing import Optional, Dict, Any


class ModelType(Enum):
    """Supported model types across different frameworks."""
    # YOLO models
    YOLOV8_NANO = "yolov8n"
    YOLOV8_SMALL = "yolov8s"
    YOLOV8_MEDIUM = "yolov8m"
    YOLOV8_LARGE = "yolov8l"
    YOLOV8_XLARGE = "yolov8x"
    
    # YOLO11 models  
    YOLO11_NANO = "yolo11n"
    YOLO11_MEDIUM = "yolo11m"
    
    # Segmentation models
    YOLOV8_SEG_MEDIUM = "yolov8m-seg"
    YOLO11_SEG_MEDIUM = "yolov11m-seg"
    
    # RT-DETR models (fixed the typo from your original)
    RTDETR_R18VD = "rtdetr_r18vd"
    RTDETR_R50VD = "rtdetrv2_r50vd"
    
    # SAM models
    SAM_VIT_H = "sam_vit_h"
    SAM2_HIERA_LARGE = "sam2_hiera_large"
    
    # Transformers models
    DETR = "detr"
    DINO = "dino"
    
    # Custom models
    CUSTOM = "custom"

    # Classification models
    YOLOV8_CLS_MEDIUM = "yolov8m-cls"
    YOLOV8_NANO_CLS = "yolov8n-cls"
    YOLOV8_SMALL_CLS = "yolov8s-cls"
    YOLOV8_LARGE_CLS = "yolov8l-cls"
    YOLOV8_XLARGE_CLS = "yolov8x-cls"
    
    YOLOV8_NANO_SEG = "yolov8n-seg"
    YOLOV8_SMALL_SEG = "yolov8s-seg"
    YOLOV8_LARGE_SEG = "yolov8l-seg"
    YOLOV8_XLARGE_SEG = "yolov8x-seg"
    
    YOLOV8_NANO_POSE = "yolov8n-pose"
    YOLOV8_SMALL_POSE = "yolov8s-pose"
    YOLOV8_MEDIUM_POSE = "yolov8m-pose"


# Model file mapping based on your actual model files
# Complete MODEL_FILE_MAPPING in models.py
# Complete MODEL_FILE_MAPPING in models.py
# Fix your MODEL_FILE_MAPPING - remove duplicates
MODEL_FILE_MAPPING = {
    # Detection models
    ModelType.YOLOV8_NANO: "yolov8n.pt",
    ModelType.YOLOV8_SMALL: "yolov8s.pt", 
    ModelType.YOLOV8_MEDIUM: "yolov8m.pt",
    ModelType.YOLOV8_LARGE: "yolov8l.pt",
    ModelType.YOLOV8_XLARGE: "yolov8x.pt",
    
    # YOLOv11 models
    ModelType.YOLO11_NANO: "yolo11n.pt",
    ModelType.YOLO11_MEDIUM: "yolo11m.pt",
    
    # Segmentation models  
    ModelType.YOLOV8_SEG_MEDIUM: "yolov8m-seg.pt",
    ModelType.YOLO11_SEG_MEDIUM: "yolo11m-seg.pt",
    
    # Classification models
    ModelType.YOLOV8_CLS_MEDIUM: "yolov8m-cls.pt",
    ModelType.YOLOV8_NANO_CLS: "yolov8n-cls.pt",
    ModelType.YOLOV8_SMALL_CLS: "yolov8s-cls.pt",
    ModelType.YOLOV8_LARGE_CLS: "yolov8l-cls.pt",
    ModelType.YOLOV8_XLARGE_CLS: "yolov8x-cls.pt",
    
    # Additional segmentation models
    ModelType.YOLOV8_NANO_SEG: "yolov8n-seg.pt",
    ModelType.YOLOV8_SMALL_SEG: "yolov8s-seg.pt",
    ModelType.YOLOV8_LARGE_SEG: "yolov8l-seg.pt",
    ModelType.YOLOV8_XLARGE_SEG: "yolov8x-seg.pt",
    
    # Pose models
    ModelType.YOLOV8_NANO_POSE: "yolov8n-pose.pt",
    ModelType.YOLOV8_SMALL_POSE: "yolov8s-pose.pt",
    ModelType.YOLOV8_MEDIUM_POSE: "yolov8m-pose.pt",
    
    # RT-DETR models (matching your actual files)
    ModelType.RTDETR_R18VD: "rtdetr_r18vd_5x_coco_objects365_from_paddle.pth",
    ModelType.RTDETR_R50VD: "rtdetrv2_r50vd_m_7x_coco_ema.pth",
    
    # SAM models  
    ModelType.SAM_VIT_H: "sam_vit_h_4b8939.pth",
    ModelType.SAM2_HIERA_LARGE: "sam2_hiera_large.pt",
}





def get_model_path(model_type: ModelType, model_files_dir: str = "models") -> str:
    """
    Get the full path to a model file.
    
    Args:
        model_type: The model type enum
        model_files_dir: Directory containing model files
        
    Returns:
        Full path to the model file
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If no file mapping exists for model type
    """
    if model_type not in MODEL_FILE_MAPPING:
        raise ValueError(f"No file mapping found for model type: {model_type}")
    
    filename = MODEL_FILE_MAPPING[model_type]
    model_path = os.path.join(model_files_dir, filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return model_path


def validate_available_models(model_files_dir: str = "models") -> Dict[str, bool]:
    """
    Validate which models are available.
    
    Args:
        model_files_dir: Directory containing model files
        
    Returns:
        Dictionary mapping model types to availability
    """
    availability = {}
    
    for model_type, filename in MODEL_FILE_MAPPING.items():
        model_path = os.path.join(model_files_dir, filename)
        availability[model_type.value] = os.path.exists(model_path)
    
    return availability


def get_available_models_for_task(task_models: Dict) -> Dict:
    """
    Filter available models for specific tasks.
    
    Args:
        task_models: Dict mapping tasks to model lists
        
    Returns:
        Dict with only available models for each task
    """
    available = validate_available_models()
    available_task_models = {}
    
    for task, models in task_models.items():
        available_models = [
            model for model in models 
            if model in MODEL_FILE_MAPPING and available.get(model.value, False)
        ]
        available_task_models[task] = available_models
    
    return available_task_models


class ModelConfig:
    """Configuration for model loading and inference."""
    
    def __init__(self,
                 model_type: ModelType,
                 device: str = "auto",
                 confidence: float = 0.5,
                 iou_threshold: float = 0.45,
                 model_path: Optional[str] = None,
                 **kwargs):
        """
        Initialize model configuration.
        
        Args:
            model_type: Type of model to load
            device: Device for inference ("cpu", "cuda", "auto")
            confidence: Confidence threshold for predictions
            iou_threshold: IoU threshold for NMS
            model_path: Custom path to model file (overrides default)
            **kwargs: Additional model-specific parameters
        """
        self.model_type = model_type
        self.device = device
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        
        # Use provided path or get default path
        if model_path:
            self.model_path = model_path
        else:
            try:
                self.model_path = get_model_path(model_type)
            except (ValueError, FileNotFoundError) as e:
                print(f"Warning: {e}")
                self.model_path = None
        
        self.extra_params = kwargs
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model_type': self.model_type.value,
            'device': self.device,
            'confidence': self.confidence,
            'iou_threshold': self.iou_threshold,
            'model_path': self.model_path,
            **self.extra_params
        }
    
    def is_available(self) -> bool:
        """Check if the model file is available."""
        return self.model_path is not None and os.path.exists(self.model_path)
