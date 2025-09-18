"""
Factory for creating appropriate processors based on task and model types
"""
from typing import Dict, List
from visionary.models import ModelType, ModelConfig
from visionary.model_files.task_types import TaskType
from visionary.processors.base import BaseProcessor
from visionary.processors.detection_processor import DetectionProcessor
from visionary.processors.segmentation_processor import SegmentationProcessor
from visionary.processors.keypoint_processor import KeypointProcessor
from visionary.processors.classification_processor import ClassificationProcessor
from visionary.processors.tracking_processor import TrackingProcessor
from visionary.processors.exceptions import VisionaryError
from visionary.utils.input_handler import InputType


class ProcessorFactory:
    """Factory class for creating task-specific processors."""
    
    def __init__(self):
        """Initialize the processor factory."""
        self._processors: Dict[TaskType, type] = {
            TaskType.DETECTION: DetectionProcessor,
            TaskType.SEGMENTATION: SegmentationProcessor,
            TaskType.INSTANCE_SEGMENTATION: SegmentationProcessor,
            TaskType.KEYPOINT_DETECTION: KeypointProcessor,
            TaskType.CLASSIFICATION: ClassificationProcessor,
            TaskType.TRACKING: TrackingProcessor,
            TaskType.VIDEO_TRACKING: TrackingProcessor,
            TaskType.VIDEO_SEGMENTATION: SegmentationProcessor,
        }
        
        self._task_models: Dict[TaskType, List[ModelType]] = {
            TaskType.DETECTION: [
                ModelType.YOLOV8_NANO, ModelType.YOLOV8_SMALL, 
                ModelType.YOLOV8_MEDIUM, ModelType.YOLOV8_LARGE,
                ModelType.YOLO11_NANO, ModelType.YOLO11_MEDIUM,
                ModelType.RTDETR_R18VD, ModelType.RTDETR_R50VD
            ],
            TaskType.SEGMENTATION: [
                ModelType.YOLOV8_SEG_MEDIUM, ModelType.YOLO11_SEG_MEDIUM,
                ModelType.SAM_VIT_H, ModelType.SAM2_HIERA_LARGE
            ],
            TaskType.INSTANCE_SEGMENTATION: [
                ModelType.YOLOV8_SEG_MEDIUM, ModelType.YOLO11_SEG_MEDIUM
            ],
            TaskType.KEYPOINT_DETECTION: [
                ModelType.YOLOV8_MEDIUM, ModelType.YOLO11_MEDIUM
            ],
            TaskType.CLASSIFICATION: [
                ModelType.YOLOV8_CLS_MEDIUM, ModelType.YOLOV8_NANO_CLS, 
                ModelType.YOLOV8_SMALL_CLS, ModelType.YOLOV8_LARGE_CLS, 
                ModelType.YOLOV8_XLARGE_CLS
            ],
            TaskType.TRACKING: [
                ModelType.YOLOV8_MEDIUM, ModelType.YOLO11_MEDIUM
            ],
            TaskType.VIDEO_TRACKING: [
                ModelType.YOLOV8_MEDIUM, ModelType.YOLO11_MEDIUM
            ],
        }
        
        # Default models for each task
        self._default_models: Dict[TaskType, ModelType] = {
            TaskType.DETECTION: ModelType.YOLOV8_MEDIUM,
            TaskType.SEGMENTATION: ModelType.YOLOV8_SEG_MEDIUM,
            TaskType.INSTANCE_SEGMENTATION: ModelType.YOLOV8_SEG_MEDIUM,
            TaskType.KEYPOINT_DETECTION: ModelType.YOLOV8_MEDIUM,
            TaskType.CLASSIFICATION: ModelType.YOLOV8_CLS_MEDIUM,
            TaskType.TRACKING: ModelType.YOLOV8_MEDIUM,
            TaskType.VIDEO_TRACKING: ModelType.YOLOV8_MEDIUM,
        }

        self._validate_default_models()
        self._filter_available_models()

    def _filter_available_models(self):
        """Filter task models to only include available ones."""
        from visionary.models import validate_available_models, MODEL_FILE_MAPPING
        
        availability = validate_available_models()
        
        # Filter each task's models to only include available ones
        for task_type in self._task_models:
            available_models = []
            for model_type in self._task_models[task_type]:
                if model_type in MODEL_FILE_MAPPING and availability.get(model_type.value, False):
                    available_models.append(model_type)
            
            self._task_models[task_type] = available_models
            
            # Update default model if current default is not available
            if task_type in self._default_models:
                default_model = self._default_models[task_type]
                if default_model not in available_models and available_models:
                    # Use first available model as new default
                    self._default_models[task_type] = available_models[0]
                    print(f"Updated default model for {task_type} to {available_models[0]}")
    
    def _validate_default_models(self):
        """Validate that default models are available."""
        from visionary.models import validate_available_models, MODEL_FILE_MAPPING
        
        availability = validate_available_models()
        
        for task_type, default_model in self._default_models.items():
            if default_model not in MODEL_FILE_MAPPING:
                raise VisionaryError(f"Default model {default_model} for task {task_type} has no file mapping")
            
            if not availability.get(default_model.value, False):
                print(f"Warning: Default model {default_model} for task {task_type} is not available")
    
    def get_processor(self, 
                     task_type: TaskType, 
                     input_type: InputType,
                     model_config: ModelConfig) -> BaseProcessor:
        """
        Get appropriate processor for the given task and input type.
        
        Args:
            task_type: Type of CV task
            input_type: Type of input data
            model_config: Model configuration
            
        Returns:
            Appropriate processor instance
            
        Raises:
            VisionaryError: If task type is not supported
        """
        if task_type not in self._processors:
            raise VisionaryError(f"Unsupported task type: {task_type}")
            
        processor_class = self._processors[task_type]
        return processor_class(model_config, input_type)
    
    def get_default_model(self, task_type: TaskType) -> ModelType:
        """Get default model for a task type."""
        if task_type not in self._default_models:
            raise VisionaryError(f"No default model for task: {task_type}")
        return self._default_models[task_type]
    
    def get_models_for_task(self, task_type: TaskType) -> List[str]:
        """Get list of supported models for a task."""
        if task_type not in self._task_models:
            return []
        return [model.value for model in self._task_models[task_type]]
    
    def get_all_supported_models(self) -> Dict[str, List[str]]:
        """Get all supported models organized by task."""
        return {
            task.value: self.get_models_for_task(task) 
            for task in self._task_models.keys()
        }
    
    def is_model_supported(self, task_type: TaskType, model_type: ModelType) -> bool:
        """Check if a model is supported for a given task."""
        return (task_type in self._task_models and 
                model_type in self._task_models[task_type])
