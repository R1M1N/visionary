from typing import Union, Optional, Dict, Any, List
from pathlib import Path
import logging

from visionary.model_files.task_types import TaskType  # ← Use this import
from visionary.model_files.task_config import TaskConfig  # ← Add this import
from .models import ModelType, ModelConfig
from .utils.input_handler import InputType, detect_input_type
from .processors import ProcessorFactory
from .processors.exceptions import VisionaryError



class VisionaryAPI:
    """
    Unified interface for computer vision tasks.
    
    Automatically selects appropriate models and performs tasks on images/videos.
    """
    
    def __init__(self, 
                 default_device: str = "auto",
                 cache_models: bool = True,
                 log_level: str = "INFO"):
        """
        Initialize the Visionary API.
        
        Args:
            default_device: Default device for inference ("cpu", "cuda", "auto")
            cache_models: Whether to cache loaded models
            log_level: Logging level
        """
        self.default_device = default_device
        self.cache_models = cache_models
        self.processor_factory = ProcessorFactory()
        self._setup_logging(log_level)
        
    def process(self, 
                input_data: Union[str, Path, "np.ndarray", "PIL.Image"],
                task: Union[str, TaskType],
                model: Optional[Union[str, ModelType]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Main processing function that automatically handles any CV task.
        
        Args:
            input_data: Path to image/video, numpy array, or PIL image
            task: Task to perform (detection, segmentation, keypoint, tracking, etc.)
            model: Specific model to use (auto-selected if None)
            **kwargs: Additional parameters for the task
            
        Returns:
            Dictionary containing results and metadata
        """
        try:
            # Parse inputs - Use correct parsing order
            task_type = self._parse_task(task)
            input_type = detect_input_type(input_data)
            model_type = self._parse_model(model, task_type)  # Pass task_type here
            
            # Create configurations
            task_config = TaskConfig(task_type, **kwargs)
            model_config = ModelConfig(model_type, device=self.default_device)
            
            # Get appropriate processor
            processor = self.processor_factory.get_processor(
                task_type, input_type, model_config
            )
            
            # Process the input
            results = processor.process(input_data, task_config)
            
            return {
                'results': results,
                'metadata': {
                    'task': task_type.value,
                    'model': model_type.value,
                    'input_type': input_type.value,
                    'success': True
                }
            }
        
        except Exception as e:
            logging.error(f"Processing failed: {str(e)}")
            return {
                'results': None,
                'metadata': {
                    'task': task if isinstance(task, str) else task.value,
                    'success': False,
                    'error': str(e)
                }
            }
    
    def batch_process(self, 
                     inputs: List[Union[str, Path]], 
                     task: Union[str, TaskType],
                     **kwargs) -> List[Dict[str, Any]]:
        """Process multiple inputs in batch."""
        return [self.process(inp, task, **kwargs) for inp in inputs]
    
    def get_supported_tasks(self) -> List[str]:
        """Get list of supported tasks."""
        return [task.value for task in TaskType]
    
    def get_supported_models(self, task: Optional[str] = None) -> Dict[str, List[str]]:
        """Get supported models for each task or specific task."""
        if task:
            task_type = self._parse_task(task)
            return {task: self.processor_factory.get_models_for_task(task_type)}
        return self.processor_factory.get_all_supported_models()
    
    def _parse_task(self, task: Union[str, TaskType]) -> TaskType:
        """Parse task input to TaskType enum."""
        if isinstance(task, str):
            try:
                return TaskType(task.lower())
            except ValueError:
                raise VisionaryError(f"Unsupported task: {task}")
        return task
    
    def _parse_model(self, model: Optional[Union[str, ModelType]], 
                    task_type: TaskType) -> ModelType:
        """Parse model input and auto-select if needed."""
        if model is None:
            return self.processor_factory.get_default_model(task_type)
        
        if isinstance(model, str):
            # Find matching ModelType by value
            for model_type in ModelType:
                if model_type.value == model:
                    return model_type
            raise VisionaryError(f"Unknown model: {model}")
        return model
    
    def _setup_logging(self, level: str):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
