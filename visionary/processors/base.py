"""
Base processor class for all computer vision tasks
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Union
from pathlib import Path
import numpy as np

from visionary.models import ModelConfig, ModelType
from visionary.model_files.task_types import TaskType
from visionary.model_files.task_config import TaskConfig
from visionary.utils.input_handler import InputType


class BaseProcessor(ABC):
    """Abstract base class for all task processors."""
    
    def __init__(self, model_config: ModelConfig, input_type: InputType):
        """
        Initialize the processor.
        
        Args:
            model_config: Configuration for the model
            input_type: Type of input data this processor will handle
        """
        self.model_config = model_config
        self.input_type = input_type
        self.model = None
        self._load_model()
    
    @abstractmethod
    def _load_model(self):
        """Load the model based on configuration."""
        pass
    
    @abstractmethod
    def process(self, 
                input_data: Union[str, Path, np.ndarray, "PIL.Image"],
                task_config: TaskConfig) -> Dict[str, Any]:
        """
        Process input data with the loaded model.
        
        Args:
            input_data: Input to process
            task_config: Task-specific configuration
            
        Returns:
            Dictionary containing processing results
        """
        pass
    
    def _preprocess_input(self, input_data):
        """Preprocess input data before model inference."""
        # Common preprocessing logic can go here
        return input_data
    
    def _postprocess_output(self, output, task_config: TaskConfig):
        """Postprocess model output."""
        # Common postprocessing logic can go here
        return output
