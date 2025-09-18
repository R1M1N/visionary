"""
Base adapter interface for model integration in Visionary.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Union
import numpy as np
from ..core import Detections


class BaseAdapter(ABC):
    """
    Abstract base class for model adapters.
    
    All model adapters should inherit from this class and implement
    the required methods for converting model outputs to Detections.
    """
    
    def __init__(self, model_name: str = "unknown"):
        self.model_name = model_name
        self.supported_tasks = ["detection"]
    
    @abstractmethod
    def validate_input(self, results: Any) -> bool:
        """
        Validate if the input results are compatible with this adapter.
        
        Args:
            results: Raw model results
            
        Returns:
            bool: True if compatible, False otherwise
        """
        pass
    
    @abstractmethod
    def extract_detections(self, results: Any) -> Detections:
        """
        Extract object detections from model results.
        
        Args:
            results: Raw model results
            
        Returns:
            Detections object with standardized format
        """
        pass
    
    @abstractmethod
    def extract_segmentation(self, results: Any) -> Detections:
        """
        Extract instance segmentation from model results.
        
        Args:
            results: Raw model results
            
        Returns:
            Detections object with masks included
        """
        pass
    
    def process(self, results: Any, task: str = "detection") -> Detections:
        """
        Main processing method that routes to appropriate extraction method.
        
        Args:
            results: Raw model results
            task: Task type ("detection", "segmentation", "classification")
            
        Returns:
            Detections object
        """
        if not self.validate_input(results):
            raise ValueError(f"Input not compatible with {self.model_name} adapter")
        
        if task == "detection":
            return self.extract_detections(results)
        elif task == "segmentation":
            return self.extract_segmentation(results)
        else:
            raise ValueError(f"Unsupported task: {task}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model and adapter."""
        return {
            "adapter_name": self.__class__.__name__,
            "model_name": self.model_name,
            "supported_tasks": self.supported_tasks
        }
