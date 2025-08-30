"""
Model adapter factory for Visionary.
"""

from typing import Any, Optional, Dict, Type
from .base import BaseAdapter
from .ultralytics import UltralyticsAdapter
from .transformers import TransformersAdapter
from .detectron2 import Detectron2Adapter


class AdapterFactory:
    """Factory class for creating appropriate model adapters."""
    
    _adapters: Dict[str, Type[BaseAdapter]] = {
        'ultralytics': UltralyticsAdapter,
        'yolo': UltralyticsAdapter,
        'transformers': TransformersAdapter,
        'detr': TransformersAdapter,
        'detectron2': Detectron2Adapter,
    }
    
    @classmethod
    def create_adapter(cls, model_type: str) -> BaseAdapter:
        """
        Create appropriate adapter for model type.
        
        Args:
            model_type: Type of model ('ultralytics', 'transformers', etc.)
            
        Returns:
            BaseAdapter instance
        """
        model_type = model_type.lower()
        
        if model_type in cls._adapters:
            return cls._adapters[model_type]()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @classmethod
    def auto_detect_adapter(cls, results: Any) -> Optional[BaseAdapter]:
        """
        Automatically detect appropriate adapter based on results format.
        
        Args:
            results: Model results
            
        Returns:
            BaseAdapter instance or None if not detected
        """
        for adapter_class in cls._adapters.values():
            adapter = adapter_class()
            if adapter.validate_input(results):
                return adapter
        
        return None
    
    @classmethod
    def register_adapter(cls, name: str, adapter_class: Type[BaseAdapter]):
        """Register a custom adapter."""
        cls._adapters[name] = adapter_class


# Convenience functions
def create_adapter(model_type: str) -> BaseAdapter:
    """Create adapter for specified model type."""
    return AdapterFactory.create_adapter(model_type)


def auto_detect_adapter(results: Any) -> Optional[BaseAdapter]:
    """Auto-detect appropriate adapter."""
    return AdapterFactory.auto_detect_adapter(results)


__all__ = [
    'BaseAdapter',
    'UltralyticsAdapter', 
    'TransformersAdapter',
    'Detectron2Adapter',
    'AdapterFactory',
    'create_adapter',
    'auto_detect_adapter'
]
