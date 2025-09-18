"""
Input type detection and handling utilities
"""
from enum import Enum
from pathlib import Path
from typing import Union
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None


class InputType(Enum):
    """Types of input data supported."""
    IMAGE_PATH = "image_path"
    VIDEO_PATH = "video_path"
    NUMPY_ARRAY = "numpy_array"
    PIL_IMAGE = "pil_image"
    URL = "url"
    UNKNOWN = "unknown"


def detect_input_type(input_data: Union[str, Path, np.ndarray, "PIL.Image"]) -> InputType:
    """
    Detect the type of input data.
    
    Args:
        input_data: Input to analyze
        
    Returns:
        InputType enum representing the detected type
    """
    # Handle string/path inputs
    if isinstance(input_data, (str, Path)):
        path_str = str(input_data).lower()
        
        # Check if it's a URL
        if path_str.startswith(('http://', 'https://', 'ftp://')):
            return InputType.URL
            
        # Check file extension for images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'}
        
        path_obj = Path(path_str)
        extension = path_obj.suffix.lower()
        
        if extension in image_extensions:
            return InputType.IMAGE_PATH
        elif extension in video_extensions:
            return InputType.VIDEO_PATH
            
    # Handle numpy arrays
    elif isinstance(input_data, np.ndarray):
        return InputType.NUMPY_ARRAY
        
    # Handle PIL Images
    elif Image and isinstance(input_data, Image.Image):
        return InputType.PIL_IMAGE
        
    return InputType.UNKNOWN


def validate_input(input_data: Union[str, Path, np.ndarray, "PIL.Image"]) -> bool:
    """
    Validate that input data exists and is accessible.
    
    Args:
        input_data: Input to validate
        
    Returns:
        True if input is valid, False otherwise
    """
    input_type = detect_input_type(input_data)
    
    if input_type in [InputType.IMAGE_PATH, InputType.VIDEO_PATH]:
        return Path(input_data).exists()
    elif input_type == InputType.NUMPY_ARRAY:
        return len(input_data.shape) >= 2  # At least 2D array
    elif input_type == InputType.PIL_IMAGE:
        return hasattr(input_data, 'size')
    elif input_type == InputType.URL:
        return True  # Basic URL format validation already done
        
    return False
