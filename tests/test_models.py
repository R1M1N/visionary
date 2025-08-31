# tests/test_models.py
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from visionary.models import validate_available_models, MODEL_FILE_MAPPING

def test_model_availability():
    print("Testing model availability...")
    
    availability = validate_available_models()
    
    print(f"\nModel availability status:")
    for model_type, filename in MODEL_FILE_MAPPING.items():
        available = availability.get(model_type.value, False)
        status = "✓ AVAILABLE" if available else "✗ MISSING"
        print(f"  {status} {model_type.name}: {filename}")
    
    print(f"\nTotal models: {len(MODEL_FILE_MAPPING)}")
    available_count = sum(availability.values())
    print(f"Available models: {available_count}")
    
    if available_count == len(MODEL_FILE_MAPPING):
        print("🎉 All models are available!")
    else:
        missing = len(MODEL_FILE_MAPPING) - available_count
        print(f"⚠️  {missing} models are missing")

if __name__ == "__main__":
    test_model_availability()
