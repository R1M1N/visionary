# # debug_setup.py
# import os
# from pathlib import Path

# def debug_setup():
#     print(f"Current working directory: {os.getcwd()}")
    
#     model_files_dir = Path("model_files")
#     print(f"Model files directory: {model_files_dir.absolute()}")
#     print(f"Directory exists: {model_files_dir.exists()}")
    
#     if model_files_dir.exists():
#         model_files = list(model_files_dir.glob("*.pt")) + list(model_files_dir.glob("*.pth"))
#         print(f"\nFound {len(model_files)} model files:")
#         for f in sorted(model_files):
#             print(f"  ✓ {f.name}")
        
#         # Check specifically for the required models
#         required = ["yolov8m.pt", "yolov8m-seg.pt", "yolov8m-cls.pt"]
#         print(f"\nChecking required models:")
#         for req in required:
#             exists = (model_files_dir / req).exists()
#             status = "✓ FOUND" if exists else "✗ MISSING"
#             print(f"  {status} {req}")
#     else:
#         print("❌ Model files directory not found!")

# if __name__ == "__main__":
#     debug_setup()

# =======================================================================================================================================
# # debug_models.py
# import sys
# from pathlib import Path

# # Add src to path
# sys.path.insert(0, str(Path(__file__).parent / "src"))

# from visionary.models import validate_available_models, MODEL_FILE_MAPPING
# from visionary.processors.factory import ProcessorFactory

# print("=== MODEL AVAILABILITY CHECK ===")
# availability = validate_available_models()

# print(f"\nModels directory being checked: models/")
# print(f"Total models in mapping: {len(MODEL_FILE_MAPPING)}")

# available_count = 0
# for model_type, filename in MODEL_FILE_MAPPING.items():
#     available = availability.get(model_type.value, False)
#     status = "✅ FOUND" if available else "❌ MISSING"
#     print(f"{status} {model_type.name}: {filename}")
#     if available:
#         available_count += 1

# print(f"\nSummary: {available_count}/{len(MODEL_FILE_MAPPING)} models available")

# print("\n=== PROCESSOR FACTORY CHECK ===")
# try:
#     factory = ProcessorFactory()
#     print("✅ ProcessorFactory created successfully")
    
#     print("\nSupported models by task:")
#     supported = factory.get_all_supported_models()
#     for task, models in supported.items():
#         print(f"  {task}: {models}")
        
# except Exception as e:
#     print(f"❌ ProcessorFactory failed: {e}")
# ========================================================================================================================================
# # debug_api_fixed.py
# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent / "src"))

# from visionary import VisionaryAPI
# from visionary.model_files.task_types import TaskType

# print("=== FIXED API INTERNAL DEBUG ===")

# api = VisionaryAPI()
# print(f"✅ API processor_factory: {api.processor_factory}")

# # Access the correct factory attribute
# factory = api.processor_factory
# print(f"✅ Factory type: {type(factory)}")

# # Check factory's state
# try:
#     print(f"Available tasks: {list(factory._task_models.keys())}")
#     print(f"Detection models: {factory._task_models.get(TaskType.DETECTION, [])}")
    
#     default_detection = factory.get_default_model(TaskType.DETECTION)
#     print(f"✅ Default detection model: {default_detection}")
    
# except Exception as e:
#     print(f"❌ Factory access error: {e}")
    
#     # Debug factory internal state
#     if hasattr(factory, '_default_models'):
#         print(f"Factory default models: {factory._default_models}")
#     if hasattr(factory, '_task_models'):  
#         print(f"Factory task models: {factory._task_models}")

# debug_complete.py
import sys
import traceback
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from visionary import VisionaryAPI
from visionary.model_files.task_types import TaskType
from visionary.models import ModelType
from visionary.utils.input_handler import InputType, detect_input_type


print("=== COMPLETE DEBUG: Unsupported Task Type Error ===\n")

# 1. Test API Creation
print("1. TESTING API CREATION:")
try:
    api = VisionaryAPI()
    print(f"✅ API created: {api}")
except Exception as e:
    print(f"❌ API creation failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# 2. Check ProcessorFactory
print("\n2. CHECKING PROCESSOR FACTORY:")
factory = api.processor_factory
print(f"✅ Factory: {factory}")
print(f"✅ Factory type: {type(factory)}")

# 3. Inspect Factory Internal State
print("\n3. FACTORY INTERNAL STATE:")
print(f"_processors keys: {list(factory._processors.keys())}")
print(f"_task_models keys: {list(factory._task_models.keys())}")
print(f"_default_models: {factory._default_models}")

# 4. Check if TaskType enums exist in _processors
print("\n4. CHECKING TASK TYPE MAPPINGS:")
for task_type in [TaskType.DETECTION, TaskType.VIDEO_TRACKING]:
    exists = task_type in factory._processors
    processor_class = factory._processors.get(task_type, "NOT FOUND")
    print(f"  {task_type}: exists={exists}, processor={processor_class}")

# 5. Test Task Parsing
print("\n5. TESTING TASK PARSING:")
test_tasks = ["detection", "video_tracking", TaskType.DETECTION, TaskType.VIDEO_TRACKING]
for task in test_tasks:
    try:
        parsed = api._parse_task(task)
        print(f"  '{task}' → {parsed} (type: {type(parsed)})")
    except Exception as e:
        print(f"  ❌ '{task}' failed: {e}")

# 6. Test Model Parsing
print("\n6. TESTING MODEL PARSING:")
for model, task_type in [("yolov8m", TaskType.DETECTION), (None, TaskType.DETECTION)]:
    try:
        parsed = api._parse_model(model, task_type)
        print(f"  model='{model}', task={task_type} → {parsed}")
    except Exception as e:
        print(f"  ❌ model='{model}', task={task_type} failed: {e}")

# 7. Test Direct Processor Creation
print("\n7. TESTING DIRECT PROCESSOR CREATION:")
try:
    from visionary.models import ModelConfig
    
    # Create mock input type
    class MockInputType:
        pass
    
    # Test with DETECTION
    task_type = TaskType.DETECTION
    model_type = ModelType.YOLOV8_MEDIUM
    model_config = ModelConfig(model_type)
    
    print(f"  Trying to get processor for {task_type}...")
    print(f"  Model config: {model_config}")
    
    processor = factory.get_processor(task_type, MockInputType(), model_config)
    print(f"  ✅ Got processor: {processor}")
    
except Exception as e:
    print(f"  ❌ Direct processor creation failed: {e}")
    traceback.print_exc()

# 8. Test Full Process Method Logic
print("\n8. TESTING PROCESS METHOD STEP BY STEP:")
try:
    input_data = "data/traffic.jpg"
    task = "detection"
    model = "yolov8m"
    
    print(f"  Input: data='{input_data}', task='{task}', model='{model}'")
    
    # Step 1: Parse task
    task_type = api._parse_task(task)
    print(f"  Step 1 - Parsed task: {task_type}")
    
    # Step 2: Detect input type
    input_type = detect_input_type(input_data)
    print(f"  Step 2 - Input type: {input_type}")
    
    # Step 3: Parse model
    model_type = api._parse_model(model, task_type)
    print(f"  Step 3 - Parsed model: {model_type}")
    
    # Step 4: Create model config
    from visionary.model_files.task_config import TaskConfig
    from visionary.models import ModelConfig
    
    task_config = TaskConfig(task_type)
    model_config = ModelConfig(model_type, device=api.default_device)
    print(f"  Step 4 - Configs created: task_config={task_config}, model_config={model_config}")
    
    # Step 5: Get processor (THIS IS WHERE IT LIKELY FAILS)
    print(f"  Step 5 - Getting processor for task_type={task_type}...")
    print(f"           Available processors: {list(factory._processors.keys())}")
    print(f"           Task type in processors? {task_type in factory._processors}")
    
    processor = factory.get_processor(task_type, input_type, model_config)
    print(f"  ✅ Step 5 - Got processor: {processor}")
    
except Exception as e:
    print(f"  ❌ Process method simulation failed at: {e}")
    traceback.print_exc()

# 9. Compare TaskType Instances
print("\n9. COMPARING TASKTYPE INSTANCES:")
detection1 = TaskType.DETECTION
detection2 = TaskType("detection")
print(f"  TaskType.DETECTION: {detection1}")
print(f"  TaskType('detection'): {detection2}")
print(f"  Are they equal? {detection1 == detection2}")
print(f"  Same ID? {id(detection1) == id(detection2)}")

# 10. Final Summary
print("\n10. SUMMARY:")
print(f"✅ API created successfully")
print(f"✅ Factory has {len(factory._processors)} processors")
print(f"✅ Factory has {len(factory._task_models)} task model mappings")
print(f"✅ Factory has {len(factory._default_models)} default models")

if TaskType.DETECTION in factory._processors:
    print(f"✅ DETECTION task is properly mapped")
else:
    print(f"❌ DETECTION task is NOT in processors!")
    
if TaskType.VIDEO_TRACKING in factory._processors:
    print(f"✅ VIDEO_TRACKING task is properly mapped")
else:
    print(f"❌ VIDEO_TRACKING task is NOT in processors!")

print("\n=== DEBUG COMPLETE ===")
