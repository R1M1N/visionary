import os
import cv2
from visionary.utils.image import ImageUtils, ImageSink
from visionary.detection.adapters.ultralytics import UltralyticsAdapter
from visionary.trackers.byte_track import ByteTracker
from visionary.annotators import BoxAnnotator, LabelAnnotator

# Paths
MODEL_DIR = 'models'
DATA_DIR = 'data'
OUTPUT_DIR = 'output'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load models - Fix: Use 'weights' parameter instead of 'model_path'
model_weights = os.path.join(MODEL_DIR, 'yolov8m.pt')
detector = UltralyticsAdapter(weights=model_weights, device='cuda')
tracker = ByteTracker()
box_annotator = BoxAnnotator()
label_annotator = LabelAnnotator()

def process_image(img_path):
    # Load image using ImageUtils
    img = ImageUtils.load_image(img_path)
    
    # Use the predict method (now that model is properly loaded)
    results = detector.predict(img)
    result = results[0] if isinstance(results, list) else results
    
    # Process through your adapter to get standardized Detections
    detections = detector.process(result, task="detection")
    
    # Extract data for annotation
    boxes = detections.xyxy
    
    # Create labels
    labels = []
    if detections.class_id is not None and hasattr(detector.model, 'names'):
        for i, cls_id in enumerate(detections.class_id):
            class_name = detector.model.names[int(cls_id)]
            labels.append(f"{class_name} {i+1}")
    else:
        labels = [f"Object {i+1}" for i in range(len(boxes))]
    
    # Apply annotations
    img_annotated = box_annotator.annotate(img, boxes)
    # Convert boxes and labels to the format expected by LabelAnnotator
    detections = [{'bbox': box, 'label': label} for box, label in zip(boxes, labels)]

    # Now call annotate with the correct format
    img_annotated = label_annotator.annotate(img_annotated, detections)

    
    # Save using ImageSink
    sink = ImageSink(OUTPUT_DIR, prefix='processed_', extension='.jpg')
    sink.write(img_annotated)
    print(f"Saved annotated image")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    save_path = os.path.join(OUTPUT_DIR, 'processed_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Same fix for video processing
        results = detector.predict(frame)
        result = results[0] if isinstance(results, list) else results
        detections = detector.process(result, task="detection")
        
        boxes = detections.xyxy
        labels = []
        if detections.class_id is not None and hasattr(detector.model, 'names'):
            for i, cls_id in enumerate(detections.class_id):
                class_name = detector.model.names[int(cls_id)]
                labels.append(f"{class_name} {i+1}")
        else:
            labels = [f"Object {i+1}" for i in range(len(boxes))]

        frame = box_annotator.annotate(frame, boxes)
        # Convert boxes and labels to the format expected by LabelAnnotator
        detections = [{'bbox': box, 'label': label} for box, label in zip(boxes, labels)]
        frame = label_annotator.annotate(frame, detections)


        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Saved processed video to {save_path}")

if __name__ == '__main__':
    image_path = os.path.join(DATA_DIR, 'detect(1).jpg')
    video_path = os.path.join(DATA_DIR, 'driving_10s.mp4')

    process_image(image_path)
    process_video(video_path)

    print("Processing completed.")
