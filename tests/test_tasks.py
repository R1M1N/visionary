
import os
import cv2
from visionary.utils import image as vi_image
from visionary.utils import video as vi_video
from visionary.detection.adapters.ultralytics import YOLODetector
from visionary.trackers.byte_track import ByteTrack
from visionary.annotators import BoxAnnotator, LabelAnnotator


# Paths
MODEL_DIR = 'models'
DATA_DIR = 'data'
OUTPUT_DIR = 'output'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load models
model_weights = os.path.join(MODEL_DIR, 'yolov8m.pt')
detector = YOLODetector(weights=model_weights, device='cuda')
tracker = ByteTrack()
box_annotator = BoxAnnotator()
label_annotator = LabelAnnotator()


def process_image(img_path):
    img = vi_image.read(img_path)
    detections = detector.predict(img)

    boxes = [det[0] for det in detections]
    labels = [f"{detector.names[det[2]]} {i+1}" for i, det in enumerate(detections)]

    img_annotated = box_annotator.annotate(img, boxes)
    img_annotated = label_annotator.annotate(img_annotated, boxes, labels)

    save_path = os.path.join(OUTPUT_DIR, 'processed_image.jpg')
    vi_image.write(save_path, img_annotated)
    print(f"Saved annotated image to {save_path}")


# Video processing

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

        detections = detector.predict(frame)
        tracks = tracker.update(detections, frame)

        boxes = [track['bbox'] for track in tracks]
        labels = [f"{detector.names[track['class']]} {track['track_id']}" for track in tracks]

        frame = box_annotator.annotate(frame, boxes)
        frame = label_annotator.annotate(frame, boxes, labels)

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
