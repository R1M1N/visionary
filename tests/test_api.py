# tests/test_api.py
import sys
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from visionary import VisionaryAPI

def draw_detection_boxes(image_path, predictions, output_path):
    """Draw real YOLO detection boxes on image."""
    if not predictions:
        print("⚠️ No predictions to draw")
        return
        
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for pred in predictions:
        bbox = pred.get('bbox', [])
        cls = pred.get('class', 'unknown')
        conf = pred.get('confidence', 0.0)
        
        if bbox and len(bbox) == 4:
            # Draw blue bounding box (like your screenshot)
            draw.rectangle(bbox, outline='blue', width=3)
            
            # Prepare label
            label = f"{cls} {conf:.2f}"
            
            # Get text size for background
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Draw blue label background
            label_bg = [bbox[0], bbox[1] - text_height - 4, 
                       bbox[0] + text_width + 4, bbox[1]]
            draw.rectangle(label_bg, fill='blue')
            
            # Draw white label text
            draw.text((bbox[0] + 2, bbox[1] - text_height - 2), 
                     label, font=font, fill='white')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"✅ Saved real detection image: {output_path}")

def draw_tracking_video(video_path, frame_tracks, output_path):
    """Draw real YOLO tracking boxes on video frames."""
    if not frame_tracks:
        print("⚠️ No tracks to draw")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
        
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw tracks for this frame
        if frame_count in frame_tracks:
            for track in frame_tracks[frame_count]:
                bbox = track.get('bbox', [])
                track_id = track.get('track_id', 0)
                cls = track.get('class', 'unknown')
                
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    
                    # Draw blue rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Add track label
                    label = f"ID {track_id}: {cls}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    
                    # Draw label background
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0], y1), (255, 0, 0), -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x1, y1 - 8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    print(f"✅ Saved real tracking video: {output_path} ({frame_count} frames)")

def test_real_yolo():
    print("=== TESTING REAL YOLO INFERENCE ===")
    
    api = VisionaryAPI()
    print("✅ VisionaryAPI created successfully")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Test real detection
    try:
        print(f"\n🔄 Running REAL detection on traffic.jpg...")
        detection_result = api.process("data/traffic.jpg", task="detection", model="yolov8m")
        
        if detection_result['metadata']['success']:
            predictions = detection_result.get('results', {}).get('predictions', [])
            
            print(f"✅ Found {len(predictions)} real detections!")
            for pred in predictions[:3]:  # Show first 3
                print(f"  - {pred['class']}: {pred['confidence']:.2f}")
            
            if predictions:
                # Draw and save real detections
                draw_detection_boxes("data/traffic.jpg", predictions, "output/traffic_real_detection.jpg")
            
            # Save raw JSON
            with open("output/detection_real.json", 'w') as f:
                json.dump(detection_result, f, indent=4)
                
        else:
            print(f"❌ Detection failed: {detection_result['metadata']['error']}")
            
    except Exception as e:
        print(f"❌ Detection failed: {e}")
    
    # Test real tracking
    try:
        print(f"\n🔄 Running REAL tracking on driving_10s.mp4...")
        tracking_result = api.process("data/driving_10s.mp4", task="video_tracking", model="yolov8m")
        
        if tracking_result['metadata']['success']:
            frame_tracks = tracking_result.get('results', {}).get('frame_tracks', {})
            total_tracks = tracking_result.get('results', {}).get('total_tracks', 0)
            
            print(f"✅ Found {total_tracks} unique tracks across {len(frame_tracks)} frames!")
            
            if frame_tracks:
                # Draw and save real tracking video
                draw_tracking_video("data/driving_10s.mp4", frame_tracks, "output/driving_real_tracking.mp4")
            
            # Save raw JSON
            with open("output/tracking_real.json", 'w') as f:
                json.dump(tracking_result, f, indent=4)
                
        else:
            print(f"❌ Tracking failed: {tracking_result['metadata']['error']}")
            
    except Exception as e:
        print(f"❌ Tracking failed: {e}")
    
    print(f"\n🎉 Check 'output/' folder for REAL annotated results!")

if __name__ == "__main__":
    test_real_yolo()
