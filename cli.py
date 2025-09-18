"""Command-line interface for Visionary."""

import argparse
from .unified_interface import VisionaryAPI

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Visionary Computer Vision CLI')
    parser.add_argument('input', help='Input image or video path')
    parser.add_argument('--task', choices=['detection', 'video_tracking'], 
                       default='detection', help='Task to perform')
    parser.add_argument('--model', default='yolov8m', help='Model to use')
    parser.add_argument('--output', help='Output path')
    
    args = parser.parse_args()
    
    api = VisionaryAPI()
    result = api.process(args.input, task=args.task, model=args.model)
    
    print(f"✅ Processed {args.input}")
    print(f"📊 Results: {len(result.get('results', {}).get('predictions', []))} detections")
    
    if args.output:
        print(f"💾 Saved to: {args.output}")

if __name__ == '__main__':
    main()
