"""
Quick inference test using the trained YOLO model.
This script will run inference on a few evaluation images to test the pipeline.
"""

import json
import os
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from PIL import Image

def quick_inference_test():
    """Run inference on a few images to test the pipeline"""
    
    # Model path from test training
    model_path = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/runs/segment/test_run/weights/best.pt"
    evaluation_images_dir = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/evaluation_images"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please run test_yolo_setup.py first to create a test model")
        return False
    
    # Check if evaluation images exist
    if not os.path.exists(evaluation_images_dir):
        print(f"❌ Evaluation images not found at: {evaluation_images_dir}")
        return False
    
    print("✅ Model and evaluation images found!")
    
    # Load model
    model = YOLO(model_path)
    print("✅ Model loaded successfully!")
    
    # Get first few evaluation images
    image_files = sorted([f for f in os.listdir(evaluation_images_dir) if f.endswith('.tif')])[:3]
    
    print(f"🧪 Testing inference on {len(image_files)} images...")
    
    class_names = {0: 'individual_tree', 1: 'group_of_trees'}
    results_data = {"images": []}
    
    for image_file in image_files:
        print(f"Processing: {image_file}")
        
        image_path = os.path.join(evaluation_images_dir, image_file)
        
        # Load image to get dimensions
        img = Image.open(image_path)
        width, height = img.size
        
        # Extract metadata from filename
        if image_file.startswith('10cm'):
            cm_resolution = 10
            scene_type = "agriculture_plantation"
        elif image_file.startswith('20cm'):
            cm_resolution = 20
            scene_type = "mixed_forest"
        elif image_file.startswith('40cm'):
            cm_resolution = 40
            scene_type = "industrial_area"
        else:
            cm_resolution = 10
            scene_type = "mixed_forest"
        
        # Run inference
        results = model.predict(
            source=image_path,
            conf=0.25,
            iou=0.45,
            save=False,
            verbose=False
        )
        
        # Process results
        annotations = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.masks is not None:
                for j in range(len(result.boxes)):
                    confidence = float(result.boxes.conf[j])
                    class_id = int(result.boxes.cls[j])
                    class_name = class_names.get(class_id, 'unknown')
                    
                    # Get mask and convert to polygon
                    mask = result.masks.xy[j]
                    
                    if len(mask) > 0:
                        segmentation = mask.flatten().tolist()
                        
                        if len(segmentation) >= 6:
                            annotation = {
                                "class": class_name,
                                "confidence_score": round(confidence, 2),
                                "segmentation": [round(coord, 1) for coord in segmentation]
                            }
                            annotations.append(annotation)
        
        # Create result
        image_result = {
            "file_name": image_file,
            "width": width,
            "height": height,
            "cm_resolution": cm_resolution,
            "scene_type": scene_type,
            "annotations": annotations
        }
        
        results_data["images"].append(image_result)
        print(f"  Found {len(annotations)} annotations")
    
    # Save test results
    output_path = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/test_predictions.json"
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n✅ Inference test completed!")
    print(f"📄 Test results saved to: {output_path}")
    
    # Print summary
    total_annotations = sum(len(img["annotations"]) for img in results_data["images"])
    print(f"📊 Total annotations: {total_annotations}")
    
    return True

if __name__ == "__main__":
    print("🧪 YOLO Inference Test")
    print("======================")
    
    success = quick_inference_test()
    
    if success:
        print("\n🎉 Inference test successful!")
        print("📋 Your YOLO pipeline is working end-to-end!")
        print("\n📈 Next steps:")
        print("1. Run longer training: python train_yolo.py")
        print("2. Run full inference: python inference_yolo.py")
        print("3. Experiment with different confidence thresholds")
    else:
        print("\n❌ Inference test failed. Check the errors above.")