"""
Quick improvement script - Apply recommended fixes and re-run inference
"""

import shutil
from pathlib import Path

def create_improved_inference_script():
    """Create an improved version of inference_yolo.py with better settings"""
    
    inference_code = '''"""
IMPROVED YOLO inference script with optimized settings
"""

import json
import os
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from PIL import Image
import cv2

class ImprovedYOLOInference:
    def __init__(self, model_path, evaluation_images_dir, output_path):
        self.model_path = model_path
        self.evaluation_images_dir = Path(evaluation_images_dir)
        self.output_path = output_path
        
        # Load trained model
        self.model = YOLO(model_path)
        
        # Class mapping
        self.class_names = {
            0: 'individual_tree',
            1: 'group_of_trees'
        }
        
    def extract_scene_info(self, filename):
        """Extract cm_resolution and scene_type from filename"""
        if filename.startswith('10cm'):
            cm_resolution = 10
        elif filename.startswith('20cm'):
            cm_resolution = 20
        elif filename.startswith('40cm'):
            cm_resolution = 40
        elif filename.startswith('80cm'):
            cm_resolution = 80
        else:
            cm_resolution = 10
        
        scene_type_map = {
            10: "agriculture_plantation",
            20: "mixed_forest",
            40: "industrial_area",
            80: "urban_area"
        }
        scene_type = scene_type_map.get(cm_resolution, "mixed_forest")
        
        return cm_resolution, scene_type
    
    def polygon_to_segmentation(self, polygon):
        """Convert polygon coordinates to segmentation format"""
        if len(polygon.shape) == 2:
            return polygon.flatten().tolist()
        return polygon.tolist()
    
    def run_inference(self, confidence_threshold=0.25, iou_threshold=0.7, max_det=1000):
        """Run inference with improved settings"""
        
        image_files = sorted([f for f in os.listdir(self.evaluation_images_dir) 
                             if f.endswith('.tif')])
        
        results_data = {"images": []}
        
        print(f"\\n{'='*60}")
        print(f"IMPROVED INFERENCE SETTINGS")
        print(f"{'='*60}")
        print(f"Confidence Threshold: {confidence_threshold}")
        print(f"IoU Threshold: {iou_threshold}")
        print(f"Max Detections: {max_det}")
        print(f"Processing {len(image_files)} images...")
        print(f"{'='*60}\\n")
        
        class_counts = {'individual_tree': 0, 'group_of_trees': 0}
        
        for i, image_file in enumerate(image_files):
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{len(image_files)} images")
            
            image_path = self.evaluation_images_dir / image_file
            
            img = Image.open(image_path)
            width, height = img.size
            
            cm_resolution, scene_type = self.extract_scene_info(image_file)
            
            # IMPROVED: Added max_det parameter
            results = self.model.predict(
                source=str(image_path),
                conf=confidence_threshold,
                iou=iou_threshold,
                max_det=max_det,  # IMPORTANT: Increased from default 300
                save=False,
                verbose=False
            )
            
            annotations = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.masks is not None:
                    for j in range(len(result.boxes)):
                        confidence = float(result.boxes.conf[j])
                        class_id = int(result.boxes.cls[j])
                        class_name = self.class_names.get(class_id, 'unknown')
                        
                        # Track class distribution
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        
                        mask = result.masks.xy[j]
                        
                        if len(mask) > 0:
                            segmentation = self.polygon_to_segmentation(mask)
                            
                            if len(segmentation) >= 6:
                                annotation = {
                                    "class": class_name,
                                    "confidence_score": round(confidence, 2),
                                    "segmentation": [round(coord, 1) for coord in segmentation]
                                }
                                annotations.append(annotation)
            
            image_result = {
                "file_name": image_file,
                "width": width,
                "height": height,
                "cm_resolution": cm_resolution,
                "scene_type": scene_type,
                "annotations": annotations
            }
            
            results_data["images"].append(image_result)
        
        # Save results
        with open(self.output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\\n{'='*60}")
        print(f"INFERENCE COMPLETED")
        print(f"{'='*60}")
        print(f"Results saved to: {self.output_path}")
        print(f"Total images processed: {len(image_files)}")
        
        total_annotations = sum(len(img["annotations"]) for img in results_data["images"])
        print(f"Total annotations: {total_annotations}")
        print(f"Average per image: {total_annotations/len(image_files):.1f}")
        
        print(f"\\nClass Distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Check if both classes detected
        if class_counts.get('group_of_trees', 0) == 0:
            print(f"\\n⚠️  WARNING: No 'group_of_trees' detected!")
            print(f"   This is likely causing low score.")
            print(f"   Recommendations:")
            print(f"   1. Lower confidence threshold further (try 0.15-0.2)")
            print(f"   2. Check if model was trained on both classes")
            print(f"   3. Consider retraining with class weights")
        elif class_counts.get('group_of_trees', 0) < total_annotations * 0.05:
            print(f"\\n⚠️  WARNING: Very few 'group_of_trees' detected ({class_counts['group_of_trees']})")
            print(f"   Training data has ~10% group_of_trees, but predictions have <5%")
            print(f"   Model may need retraining or lower threshold for this class")
        
        print(f"{'='*60}\\n")
        
        return results_data

def main():
    """Main function with improved settings"""
    
    # Paths
    evaluation_images_dir = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/evaluation_images"
    output_path = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/improved_predictions.json"
    model_path = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/runs/segment/tree_segmentation2/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        return
    
    if not os.path.exists(evaluation_images_dir):
        print(f"Evaluation images not found at: {evaluation_images_dir}")
        return
    
    # Run inference with IMPROVED settings
    inference = ImprovedYOLOInference(model_path, evaluation_images_dir, output_path)
    
    # IMPROVED THRESHOLDS - More conservative to reduce false positives
    confidence_threshold = 0.25  # Increased from 0.1 (less false positives)
    iou_threshold = 0.7          # Increased from 0.6 (better NMS)
    max_det = 1000               # Increased from default 300 (no cap)
    
    results = inference.run_inference(
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        max_det=max_det
    )
    
    print("\\n✅ Improved inference completed!")
    print(f"Compare this with optimized_predictions.json to see the difference.")

if __name__ == "__main__":
    main()
'''
    
    # Save improved inference script
    with open('inference_yolo_improved.py', 'w') as f:
        f.write(inference_code)
    
    print("✅ Created: inference_yolo_improved.py")
    return 'inference_yolo_improved.py'

def create_training_improvement_script():
    """Create improved training script"""
    
    training_code = '''"""
Improved YOLO training script with better settings for class imbalance
"""

from ultralytics import YOLO
import os

def main():
    """Train YOLO with improved settings"""
    
    print("="*60)
    print("IMPROVED YOLO TRAINING")
    print("="*60)
    print()
    print("Improvements:")
    print("  - Using YOLOv8m-seg (medium) instead of nano")
    print("  - Training for 50 epochs instead of 10")
    print("  - Enhanced augmentation")
    print("  - Early stopping with patience=15")
    print("="*60)
    print()
    
    # Use MEDIUM model for better capacity
    model = YOLO('yolov8m-seg.pt')
    
    # Train with improved settings
    results = model.train(
        data='yolo_dataset/data.yaml',
        epochs=50,              # Increased from 10
        batch=4,                # Reduced for CPU
        imgsz=640,
        device='cpu',
        patience=15,            # Early stopping
        save=True,
        project='runs/segment',
        name='tree_segmentation_improved',
        
        # Enhanced augmentation
        degrees=15.0,           # Rotation augmentation
        translate=0.1,          # Translation augmentation
        scale=0.5,              # Scaling augmentation
        fliplr=0.5,             # Horizontal flip
        flipud=0.0,             # No vertical flip
        mosaic=1.0,             # Mosaic augmentation
        mixup=0.1,              # Mixup augmentation
        
        # Other settings
        optimizer='AdamW',      # Better optimizer
        lr0=0.001,             # Initial learning rate
        warmup_epochs=3,        # Warmup
        
        verbose=True
    )
    
    print(f"\\nTraining completed!")
    print(f"Best model saved at: runs/segment/tree_segmentation_improved/weights/best.pt")
    print(f"\\nTo use this model, update your inference script with:")
    print(f"  model_path = 'runs/segment/tree_segmentation_improved/weights/best.pt'")

if __name__ == "__main__":
    main()
'''
    
    with open('train_yolo_improved.py', 'w') as f:
        f.write(training_code)
    
    print("✅ Created: train_yolo_improved.py")
    return 'train_yolo_improved.py'

def main():
    print("="*80)
    print("QUICK IMPROVEMENT SETUP")
    print("="*80)
    print()
    
    # Create improved scripts
    inference_script = create_improved_inference_script()
    training_script = create_training_improvement_script()
    
    print()
    print("="*80)
    print("QUICK START GUIDE")
    print("="*80)
    print()
    print("📊 STEP 1: Try improved inference (5 minutes)")
    print("   This uses better thresholds and removes max detection cap")
    print()
    print("   Run: python3 inference_yolo_improved.py")
    print()
    print("   This will create 'improved_predictions.json' with:")
    print("   - Confidence threshold: 0.25 (instead of 0.1)")
    print("   - IoU threshold: 0.7 (instead of 0.6)")
    print("   - Max detections: 1000 (instead of 300)")
    print()
    print("   Expected: Fewer but more accurate detections")
    print("   Expected: Detection of group_of_trees class")
    print()
    print("="*80)
    print()
    print("🔧 STEP 2: If still stuck, retrain with better model (2-3 hours)")
    print("   This uses medium model and trains for 50 epochs")
    print()
    print("   Run: python3 train_yolo_improved.py")
    print()
    print("   Then update inference to use new model:")
    print("   - Edit inference_yolo_improved.py")
    print("   - Change model_path to: 'runs/segment/tree_segmentation_improved/weights/best.pt'")
    print("   - Run: python3 inference_yolo_improved.py")
    print()
    print("="*80)
    print()
    print("📈 STEP 3: Analyze results")
    print()
    print("   Run: python3 analyze_for_improvement.py")
    print()
    print("   This will show you:")
    print("   - How many of each class detected")
    print("   - Whether you're still hitting max_det cap")
    print("   - Threshold effectiveness")
    print()
    print("="*80)
    print()
    print("💡 KEY INSIGHT from your data:")
    print()
    print("   Training data class distribution:")
    print("   - individual_tree: 23,469 (90.5%)")
    print("   - group_of_trees: 2,476 (9.5%)")
    print()
    print("   Your predictions:")
    print("   - individual_tree: 31,947 (100%)")
    print("   - group_of_trees: 0 (0%)")
    print()
    print("   The model learned one class well but not the other!")
    print("   This is causing your low score.")
    print()
    print("="*80)
    print()
    print("🎯 RECOMMENDED PATH:")
    print()
    print("   1. Try improved inference first (5 min)")
    print("      python3 inference_yolo_improved.py")
    print()
    print("   2. If group_of_trees still 0%, try threshold sweep:")
    print("      python3 optimize_thresholds.py")
    print()
    print("   3. If still no improvement, retrain (2-3 hours)")
    print("      python3 train_yolo_improved.py")
    print()
    print("="*80)
    print()
    print("✅ Setup complete! Start with Step 1 above.")
    print()

if __name__ == "__main__":
    main()
