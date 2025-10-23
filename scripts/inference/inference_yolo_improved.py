"""
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
        
        print(f"\n{'='*60}")
        print(f"IMPROVED INFERENCE SETTINGS")
        print(f"{'='*60}")
        print(f"Confidence Threshold: {confidence_threshold}")
        print(f"IoU Threshold: {iou_threshold}")
        print(f"Max Detections: {max_det}")
        print(f"Processing {len(image_files)} images...")
        print(f"{'='*60}\n")
        
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
        
        print(f"\n{'='*60}")
        print(f"INFERENCE COMPLETED")
        print(f"{'='*60}")
        print(f"Results saved to: {self.output_path}")
        print(f"Total images processed: {len(image_files)}")
        
        total_annotations = sum(len(img["annotations"]) for img in results_data["images"])
        print(f"Total annotations: {total_annotations}")
        print(f"Average per image: {total_annotations/len(image_files):.1f}")
        
        print(f"\nClass Distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Check if both classes detected
        if class_counts.get('group_of_trees', 0) == 0:
            print(f"\n⚠️  WARNING: No 'group_of_trees' detected!")
            print(f"   This is likely causing low score.")
            print(f"   Recommendations:")
            print(f"   1. Lower confidence threshold further (try 0.15-0.2)")
            print(f"   2. Check if model was trained on both classes")
            print(f"   3. Consider retraining with class weights")
        elif class_counts.get('group_of_trees', 0) < total_annotations * 0.05:
            print(f"\n⚠️  WARNING: Very few 'group_of_trees' detected ({class_counts['group_of_trees']})")
            print(f"   Training data has ~10% group_of_trees, but predictions have <5%")
            print(f"   Model may need retraining or lower threshold for this class")
        
        print(f"{'='*60}\n")
        
        return results_data

def main():
    """Main function with improved settings"""
    
    # Get project root (2 levels up from this script)
    project_root = Path(__file__).parent.parent.parent
    
    # Paths (relative to project root)
    evaluation_images_dir = project_root / "data" / "evaluation_images"
    output_path = project_root / "results" / "improved_predictions.json"
    model_path = project_root / "runs" / "segment" / "tree_segmentation2" / "weights" / "best.pt"
    
    if not model_path.exists():
        print(f"Model not found at: {model_path}")
        return
    
    if not evaluation_images_dir.exists():
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
    
    print("\n✅ Improved inference completed!")
    print(f"Compare this with optimized_predictions.json to see the difference.")

if __name__ == "__main__":
    main()
