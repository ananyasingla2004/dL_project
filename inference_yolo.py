"""
YOLO inference script for tree segmentation on evaluation images.
This script runs inference and formats output according to the sample_answer.json format.
"""

import json
import os
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from PIL import Image
import cv2

class YOLOInference:
    def __init__(self, model_path, evaluation_images_dir, output_path):
        self.model_path = model_path
        self.evaluation_images_dir = Path(evaluation_images_dir)
        self.output_path = output_path
        
        # Load trained model
        self.model = YOLO(model_path)
        
        # Class mapping (should match training)
        self.class_names = {
            0: 'individual_tree',
            1: 'group_of_trees'
        }
        
    def extract_scene_info(self, filename):
        """Extract cm_resolution and scene_type from filename"""
        # Extract resolution from filename (e.g., "10cm_evaluation_1.tif" -> 10)
        if filename.startswith('10cm'):
            cm_resolution = 10
        elif filename.startswith('20cm'):
            cm_resolution = 20
        elif filename.startswith('40cm'):
            cm_resolution = 40
        else:
            cm_resolution = 10  # default
        
        # For scene_type, we'll need to infer or use a default
        # You might need to adjust this based on your dataset knowledge
        scene_type_map = {
            10: "agriculture_plantation",
            20: "mixed_forest",
            40: "industrial_area"
        }
        scene_type = scene_type_map.get(cm_resolution, "mixed_forest")
        
        return cm_resolution, scene_type
    
    def polygon_to_segmentation(self, polygon):
        """Convert polygon coordinates to segmentation format"""
        # Flatten polygon coordinates to match the expected format
        if len(polygon.shape) == 2:
            return polygon.flatten().tolist()
        return polygon.tolist()
    
    def run_inference(self, confidence_threshold=0.25, iou_threshold=0.45):
        """Run inference on all evaluation images"""
        
        # Get all evaluation images
        image_files = sorted([f for f in os.listdir(self.evaluation_images_dir) 
                             if f.endswith('.tif')])
        
        results_data = {"images": []}
        
        print(f"Processing {len(image_files)} evaluation images...")
        
        for i, image_file in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_file}")
            
            image_path = self.evaluation_images_dir / image_file
            
            # Load image to get dimensions
            img = Image.open(image_path)
            width, height = img.size
            
            # Extract metadata from filename
            cm_resolution, scene_type = self.extract_scene_info(image_file)
            
            # Run YOLO inference
            results = self.model.predict(
                source=str(image_path),
                conf=confidence_threshold,
                iou=iou_threshold,
                save=False,
                verbose=False
            )
            
            # Process results
            annotations = []
            
            if results and len(results) > 0:
                result = results[0]  # Get first result
                
                if result.masks is not None:
                    # Process each detection
                    for j in range(len(result.boxes)):
                        # Get confidence score
                        confidence = float(result.boxes.conf[j])
                        
                        # Get class
                        class_id = int(result.boxes.cls[j])
                        class_name = self.class_names.get(class_id, 'unknown')
                        
                        # Get mask and convert to polygon
                        mask = result.masks.xy[j]  # Get polygon coordinates
                        
                        if len(mask) > 0:
                            # Convert to segmentation format (flatten coordinates)
                            segmentation = self.polygon_to_segmentation(mask)
                            
                            # Ensure we have valid coordinates
                            if len(segmentation) >= 6:  # At least 3 points (x,y pairs)
                                annotation = {
                                    "class": class_name,
                                    "confidence_score": round(confidence, 2),
                                    "segmentation": [round(coord, 1) for coord in segmentation]
                                }
                                annotations.append(annotation)
            
            # Create image result
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
        
        print(f"\nInference completed!")
        print(f"Results saved to: {self.output_path}")
        print(f"Total images processed: {len(image_files)}")
        
        # Print summary statistics
        total_annotations = sum(len(img["annotations"]) for img in results_data["images"])
        print(f"Total annotations generated: {total_annotations}")
        
        # Count by class
        class_counts = {}
        for img in results_data["images"]:
            for ann in img["annotations"]:
                class_name = ann["class"]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print("Annotations by class:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")
        
        return results_data

def main():
    """Main function to run inference"""
    
    # Paths
    evaluation_images_dir = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/evaluation_images"
    output_path = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/optimized_predictions.json"
    
    # Model path - update this after training
    model_path = "/Users/ananyasingla/Downloads/PLAKSHA/SEM 5/DL/DL_Project/runs/segment/tree_segmentation2/weights/best.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Please train the model first using train_yolo.py")
        print("Or update the model_path variable to point to your trained model")
        return
    
    # Check if evaluation images exist
    if not os.path.exists(evaluation_images_dir):
        print(f"Evaluation images not found at: {evaluation_images_dir}")
        print("Please extract evaluation_images.zip first")
        return
    
    # Run inference
    inference = YOLOInference(model_path, evaluation_images_dir, output_path)
    
    # OPTIMIZED thresholds based on threshold optimization results
    confidence_threshold = 0.1   # Optimized: Lower threshold for more detections
    iou_threshold = 0.6          # Optimized: Higher IoU for better precision
    
    results = inference.run_inference(confidence_threshold, iou_threshold)
    
    print("\nInference completed successfully!")
    print(f"Predictions saved to: {output_path}")

if __name__ == "__main__":
    main()